// tbn_blocked_gemm_neon.cpp
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <bit>
#include "GeMM.hpp"
#include <arm_neon.h>

// Load 64 bits scalar
static inline std::uint64_t load_u64(const std::uint8_t* p) {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(std::uint64_t));
    return v;
}

// Microkernel: compute a mmk x nmk tile, accumulating diff values.
// Optimized scalar version with 64-bit processing (like 03-blocked but with tiling).
static inline void MicrokernelTBN_NEON(
    const std::uint8_t* AblockP, const std::uint8_t* AblockM,
    const std::uint8_t* Bblock,
    std::uint32_t mmk, std::uint32_t nmk, std::uint32_t keff,
    std::int32_t* Ctile, std::uint32_t CtileStride
) {
    const std::uint32_t rowBytesAblk = keff / 8;
    const std::uint32_t colBytesBblk = keff / 8;
    const std::uint32_t chunks64 = rowBytesAblk / 8;  // Process 64 bits at a time

    for (std::uint32_t r = 0; r < mmk; ++r) {
        const std::uint8_t* ArowP = AblockP + r * rowBytesAblk;
        const std::uint8_t* ArowM = AblockM + r * rowBytesAblk;

        for (std::uint32_t c = 0; c < nmk; ++c) {
            const std::uint8_t* Bcol = Bblock + c * colBytesBblk;

            int posCount = 0;
            int negCount = 0;

            for (std::uint32_t w = 0; w < chunks64; ++w) {
                // Scalar 64-bit loads (direct to GPR, no NEON overhead)
                const std::uint64_t ap = load_u64(ArowP + w * 8);
                const std::uint64_t am = load_u64(ArowM + w * 8);
                const std::uint64_t bc = load_u64(Bcol  + w * 8);

                // Scalar bitwise operations (very fast on modern CPUs)
                const std::uint64_t posMask = (ap | bc) & (am | ~bc);
                const std::uint64_t negMask = (ap | ~bc) & (am | bc);

                // Hardware popcount (single instruction)
                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }

            Ctile[r * CtileStride + c] += (posCount - negCount);
        }
    }
}

// Pack A sub-block (rows [y:y+meff), keff columns) into contiguous row-major buffers for microkernel.
static inline void PackA_SubBlock_RowMajor(
    const std::uint8_t* Ap, const std::uint8_t* Am,
    std::uint32_t n, // full A width
    std::uint32_t y, std::uint32_t meff,
    std::uint32_t d, std::uint32_t keff,
    std::uint8_t* AblockP, std::uint8_t* AblockM
) {
    const std::uint32_t rowBytesA = n / 8;
    const std::uint32_t subRowBytes = keff / 8;

    // Copy bit-range [d : d+keff) from each row into Ablock buffers
    for (std::uint32_t r = 0; r < meff; ++r) {
        const std::uint8_t* srcP = Ap + (y + r) * rowBytesA + (d / 8);
        const std::uint8_t* srcM = Am + (y + r) * rowBytesA + (d / 8);
        std::uint8_t* dstP = AblockP + r * subRowBytes;
        std::uint8_t* dstM = AblockM + r * subRowBytes;
        std::memcpy(dstP, srcP, subRowBytes);
        std::memcpy(dstM, srcM, subRowBytes);
    }
}

// Pack B sub-block (columns [x:x+neff), keff rows) into contiguous column-major buffer for microkernel.
static inline void PackB_SubBlock_ColMajor(
    const std::uint8_t* B,
    std::uint32_t n, // full B height (rows)
    std::uint32_t x, std::uint32_t neff,
    std::uint32_t d, std::uint32_t keff,
    std::uint8_t* Bblock
) {
    const std::uint32_t colBytesB = n / 8;
    const std::uint32_t subColBytes = keff / 8;

    for (std::uint32_t c = 0; c < neff; ++c) {
        const std::uint8_t* src = B + (x + c) * colBytesB + (d / 8);
        std::uint8_t* dst = Bblock + c * subColBytes;
        std::memcpy(dst, src, subColBytes);
    }
}

// A - m x n, B - n x k, tp parameters of cache; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is int32 (accumulated diff values)
// n mod 128 == 0  (for NEON 128-bit alignment)
// m mod mmk == 0
// k mod nmk == 0
std::int32_t* GemmTBN_Blocked(
    const std::uint8_t* Ap, const std::uint8_t* Am, const std::uint8_t* B,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    const TilingParams& tp
) {
    // Constraints for NEON: n and kblk must be multiple of 128 (for 128-bit chunks)
    if ((m % tp.mmk) || (k % tp.nmk) || (n % 128) || (tp.kblk % 128)) {
        throw std::invalid_argument("GemmTBN_Blocked: m%mmk==0, k%nmk==0, n%128==0, kblk%128==0 required for NEON");
    }

    // Output: int32 matrix m x k
    std::int32_t* C = new std::int32_t[static_cast<size_t>(m) * k];
    std::memset(C, 0, static_cast<size_t>(m) * k * sizeof(std::int32_t));

    // Buffers sized by inner L1-friendly tiles
    const std::uint32_t subRowBytes = tp.kblk / 8; // keff <= kblk; we'll use keff chunks
    // Allocate max-sized sub-blocks
    std::uint8_t* AblockP = new std::uint8_t[static_cast<size_t>(tp.mmk) * subRowBytes];
    std::uint8_t* AblockM = new std::uint8_t[static_cast<size_t>(tp.mmk) * subRowBytes];
    std::uint8_t* Bblock  = new std::uint8_t[static_cast<size_t>(tp.nmk) * subRowBytes];

    for (std::uint32_t y = 0; y < m; y += tp.mblk) {
        const std::uint32_t meff_outer = (y + tp.mblk <= m) ? tp.mblk : (m - y);

        for (std::uint32_t x = 0; x < k; x += tp.nblk) {
            const std::uint32_t neff_outer = (x + tp.nblk <= k) ? tp.nblk : (k - x);

            for (std::uint32_t d = 0; d < n; d += tp.kblk) {
                const std::uint32_t keff_outer = (d + tp.kblk <= n) ? tp.kblk : (n - d);

                // Iterate microkernel tiles within outer blocks
                for (std::uint32_t r = 0; r < meff_outer; r += tp.mmk) {
                    const std::uint32_t meff = (r + tp.mmk <= meff_outer) ? tp.mmk : (meff_outer - r);

                    // Pack A sub-block rows [y+r : y+r+meff), columns [d : d+keff_outer)
                    PackA_SubBlock_RowMajor(
                        Ap, Am, n, y + r, meff, d, keff_outer,
                        AblockP, AblockM
                    );

                    for (std::uint32_t c = 0; c < neff_outer; c += tp.nmk) {
                        const std::uint32_t neff = (c + tp.nmk <= neff_outer) ? tp.nmk : (neff_outer - c);

                        // Pack B sub-block columns [x+c : x+c+neff), rows [d : d+keff_outer)
                        PackB_SubBlock_ColMajor(
                            B, n, x + c, neff, d, keff_outer,
                            Bblock
                        );

                        // Get pointer to the right location in C for this microkernel tile
                        std::int32_t* Ctile = C + (y + r) * k + (x + c);

                        // Run NEON-optimized microkernel, accumulating directly into C
                        MicrokernelTBN_NEON(
                            AblockP, AblockM, Bblock,
                            meff, neff, keff_outer,
                            Ctile, k  // stride is k (width of C)
                        );
                    }
                }
            }
        }
    }

    delete[] AblockP;
    delete[] AblockM;
    delete[] Bblock;

    return C;
}
