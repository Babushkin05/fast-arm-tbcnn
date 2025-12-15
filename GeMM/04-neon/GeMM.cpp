// tbn_blocked_gemm.hpp
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <bit>
#include <utility>
#include "GeMM.hpp"

static inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(uint64_t));
    return v;
}

// Microkernel: compute a mmk x nmk.
// AblockP/AblockM: layout is row-major by mmk rows, contiguous columns across keff,
// each row has keff bits -> keff/8 bytes, processed in 64-bit words.
// Bblock: column-major by nmk cols, each col has keff bits -> keff/8 bytes.
static inline void MicrokernelTBN_noSIMD(
    const uint8_t* AblockP, const uint8_t* AblockM,
    const uint8_t* Bblock,
    uint32_t mmk, uint32_t nmk, uint32_t keff,
    uint8_t* CplusTile, uint8_t* CminusTile,
    uint32_t CRowBytes // bytes per row in full C
) {
    const uint32_t rowBytesAblk = keff / 8; // bytes per row in Ablock
    const uint32_t colBytesBblk = keff / 8; // bytes per col in Bblock
    const uint32_t chunks64 = rowBytesAblk / 8; // keff must be multiple of 64

    for (uint32_t r = 0; r < mmk; ++r) {
        const uint8_t* ArowP = AblockP + r * rowBytesAblk;
        const uint8_t* ArowM = AblockM + r * rowBytesAblk;

        for (uint32_t c = 0; c < nmk; ++c) {
            const uint8_t* Bcol = Bblock + c * colBytesBblk;

            int posCount = 0;
            int negCount = 0;

            for (uint32_t w = 0; w < chunks64; ++w) {
                const uint64_t ap = load_u64(ArowP + w * 8);
                const uint64_t am = load_u64(ArowM + w * 8);
                const uint64_t bc = load_u64(Bcol  + w * 8);

                const uint64_t posMask = (ap | bc) & (am | ~bc);
                const uint64_t negMask = (ap | ~bc) & (am | bc);

                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }

            const int diff = posCount - negCount;

            if (diff > 0) {
                CplusTile[r * nmk + c] = 1;
            } else if (diff < 0) {
                CminusTile[r * nmk + c] = 1;
            }
        }
    }
}

// Pack A sub-block (rows [y:y+meff), keff columns) into contiguous row-major buffers for microkernel.
static inline void PackA_SubBlock_RowMajor(
    const uint8_t* Ap, const uint8_t* Am,
    uint32_t n, // full A width
    uint32_t y, uint32_t meff,
    uint32_t d, uint32_t keff,
    uint8_t* AblockP, uint8_t* AblockM
) {
    const uint32_t rowBytesA = n / 8;
    const uint32_t subRowBytes = keff / 8;

    // Copy bit-range [d : d+keff) from each row into Ablock buffers
    for (uint32_t r = 0; r < meff; ++r) {
        const uint8_t* srcP = Ap + (y + r) * rowBytesA + (d / 8);
        const uint8_t* srcM = Am + (y + r) * rowBytesA + (d / 8);
        uint8_t* dstP = AblockP + r * subRowBytes;
        uint8_t* dstM = AblockM + r * subRowBytes;
        std::memcpy(dstP, srcP, subRowBytes);
        std::memcpy(dstM, srcM, subRowBytes);
    }
}

// Pack B sub-block (columns [x:x+neff), keff rows) into contiguous column-major buffer for microkernel.
static inline void PackB_SubBlock_ColMajor(
    const uint8_t* B,
    uint32_t n, // full B height (rows)
    uint32_t x, uint32_t neff,
    uint32_t d, uint32_t keff,
    uint8_t* Bblock
) {
    const uint32_t colBytesB = n / 8;
    const uint32_t subColBytes = keff / 8;

    for (uint32_t c = 0; c < neff; ++c) {
        const uint8_t* src = B + (x + c) * colBytesB + (d / 8);
        uint8_t* dst = Bblock + c * subColBytes;
        std::memcpy(dst, src, subColBytes);
    }
}

// A - m x n, B - n x k, tp parameters of cache; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is ternary
// n mod 64 == 0
// m mod 8 == 0
// k mod 8 == 0
std::pair<std::unique_ptr<uint8_t[]>, std::unique_ptr<uint8_t[]>> GemmTBN_Blocked(
    std::pair<std::unique_ptr<const uint8_t[]>, std::unique_ptr<const uint8_t[]>> A,
    std::unique_ptr<const uint8_t[]> B,
    uint32_t m, uint32_t n, uint32_t k,
    const TilingParams& tp
) {
    // Extract raw pointers from RAII containers for internal use (no ownership transfer here).
    const uint8_t* Ap = A.first ? A.first.get() : nullptr;
    const uint8_t* Am = A.second? A.second.get() : nullptr;
    const uint8_t* Bptr = B ? B.get() : nullptr;
    // Constraints for simple pipeline
    if ((m % tp.mmk) || (k % tp.nmk) || (n % 64)) {
        throw std::invalid_argument("GemmTBN_Blocked: m%mmk==0, k%nmk==0, n%64==0 required");
    }

    const uint32_t rowBytesA = n / 8;
    const uint32_t colBytesB = n / 8;
    const uint32_t rowBytesC = k / 8;

    // Output bit-planes
    std::unique_ptr<uint8_t[]> Cplus(new uint8_t[static_cast<size_t>(m) * rowBytesC]);
    std::unique_ptr<uint8_t[]> Cminus(new uint8_t[static_cast<size_t>(m) * rowBytesC]);
    std::memset(Cplus.get(),  0, static_cast<size_t>(m) * rowBytesC);
    std::memset(Cminus.get(), 0, static_cast<size_t>(m) * rowBytesC);

    // Buffers sized by inner L1-friendly tiles
    const uint32_t subRowBytes = tp.kblk / 8; // keff <= kblk; we'll use keff chunks
    // Allocate max-sized sub-blocks
    uint8_t* AblockP = new uint8_t[static_cast<size_t>(tp.mmk) * (tp.kblk / 8)];
    uint8_t* AblockM = new uint8_t[static_cast<size_t>(tp.mmk) * (tp.kblk / 8)];
    uint8_t* Bblock  = new uint8_t[static_cast<size_t>(tp.nmk) * (tp.kblk / 8)];

    // Temporary sign buffers for one microkernel call (mmk x nmk)
    uint8_t* CplusTileTemp  = new uint8_t[static_cast<size_t>(tp.mmk) * tp.nmk];
    uint8_t* CminusTileTemp = new uint8_t[static_cast<size_t>(tp.mmk) * tp.nmk];

    for (uint32_t y = 0; y < m; y += tp.mblk) {
        const uint32_t meff_outer = (y + tp.mblk <= m) ? tp.mblk : (m - y);

        for (uint32_t x = 0; x < k; x += tp.nblk) {
            const uint32_t neff_outer = (x + tp.nblk <= k) ? tp.nblk : (k - x);

            for (uint32_t d = 0; d < n; d += tp.kblk) {
                const uint32_t keff_outer = (d + tp.kblk <= n) ? tp.kblk : (n - d);

                // Iterate microkernel tiles within outer blocks
                for (uint32_t r = 0; r < meff_outer; r += tp.mmk) {
                    const uint32_t meff = (r + tp.mmk <= meff_outer) ? tp.mmk : (meff_outer - r);

                    // Pack A sub-block rows [y+r : y+r+meff), columns [d : d+keff_outer)
                    PackA_SubBlock_RowMajor(
                        Ap, Am, n, y + r, meff, d, keff_outer,
                        AblockP, AblockM
                    );

                    for (uint32_t c = 0; c < neff_outer; c += tp.nmk) {
                        const uint32_t neff = (c + tp.nmk <= neff_outer) ? tp.nmk : (neff_outer - c);

                        // Pack B sub-block columns [x+c : x+c+neff), rows [d : d+keff_outer)
                        PackB_SubBlock_ColMajor(
                            Bptr, n, x + c, neff, d, keff_outer,
                            Bblock
                        );

                        // Run microkernel for mmk x nmk tile
                        MicrokernelTBN_noSIMD(
                            AblockP, AblockM, Bblock,
                            meff, neff, keff_outer,
                            CplusTileTemp, CminusTileTemp,
                            rowBytesC
                        );

                        // Scatter temp signs into bit-planes of C at position (y+r, x+c)
                        for (uint32_t tr = 0; tr < meff; ++tr) {
                            uint8_t* CrowP = Cplus.get()  + (y + r + tr) * rowBytesC;
                            uint8_t* CrowM = Cminus.get() + (y + r + tr) * rowBytesC;
                            for (uint32_t tc = 0; tc < neff; ++tc) {
                                const uint8_t sp = CplusTileTemp [tr * neff + tc];
                                const uint8_t sm = CminusTileTemp[tr * neff + tc];
                                const uint32_t col = x + c + tc;
                                const uint32_t byteIdx = col / 8;
                                const uint8_t  bitMask = static_cast<uint8_t>(1u << (col & 7));
                                if (sp) {
                                    CrowP[byteIdx] |= bitMask;
                                } else if (sm) {
                                    CrowM[byteIdx] |= bitMask;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    delete[] AblockP;
    delete[] AblockM;
    delete[] Bblock;
    delete[] CplusTileTemp;
    delete[] CminusTileTemp;

    return std::make_pair(std::move(Cplus), std::move(Cminus));
}


std::unique_ptr<int8_t[]>
UnpackTernaryRowMajor(const uint8_t* Cplus, const uint8_t* Cminus,
                      uint32_t m, uint32_t k)
{
    if ((m % 8) || (k % 8))
        throw std::invalid_argument("UnpackTernary: m,k must be multiples of 8");
    if (!Cplus || !Cminus)
        throw std::invalid_argument("UnpackTernary: null pointer");

    const uint32_t rowBytes = k / 8;
    std::unique_ptr<int8_t[]> C(new int8_t[static_cast<size_t>(m) * k]);

    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t* rowP = Cplus + i * rowBytes;
        const uint8_t* rowM = Cminus + i * rowBytes;
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t byteIdx = j / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            bool p = (rowP[byteIdx] & bit) != 0;
            bool mbit = (rowM[byteIdx] & bit) != 0;
            if (p && !mbit)
                C[i * k + j] = +1;
            else if (!p && mbit)
                C[i * k + j] = -1;
            else if (!p && !mbit)
                C[i * k + j] = 0;
            else
                throw std::runtime_error("UnpackTernary: invalid (1,1) state");
        }
    }

    return C;
}