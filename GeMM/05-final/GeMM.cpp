// GeMM.cpp - Modern C++ NEON-optimized Ternary-Binary GeMM implementation
#include "GeMM.hpp"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <bit>

namespace tbn {

// ============================================================================
// NEON helpers
// ============================================================================

namespace detail {

[[gnu::always_inline]]
inline std::uint64_t load_u64(const std::uint8_t* __restrict__ p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(std::uint64_t));
    return v;
}

} // namespace detail

// ============================================================================
// Matrix packing
// ============================================================================

TernaryMatrix TernaryMatrix::pack(
    std::span<const std::int8_t> data,
    std::uint32_t rows,
    std::uint32_t cols
) {
    if (rows % 8 != 0) {
        throw std::invalid_argument("TernaryMatrix: rows must be multiple of 8");
    }
    if (cols % 64 != 0) {
        throw std::invalid_argument("TernaryMatrix: cols must be multiple of 64");
    }
    if (data.size() != static_cast<std::size_t>(rows) * cols) {
        throw std::invalid_argument("TernaryMatrix: data size doesn't match dimensions");
    }

    const std::uint32_t rowBytes = cols / 8;
    const std::size_t totalBytes = static_cast<std::size_t>(rows) * rowBytes;

    std::vector<std::uint8_t> positive(totalBytes, 0);
    std::vector<std::uint8_t> negative(totalBytes, 0);

    for (std::uint32_t i = 0; i < rows; ++i) {
        auto* rowP = positive.data() + i * rowBytes;
        auto* rowM = negative.data() + i * rowBytes;
        for (std::uint32_t j = 0; j < cols; ++j) {
            const std::int8_t v = data[i * cols + j];
            const std::uint32_t byteIdx = j / 8;
            const std::uint8_t bit = static_cast<std::uint8_t>(1u << (j & 7));

            switch (v) {
                case 1:  rowP[byteIdx] |= bit; break;
                case -1: rowM[byteIdx] |= bit; break;
                case 0:  break;
                default:
                    throw std::invalid_argument("TernaryMatrix: values must be -1, 0, or +1");
            }
        }
    }

    return TernaryMatrix(std::move(positive), std::move(negative), rows, cols);
}

TernaryMatrix TernaryMatrix::pack_from_float(
    const float* __restrict__ data,
    std::uint32_t rows,
    std::uint32_t cols,
    float threshold_low,
    float threshold_high
) {
    if (rows % 8 != 0) {
        throw std::invalid_argument("TernaryMatrix: rows must be multiple of 8");
    }
    if (cols % 64 != 0) {
        throw std::invalid_argument("TernaryMatrix: cols must be multiple of 64");
    }

    const std::uint32_t rowBytes = cols / 8;
    const std::size_t totalBytes = static_cast<std::size_t>(rows) * rowBytes;

    std::vector<std::uint8_t> positive(totalBytes, 0);
    std::vector<std::uint8_t> negative(totalBytes, 0);

    // Fused: quantize float -> ternary AND pack into bit-planes in one pass
    for (std::uint32_t i = 0; i < rows; ++i) {
        auto* rowP = positive.data() + i * rowBytes;
        auto* rowM = negative.data() + i * rowBytes;
        const float* rowData = data + i * cols;

        for (std::uint32_t j = 0; j < cols; ++j) {
            const float v = rowData[j];
            const std::uint32_t byteIdx = j / 8;
            const std::uint8_t bit = static_cast<std::uint8_t>(1u << (j & 7));

            if (v < threshold_low) {
                rowM[byteIdx] |= bit;  // -1
            } else if (v > threshold_high) {
                rowP[byteIdx] |= bit;  // +1
            }
            // else: 0, both bits remain 0
        }
    }

    return TernaryMatrix(std::move(positive), std::move(negative), rows, cols);
}

BinaryMatrix BinaryMatrix::pack(
    std::span<const std::int8_t> data,
    std::uint32_t rows,
    std::uint32_t cols
) {
    if (rows % 64 != 0) {
        throw std::invalid_argument("BinaryMatrix: rows must be multiple of 64");
    }
    if (cols % 8 != 0) {
        throw std::invalid_argument("BinaryMatrix: cols must be multiple of 8");
    }
    if (data.size() != static_cast<std::size_t>(rows) * cols) {
        throw std::invalid_argument("BinaryMatrix: data size doesn't match dimensions");
    }

    const std::uint32_t colBytes = rows / 8;
    const std::size_t totalBytes = static_cast<std::size_t>(cols) * colBytes;

    std::vector<std::uint8_t> bits(totalBytes, 0);

    for (std::uint32_t j = 0; j < cols; ++j) {
        auto* col = bits.data() + j * colBytes;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::int8_t v = data[r * cols + j];
            const std::uint32_t byteIdx = r / 8;
            const std::uint8_t bit = static_cast<std::uint8_t>(1u << (r & 7));

            switch (v) {
                case 1:  break;
                case -1: col[byteIdx] |= bit; break;
                default:
                    throw std::invalid_argument("BinaryMatrix: values must be -1 or +1");
            }
        }
    }

    return BinaryMatrix(std::move(bits), rows, cols);
}

// ============================================================================
// Microkernel (Hybrid NEON + scalar popcount)
// ============================================================================

namespace detail {

[[gnu::always_inline, gnu::hot]]
inline void MicrokernelTBN_NEON(
    const std::uint8_t* __restrict__ AblockP,
    const std::uint8_t* __restrict__ AblockM,
    const std::uint8_t* __restrict__ Bblock,
    std::uint32_t mmk,
    std::uint32_t nmk,
    std::uint32_t keff,
    std::int32_t* __restrict__ Ctile,
    std::uint32_t CtileStride
) noexcept {
    const std::uint32_t rowBytesAblk = keff / 8;
    const std::uint32_t colBytesBblk = keff / 8;

    for (std::uint32_t r = 0; r < mmk; ++r) {
        const std::uint8_t* __restrict__ ArowP = AblockP + r * rowBytesAblk;
        const std::uint8_t* __restrict__ ArowM = AblockM + r * rowBytesAblk;

        for (std::uint32_t c = 0; c < nmk; ++c) {
            const std::uint8_t* __restrict__ Bcol = Bblock + c * colBytesBblk;

            int posCount = 0;
            int negCount = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
            // NEON path: process 128 bits (16 bytes) at a time using SIMD
            const std::uint8_t* __restrict__ ap = ArowP;
            const std::uint8_t* __restrict__ am = ArowM;
            const std::uint8_t* __restrict__ bc = Bcol;

            // Accumulators for popcount across chunks
            uint8x16_t posAcc = vdupq_n_u8(0);
            uint8x16_t negAcc = vdupq_n_u8(0);

            std::uint32_t chunks128 = rowBytesAblk / 16;
            for (std::uint32_t w = 0; w < chunks128; ++w) {
                // Load 128 bits from each bit-plane
                uint8x16_t vAp = vld1q_u8(ap + w * 16);
                uint8x16_t vAm = vld1q_u8(am + w * 16);
                uint8x16_t vBc = vld1q_u8(bc + w * 16);

                // Ternary logic: pos = (ap | bc) & (am | ~bc), neg = (ap | ~bc) & (am | bc)
                uint8x16_t vNotBc = vmvnq_u8(vBc);
                uint8x16_t posMask = vandq_u8(vorrq_u8(vAp, vBc), vorrq_u8(vAm, vNotBc));
                uint8x16_t negMask = vandq_u8(vorrq_u8(vAp, vNotBc), vorrq_u8(vAm, vBc));

                // Hardware popcount per byte: vcntq_u8
                posAcc = vaddq_u8(posAcc, vcntq_u8(posMask));
                negAcc = vaddq_u8(negAcc, vcntq_u8(negMask));
            }

            // Horizontal add across the accumulator vectors
            posCount += vaddlvq_u8(posAcc);
            negCount += vaddlvq_u8(negAcc);

            // Handle remainder (less than 16 bytes) with scalar path
            std::uint32_t processed128 = chunks128 * 16;
            std::uint32_t remaining = rowBytesAblk - processed128;
            if (remaining > 0) {
                const std::uint8_t* __restrict__ ap_rem = ArowP + processed128;
                const std::uint8_t* __restrict__ am_rem = ArowM + processed128;
                const std::uint8_t* __restrict__ bc_rem = Bcol + processed128;

                std::uint64_t ap64 = 0, am64 = 0, bc64 = 0;
                std::memcpy(&ap64, ap_rem, remaining);
                std::memcpy(&am64, am_rem, remaining);
                std::memcpy(&bc64, bc_rem, remaining);

                std::uint64_t posMask = (ap64 | bc64) & (am64 | ~bc64);
                std::uint64_t negMask = (ap64 | ~bc64) & (am64 | bc64);

                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }
#else
            // Scalar fallback: process 64 bits at a time
            const std::uint32_t chunks64 = rowBytesAblk / 8;
            for (std::uint32_t w = 0; w < chunks64; ++w) {
                const std::uint64_t ap = load_u64(ArowP + w * 8);
                const std::uint64_t am = load_u64(ArowM + w * 8);
                const std::uint64_t bc = load_u64(Bcol  + w * 8);

                const std::uint64_t posMask = (ap | bc) & (am | ~bc);
                const std::uint64_t negMask = (ap | ~bc) & (am | bc);

                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }
#endif

            Ctile[r * CtileStride + c] += (posCount - negCount);
        }
    }
}

inline void PackA_SubBlock_RowMajor(
    const std::uint8_t* __restrict__ Ap,
    const std::uint8_t* __restrict__ Am,
    std::uint32_t n,
    std::uint32_t y,
    std::uint32_t meff,
    std::uint32_t d,
    std::uint32_t keff,
    std::uint8_t* __restrict__ AblockP,
    std::uint8_t* __restrict__ AblockM
) noexcept {
    const std::uint32_t rowBytesA = n / 8;
    const std::uint32_t subRowBytes = keff / 8;

    for (std::uint32_t r = 0; r < meff; ++r) {
        const std::uint8_t* srcP = Ap + (y + r) * rowBytesA + (d / 8);
        const std::uint8_t* srcM = Am + (y + r) * rowBytesA + (d / 8);
        std::uint8_t* dstP = AblockP + r * subRowBytes;
        std::uint8_t* dstM = AblockM + r * subRowBytes;
        std::memcpy(dstP, srcP, subRowBytes);
        std::memcpy(dstM, srcM, subRowBytes);
    }
}

inline void PackB_SubBlock_ColMajor(
    const std::uint8_t* __restrict__ B,
    std::uint32_t n,
    std::uint32_t x,
    std::uint32_t neff,
    std::uint32_t d,
    std::uint32_t keff,
    std::uint8_t* __restrict__ Bblock
) noexcept {
    const std::uint32_t colBytesB = n / 8;
    const std::uint32_t subColBytes = keff / 8;

    for (std::uint32_t c = 0; c < neff; ++c) {
        const std::uint8_t* src = B + (x + c) * colBytesB + (d / 8);
        std::uint8_t* dst = Bblock + c * subColBytes;
        std::memcpy(dst, src, subColBytes);
    }
}

} // namespace detail

// ============================================================================
// GeMM Engine implementation
// ============================================================================

Int32Matrix GemmEngine::compute(
    TernaryMatrixView A,
    BinaryMatrixView B,
    const TilingParams& params
) {
    const std::uint32_t m = A.rows;
    const std::uint32_t n = A.cols;
    const std::uint32_t k = B.cols;

    params.validate(m, n, k);

    Int32Matrix C(m, k);
    std::memset(C.data().data(), 0, static_cast<std::size_t>(m) * k * sizeof(std::int32_t));

    const std::uint32_t subRowBytes = params.kblk / 8;
    std::vector<std::uint8_t> AblockP(static_cast<std::size_t>(params.mmk) * subRowBytes);
    std::vector<std::uint8_t> AblockM(static_cast<std::size_t>(params.mmk) * subRowBytes);
    std::vector<std::uint8_t> Bblock(static_cast<std::size_t>(params.nmk) * subRowBytes);

    for (std::uint32_t y = 0; y < m; y += params.mblk) {
        const std::uint32_t meff_outer = std::min(params.mblk, m - y);

        for (std::uint32_t x = 0; x < k; x += params.nblk) {
            const std::uint32_t neff_outer = std::min(params.nblk, k - x);

            for (std::uint32_t d = 0; d < n; d += params.kblk) {
                const std::uint32_t keff_outer = std::min(params.kblk, n - d);

                for (std::uint32_t r = 0; r < meff_outer; r += params.mmk) {
                    const std::uint32_t meff = std::min(params.mmk, meff_outer - r);

                    detail::PackA_SubBlock_RowMajor(
                        A.positive.data(), A.negative.data(), n,
                        y + r, meff, d, keff_outer,
                        AblockP.data(), AblockM.data()
                    );

                    for (std::uint32_t c = 0; c < neff_outer; c += params.nmk) {
                        const std::uint32_t neff = std::min(params.nmk, neff_outer - c);

                        detail::PackB_SubBlock_ColMajor(
                            B.bits.data(), n,
                            x + c, neff, d, keff_outer,
                            Bblock.data()
                        );

                        std::int32_t* Ctile = C.data().data() + (y + r) * k + (x + c);
                        detail::MicrokernelTBN_NEON(
                            AblockP.data(), AblockM.data(), Bblock.data(),
                            meff, neff, keff_outer,
                            Ctile, k
                        );
                    }
                }
            }
        }
    }

    last_flops_ = static_cast<std::uint64_t>(m) * n * k * 2;
    return C;
}

// ============================================================================
// Pre-packed weights implementation
// ============================================================================

PackedBinaryWeights PackedBinaryWeights::from_int8(
    std::span<const std::int8_t> weights,
    std::uint32_t rows_raw,
    std::uint32_t cols_raw,
    const TilingParams& tiling
) {
    // Pad rows to multiple of 64 (BinaryMatrix requirement) and cols to multiple of 8
    std::uint32_t rows_padded = ((rows_raw + 63) / 64) * 64;
    std::uint32_t cols_padded = ((cols_raw + 7) / 8) * 8;

    // Pad weight data
    std::vector<std::int8_t> padded(static_cast<std::size_t>(rows_padded) * cols_padded, static_cast<std::int8_t>(1));
    for (std::uint32_t i = 0; i < rows_raw; ++i) {
        for (std::uint32_t j = 0; j < cols_raw; ++j) {
            padded[i * cols_padded + j] = weights[i * cols_raw + j];
        }
    }

    PackedBinaryWeights result;
    result.b_packed = BinaryMatrix::pack(
        std::span<const std::int8_t>(padded.data(), padded.size()),
        rows_padded, cols_padded
    );
    result.tiling = tiling;
    result.original_m = rows_raw;
    result.original_n = cols_raw;
    result.original_k = rows_raw;  // stored as m,n for GEMM

    return result;
}

} // namespace tbn
