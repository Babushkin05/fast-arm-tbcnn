// GeMM.cpp - Modern C++ NEON-optimized Ternary-Binary GeMM implementation
#include "GeMM.hpp"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>

namespace tbn {

// ============================================================================
// NEON helpers
// ============================================================================

namespace detail {

[[gnu::always_inline]]
inline uint8x16_t load_u128(const std::uint8_t* __restrict__ p) noexcept {
    return vld1q_u8(p);
}

[[gnu::always_inline, gnu::hot]]
inline int popcount_u8x16(uint8x16_t v) noexcept {
    const uint8x16_t cnt = vcntq_u8(v);
    const uint16x8_t s16 = vpaddlq_u8(cnt);
    const uint32x4_t s32 = vpaddlq_u16(s16);
    const uint64x2_t s64 = vpaddlq_u32(s32);
    return static_cast<int>(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
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
// Microkernel (NEON-optimized)
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
    const std::uint32_t chunks128 = rowBytesAblk / 16;

    for (std::uint32_t r = 0; r < mmk; ++r) {
        const std::uint8_t* __restrict__ ArowP = AblockP + r * rowBytesAblk;
        const std::uint8_t* __restrict__ ArowM = AblockM + r * rowBytesAblk;

        for (std::uint32_t c = 0; c < nmk; ++c) {
            const std::uint8_t* __restrict__ Bcol = Bblock + c * colBytesBblk;

            int posCount = 0;
            int negCount = 0;

            for (std::uint32_t w = 0; w < chunks128; ++w) {
                const uint8x16_t ap = load_u128(ArowP + w * 16);
                const uint8x16_t am = load_u128(ArowM + w * 16);
                const uint8x16_t bc = load_u128(Bcol  + w * 16);

                const uint8x16_t posMask = vandq_u8(
                    vorrq_u8(ap, bc),
                    vorrq_u8(am, vmvnq_u8(bc))
                );
                const uint8x16_t negMask = vandq_u8(
                    vorrq_u8(ap, vmvnq_u8(bc)),
                    vorrq_u8(am, bc)
                );

                posCount += popcount_u8x16(posMask);
                negCount += popcount_u8x16(negMask);
            }

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

} // namespace tbn
