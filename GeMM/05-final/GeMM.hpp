#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <stdexcept>

namespace tbn {

// ============================================================================
// Configuration
// ============================================================================

/// Tiling parameters for cache-optimized blocked GeMM
struct TilingParams {
    std::uint32_t mblk;  ///< outer block on rows (L2)
    std::uint32_t nblk;  ///< outer block on cols (L2)
    std::uint32_t kblk;  ///< outer block on depth (L2)
    std::uint32_t mmk;   ///< microkernel rows (L1)
    std::uint32_t nmk;   ///< microkernel cols (L1)

    /// Validate parameters for NEON execution
    void validate(std::uint32_t m, std::uint32_t n, std::uint32_t k) const {
        if (n % 128 != 0) {
            throw std::invalid_argument("TilingParams: n must be multiple of 128 for NEON");
        }
        if (kblk % 128 != 0) {
            throw std::invalid_argument("TilingParams: kblk must be multiple of 128 for NEON");
        }
        if (m % mmk != 0) {
            throw std::invalid_argument("TilingParams: m must be multiple of mmk");
        }
        if (k % nmk != 0) {
            throw std::invalid_argument("TilingParams: k must be multiple of nmk");
        }
    }

    /// Default parameters for 128x128 matrices
    static constexpr TilingParams default_128x128() noexcept {
        return {.mblk = 64, .nblk = 64, .kblk = 128, .mmk = 32, .nmk = 32};
    }
};

// ============================================================================
// Matrix types
// ============================================================================

/// Non-owning view of a packed ternary matrix (two bit-planes)
struct TernaryMatrixView {
    std::span<const std::uint8_t> positive;  ///< bits for +1 values
    std::span<const std::uint8_t> negative;  ///< bits for -1 values
    std::uint32_t rows{};
    std::uint32_t cols{};
};

/// Non-owning view of a packed binary matrix (single bit-plane)
struct BinaryMatrixView {
    std::span<const std::uint8_t> bits;
    std::uint32_t rows{};
    std::uint32_t cols{};
};

/// Owning packed ternary matrix with automatic memory management
class TernaryMatrix {
public:
    TernaryMatrix() = default;

    /// Pack from int8_t ternary values {-1, 0, +1}
    [[nodiscard]] static TernaryMatrix pack(
        std::span<const std::int8_t> data,
        std::uint32_t rows,
        std::uint32_t cols
    );

    /// Fused: quantize float directly to packed ternary (no intermediate int8_t buffer)
    /// threshold_low/high: values in (low, high) become 0, below low become -1, above high become +1
    [[nodiscard]] static TernaryMatrix pack_from_float(
        const float* __restrict__ data,
        std::uint32_t rows,
        std::uint32_t cols,
        float threshold_low = -0.1f,
        float threshold_high = 0.1f
    );

    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t cols() const noexcept { return cols_; }
    [[nodiscard]] TernaryMatrixView view() const noexcept {
        return {positive_, negative_, rows_, cols_};
    }
    [[nodiscard]] bool empty() const noexcept { return positive_.empty(); }

private:
    TernaryMatrix(std::vector<std::uint8_t> positive,
                  std::vector<std::uint8_t> negative,
                  std::uint32_t rows, std::uint32_t cols)
        : positive_(std::move(positive)),
          negative_(std::move(negative)),
          rows_(rows), cols_(cols) {}

    std::vector<std::uint8_t> positive_;
    std::vector<std::uint8_t> negative_;
    std::uint32_t rows_{};
    std::uint32_t cols_{};
};

/// Owning packed binary matrix with automatic memory management
class BinaryMatrix {
public:
    BinaryMatrix() = default;

    /// Pack from int8_t binary values {-1, +1}
    [[nodiscard]] static BinaryMatrix pack(
        std::span<const std::int8_t> data,
        std::uint32_t rows,
        std::uint32_t cols
    );

    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t cols() const noexcept { return cols_; }
    [[nodiscard]] BinaryMatrixView view() const noexcept {
        return {bits_, rows_, cols_};
    }
    [[nodiscard]] bool empty() const noexcept { return bits_.empty(); }

private:
    BinaryMatrix(std::vector<std::uint8_t> bits,
                 std::uint32_t rows, std::uint32_t cols)
        : bits_(std::move(bits)), rows_(rows), cols_(cols) {}

    std::vector<std::uint8_t> bits_;
    std::uint32_t rows_{};
    std::uint32_t cols_{};
};

/// Owning int32 matrix with element access
class Int32Matrix {
public:
    Int32Matrix() = default;

    explicit Int32Matrix(std::uint32_t rows, std::uint32_t cols)
        : data_(static_cast<std::size_t>(rows) * cols, 0),
          rows_(rows), cols_(cols) {}

    [[nodiscard]] std::int32_t& at(std::uint32_t row, std::uint32_t col) noexcept {
        return data_[static_cast<std::size_t>(row) * cols_ + col];
    }

    [[nodiscard]] std::int32_t at(std::uint32_t row, std::uint32_t col) const noexcept {
        return data_[static_cast<std::size_t>(row) * cols_ + col];
    }

    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::span<std::int32_t> data() noexcept { return data_; }
    [[nodiscard]] std::span<const std::int32_t> data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    /// Get sign of value as ternary {-1, 0, +1}
    [[nodiscard]] std::int8_t sign_at(std::uint32_t row, std::uint32_t col) const noexcept {
        const auto v = at(row, col);
        return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
    }

private:
    std::vector<std::int32_t> data_;
    std::uint32_t rows_{};
    std::uint32_t cols_{};
};

// ============================================================================
// Pre-packed weights — single-shot packing for inference reuse
// ============================================================================

/// Pre-packed binary weights: owned matrices + tiling parameters
struct PackedBinaryWeights {
    TernaryMatrix a_packed;   // if activations are ternary (cache reuses this as template)
    BinaryMatrix b_packed;    // pre-packed weight matrix
    TilingParams tiling;
    std::uint32_t original_m{};
    std::uint32_t original_n{};
    std::uint32_t original_k{};

    [[nodiscard]] bool valid() const noexcept { return !b_packed.empty(); }

    /// Pack weights once; returns empty struct on failure
    [[nodiscard]] static PackedBinaryWeights from_int8(
        std::span<const std::int8_t> weights,
        std::uint32_t rows_raw,    // == K
        std::uint32_t cols_raw,    // == N
        const TilingParams& tiling
    );
};

// ============================================================================
// GeMM Engine
// ============================================================================

/// Ternary-Binary GeMM computation engine with NEON optimization
///
/// Computes C = A @ B where:
/// - A is ternary {-1, 0, +1} encoded as two bit-planes
/// - B is binary {-1, +1} encoded as one bit-plane
/// - C is int32 accumulated diff values
class GemmEngine {
public:
    GemmEngine() = default;

    /// Perform blocked GeMM with NEON optimization
    [[nodiscard]] Int32Matrix compute(
        TernaryMatrixView A,
        BinaryMatrixView B,
        const TilingParams& params
    );

    /// Perform blocked GeMM with default tiling parameters
    [[nodiscard]] Int32Matrix compute(
        TernaryMatrixView A,
        BinaryMatrixView B
    ) {
        return compute(A, B, TilingParams::default_128x128());
    }

    [[nodiscard]] std::uint64_t last_flops() const noexcept { return last_flops_; }

private:
    std::uint64_t last_flops_{};
};

} // namespace tbn
