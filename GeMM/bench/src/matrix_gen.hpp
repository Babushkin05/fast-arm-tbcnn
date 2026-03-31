#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <stdexcept>

namespace bench {

/// Types of matrices for benchmarking
enum class MatrixType {
    RandomDense,     // 33% each: -1, 0, +1
    RandomSparse,    // 90% zeros, 5% each ±1
    DenseNoZero,     // 50/50 ±1, no zeros
    Diagonal,        // Only diagonal has ±1
    Banded,          // Band of width k around diagonal
    BlockSparse      // 8x8 blocks either all-zero or all-nonzero
};

/// Parse matrix type from string
MatrixType parse_matrix_type(std::string_view s);

/// Get string name for matrix type
std::string_view matrix_type_name(MatrixType type);

/// Generate ternary matrix (values -1, 0, +1) of given type
/// Output: packed into two bit-planes (positive, negative)
/// Each row has cols/8 bytes, total rows * cols/8 bytes per plane
void generate_ternary_matrix(
    std::uint8_t* positive,
    std::uint8_t* negative,
    std::uint32_t rows,
    std::uint32_t cols,
    MatrixType type,
    std::mt19937& rng
);

/// Generate binary matrix (values -1, +1)
/// Output: packed into single bit-plane, column-major
/// Each column has rows/8 bytes, total cols * rows/8 bytes
void generate_binary_matrix(
    std::uint8_t* bits,
    std::uint32_t rows,
    std::uint32_t cols,
    std::mt19937& rng
);

} // namespace bench
