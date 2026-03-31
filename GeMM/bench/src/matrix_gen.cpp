#include "matrix_gen.hpp"
#include <cstring>
#include <random>

namespace bench {

MatrixType parse_matrix_type(std::string_view s) {
    if (s == "random_dense") return MatrixType::RandomDense;
    if (s == "random_sparse") return MatrixType::RandomSparse;
    if (s == "dense_no_zero") return MatrixType::DenseNoZero;
    if (s == "diagonal") return MatrixType::Diagonal;
    if (s == "banded") return MatrixType::Banded;
    if (s == "block_sparse") return MatrixType::BlockSparse;
    throw std::invalid_argument("Unknown matrix type: " + std::string(s));
}

std::string_view matrix_type_name(MatrixType type) {
    switch (type) {
        case MatrixType::RandomDense:   return "random_dense";
        case MatrixType::RandomSparse:  return "random_sparse";
        case MatrixType::DenseNoZero:   return "dense_no_zero";
        case MatrixType::Diagonal:      return "diagonal";
        case MatrixType::Banded:        return "banded";
        case MatrixType::BlockSparse:   return "block_sparse";
    }
    return "unknown";
}

void generate_ternary_matrix(
    std::uint8_t* positive,
    std::uint8_t* negative,
    std::uint32_t rows,
    std::uint32_t cols,
    MatrixType type,
    std::mt19937& rng
) {
    const std::uint32_t rowBytes = cols / 8;
    std::memset(positive, 0, static_cast<std::size_t>(rows) * rowBytes);
    std::memset(negative, 0, static_cast<std::size_t>(rows) * rowBytes);

    auto set_bit = [](std::uint8_t* buf, std::uint32_t row, std::uint32_t col, std::uint32_t rowBytes) {
        const std::uint32_t byteIdx = row * rowBytes + col / 8;
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << (col & 7));
        buf[byteIdx] |= bit;
    };

    std::uniform_int_distribution<int> dist3(0, 2);  // for random_dense
    std::uniform_int_distribution<int> dist10(0, 9); // for random_sparse
    std::uniform_int_distribution<int> dist2(0, 1);  // for ±1 choice

    switch (type) {
        case MatrixType::RandomDense: {
            // 33% each: -1, 0, +1
            for (std::uint32_t i = 0; i < rows; ++i) {
                for (std::uint32_t j = 0; j < cols; ++j) {
                    const int v = dist3(rng);
                    if (v == 0) set_bit(positive, i, j, rowBytes);      // +1
                    else if (v == 1) set_bit(negative, i, j, rowBytes); // -1
                    // v == 2 → 0 (both bits remain 0)
                }
            }
            break;
        }

        case MatrixType::RandomSparse: {
            // 90% zeros, 5% each ±1
            for (std::uint32_t i = 0; i < rows; ++i) {
                for (std::uint32_t j = 0; j < cols; ++j) {
                    const int v = dist10(rng);
                    if (v == 0) set_bit(positive, i, j, rowBytes);      // +1 (10%)
                    else if (v == 1) set_bit(negative, i, j, rowBytes); // -1 (10%)
                    // else → 0 (80%)
                }
            }
            break;
        }

        case MatrixType::DenseNoZero: {
            // 50/50 ±1, no zeros
            for (std::uint32_t i = 0; i < rows; ++i) {
                for (std::uint32_t j = 0; j < cols; ++j) {
                    if (dist2(rng) == 0) set_bit(positive, i, j, rowBytes);
                    else set_bit(negative, i, j, rowBytes);
                }
            }
            break;
        }

        case MatrixType::Diagonal: {
            // Only diagonal has ±1
            const std::uint32_t minDim = std::min(rows, cols);
            for (std::uint32_t i = 0; i < minDim; ++i) {
                if (dist2(rng) == 0) set_bit(positive, i, i, rowBytes);
                else set_bit(negative, i, i, rowBytes);
            }
            break;
        }

        case MatrixType::Banded: {
            // Band of width 8 around diagonal
            constexpr std::uint32_t bandwidth = 8;
            for (std::uint32_t i = 0; i < rows; ++i) {
                const std::uint32_t jStart = (i > bandwidth) ? (i - bandwidth) : 0;
                const std::uint32_t jEnd = std::min(i + bandwidth + 1, cols);
                for (std::uint32_t j = jStart; j < jEnd; ++j) {
                    if (dist2(rng) == 0) set_bit(positive, i, j, rowBytes);
                    else set_bit(negative, i, j, rowBytes);
                }
            }
            break;
        }

        case MatrixType::BlockSparse: {
            // 8x8 blocks: 30% are dense, 70% are all-zero
            constexpr std::uint32_t blockSize = 8;
            std::uniform_int_distribution<int> dist_block(0, 9);

            for (std::uint32_t bi = 0; bi < rows; bi += blockSize) {
                for (std::uint32_t bj = 0; bj < cols; bj += blockSize) {
                    if (dist_block(rng) < 3) {
                        // Dense block
                        const std::uint32_t iEnd = std::min(bi + blockSize, rows);
                        const std::uint32_t jEnd = std::min(bj + blockSize, cols);
                        for (std::uint32_t i = bi; i < iEnd; ++i) {
                            for (std::uint32_t j = bj; j < jEnd; ++j) {
                                if (dist2(rng) == 0) set_bit(positive, i, j, rowBytes);
                                else set_bit(negative, i, j, rowBytes);
                            }
                        }
                    }
                    // else: all-zero block
                }
            }
            break;
        }
    }
}

void generate_binary_matrix(
    std::uint8_t* bits,
    std::uint32_t rows,
    std::uint32_t cols,
    std::mt19937& rng
) {
    const std::uint32_t colBytes = rows / 8;
    std::memset(bits, 0, static_cast<std::size_t>(cols) * colBytes);

    std::uniform_int_distribution<int> dist2(0, 1);

    // Column-major: bit at (row, col) is at bits[col * colBytes + row/8] bit (row % 8)
    for (std::uint32_t col = 0; col < cols; ++col) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            // 0 → +1, 1 → -1
            if (dist2(rng) == 1) {
                const std::uint32_t byteIdx = col * colBytes + row / 8;
                const std::uint8_t bit = static_cast<std::uint8_t>(1u << (row & 7));
                bits[byteIdx] |= bit;
            }
        }
    }
}

} // namespace bench
