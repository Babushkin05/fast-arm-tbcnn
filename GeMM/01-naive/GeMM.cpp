#include "GeMM.hpp"
#include <cstring>

// Decode ternary value from two bit-planes
// Returns: +1 if (Ap=1, Am=0), -1 if (Ap=0, Am=1), 0 if (Ap=0, Am=0)
static inline int decode_ternary(const std::uint8_t* Ap, const std::uint8_t* Am,
                                  std::uint32_t row, std::uint32_t col, std::uint32_t rowBytes) {
    const std::uint32_t byteIdx = row * rowBytes + col / 8;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << (col & 7));
    const bool pos = (Ap[byteIdx] & bit) != 0;
    const bool neg = (Am[byteIdx] & bit) != 0;
    return pos ? (neg ? 0 : 1) : (neg ? -1 : 0);
}

// Decode binary value from single bit-plane
// Returns: +1 if bit=0, -1 if bit=1
static inline int decode_binary(const std::uint8_t* B, std::uint32_t row, std::uint32_t col,
                                 std::uint32_t colBytes) {
    const std::uint32_t byteIdx = col * colBytes + row / 8;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << (row & 7));
    return (B[byteIdx] & bit) ? -1 : 1;
}

// Naive ternary-binary GeMM - no blocking, no SIMD
// TilingParams ignored - this is the simplest implementation
std::int32_t* GemmTBN_Blocked(
    const std::uint8_t* Ap, const std::uint8_t* Am, const std::uint8_t* B,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    const TilingParams& tp
) {
    (void)tp;  // Ignored in naive implementation

    const std::uint32_t rowBytesA = n / 8;
    const std::uint32_t colBytesB = n / 8;

    std::int32_t* result = new std::int32_t[static_cast<std::size_t>(m) * k];
    std::memset(result, 0, static_cast<std::size_t>(m) * k * sizeof(std::int32_t));

    for (std::uint32_t i = 0; i < m; i++) {
        for (std::uint32_t j = 0; j < k; j++) {
            std::int32_t sum = 0;
            for (std::uint32_t l = 0; l < n; l++) {
                const int a = decode_ternary(Ap, Am, i, l, rowBytesA);
                const int b = decode_binary(B, l, j, colBytesB);
                sum += a * b;
            }
            result[i * k + j] = sum;
        }
    }
    return result;
}
