#pragma once
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <catch2/catch_all.hpp>
#include "matrices_to_test.hpp"
#include "GeMM.hpp"

inline std::pair<uint8_t*, uint8_t*>
PackTernaryRowMajor(const int8_t* A, uint32_t m, uint32_t n)
{
    if ((m % 8) || (n % 64))
        throw std::invalid_argument("PackTernary: m must be multiple of 8, n of 64");
    if (!A)
        throw std::invalid_argument("PackTernary: null pointer");

    const uint32_t rowBytes = n / 8;
    const size_t totalBytes = static_cast<size_t>(m) * rowBytes;

    uint8_t* Ap = new uint8_t[totalBytes];
    uint8_t* Am = new uint8_t[totalBytes];
    std::memset(Ap, 0, totalBytes);
    std::memset(Am, 0, totalBytes);

    for (uint32_t i = 0; i < m; ++i) {
        uint8_t* rowP = Ap + i * rowBytes;
        uint8_t* rowM = Am + i * rowBytes;
        for (uint32_t j = 0; j < n; ++j) {
            int8_t v = A[i * n + j];
            uint32_t byteIdx = j / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            if (v == 1)
                rowP[byteIdx] |= bit; // (1,0)
            else if (v == -1)
                rowM[byteIdx] |= bit; // (0,1)
            else if (v == 0)
                ; // (0,0)
            else
                throw std::invalid_argument("PackTernary: A must be -1,0,1");
        }
    }

    return {Ap, Am};
}

inline uint8_t*
PackBinaryColMajor(const int8_t* B, uint32_t n, uint32_t k)
{
    if ((n % 64) || (k % 8))
        throw std::invalid_argument("PackBinary: k must be multiple of 8, n of 64");
    if (!B)
        throw std::invalid_argument("PackBinary: null pointer");

    const uint32_t colBytes = n / 8;
    const size_t totalBytes = static_cast<size_t>(k) * colBytes;
    uint8_t* Bb = new uint8_t[totalBytes];
    std::memset(Bb, 0, totalBytes);

    for (uint32_t j = 0; j < k; ++j) {
        uint8_t* col = Bb + j * colBytes;
        for (uint32_t r = 0; r < n; ++r) {
            int8_t v = B[r * k + j];
            uint32_t byteIdx = r / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (r & 7));
            if (v == 1)
                ; // +1 => 0 bit
            else if (v == -1)
                col[byteIdx] |= bit; // -1 => 1 bit
            else
                throw std::invalid_argument("PackBinary: B must be -1 or 1");
        }
    }

    return Bb;
}

// Compare int32 result with ternary reference: sign must match
inline void CompareWithGlobalC(const int32_t* M, const int8_t* Cref, std::size_t total)
{
    for (std::size_t idx = 0; idx < total; ++idx) {
        int expected = Cref[idx];
        int actual = (M[idx] > 0) ? 1 : ((M[idx] < 0) ? -1 : 0);
        REQUIRE(actual == expected);
    }
}

// ============================================================================
// Test with original 128x128 matrices
// ============================================================================

TEST_CASE("128x128 - single k-block") {
    uint32_t m = 128, n = 128, k = 128;
    auto [Ap, Am] = PackTernaryRowMajor(A, m, n);
    auto Bb = PackBinaryColMajor(B, n, k);
    TilingParams p = {.kblk = 128, .mblk = 128, .nblk = 128, .mmk = 64, .nmk = 64};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

TEST_CASE("128x128 - multiple k-blocks") {
    uint32_t m = 128, n = 128, k = 128;
    auto [Ap, Am] = PackTernaryRowMajor(A, m, n);
    auto Bb = PackBinaryColMajor(B, n, k);
    TilingParams p = {.kblk = 64, .mblk = 128, .nblk = 128, .mmk = 64, .nmk = 64};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

TEST_CASE("128x128 - multiple all blocks") {
    uint32_t m = 128, n = 128, k = 128;
    auto [Ap, Am] = PackTernaryRowMajor(A, m, n);
    auto Bb = PackBinaryColMajor(B, n, k);
    TilingParams p = {.kblk = 64, .mblk = 64, .nblk = 64, .mmk = 32, .nmk = 32};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

TEST_CASE("128x128 - small microkernels") {
    uint32_t m = 128, n = 128, k = 128;
    auto [Ap, Am] = PackTernaryRowMajor(A, m, n);
    auto Bb = PackBinaryColMajor(B, n, k);
    TilingParams p = {.kblk = 64, .mblk = 32, .nblk = 32, .mmk = 16, .nmk = 16};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

// ============================================================================
// Test with 64x64 matrices
// ============================================================================

#include "matrices_64x64.hpp"

TEST_CASE("64x64 - single block") {
    uint32_t m = 64, n = 64, k = 64;
    auto [Ap, Am] = PackTernaryRowMajor(A64, m, n);
    auto Bb = PackBinaryColMajor(B64, n, k);
    TilingParams p = {.kblk = 64, .mblk = 64, .nblk = 64, .mmk = 32, .nmk = 32};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C64, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

TEST_CASE("64x64 - multiple k-blocks") {
    uint32_t m = 64, n = 64, k = 64;
    auto [Ap, Am] = PackTernaryRowMajor(A64, m, n);
    auto Bb = PackBinaryColMajor(B64, n, k);
    // kblk must be multiple of 64, so we can only have 1 k-block for n=64
    TilingParams p = {.kblk = 64, .mblk = 32, .nblk = 64, .mmk = 32, .nmk = 32};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C64, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}

TEST_CASE("64x64 - small tiles") {
    uint32_t m = 64, n = 64, k = 64;
    auto [Ap, Am] = PackTernaryRowMajor(A64, m, n);
    auto Bb = PackBinaryColMajor(B64, n, k);
    TilingParams p = {.kblk = 64, .mblk = 32, .nblk = 32, .mmk = 16, .nmk = 16};
    int32_t* Cresult = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    CompareWithGlobalC(Cresult, C64, static_cast<size_t>(m) * k);
    delete[] Ap; delete[] Am; delete[] Bb; delete[] Cresult;
}
