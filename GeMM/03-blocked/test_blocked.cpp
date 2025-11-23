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

inline int8_t*
UnpackTernaryRowMajor(const uint8_t* Cplus, const uint8_t* Cminus,
                      uint32_t m, uint32_t k)
{
    if ((m % 8) || (k % 8))
        throw std::invalid_argument("UnpackTernary: m,k must be multiples of 8");
    if (!Cplus || !Cminus)
        throw std::invalid_argument("UnpackTernary: null pointer");

    const uint32_t rowBytes = k / 8;
    int8_t* C = new int8_t[static_cast<size_t>(m) * k];

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

inline void CompareWithGlobalC(const int8_t* M, std::size_t m, std::size_t k)
{
    const std::size_t total = m * k;
    for (std::size_t idx = 0; idx < total; ++idx) {
        REQUIRE(M[idx] == C[idx]);
    }
}

TEST_CASE("Case 1") {
    uint32_t m = 64;
    uint32_t n = 64;
    uint32_t k = 64;

    auto [Ap, Am] = PackTernaryRowMajor(A, m, k);
    auto Bb = PackBinaryColMajor(B, n, k);

    TilingParams p = {.kblk = 64, .mblk = 64, .nblk = 64, .mmk = 64, .nmk = 64};

    auto [Cp, Cm] = GemmTBN_Blocked(Ap, Am, Bb, m, n, k, p);
    auto Cl = UnpackTernaryRowMajor(Cp, Cm, m, k);

    CompareWithGlobalC(Cl, m, k);
}
