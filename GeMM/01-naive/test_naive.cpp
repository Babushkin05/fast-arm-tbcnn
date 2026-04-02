#include <catch2/catch_all.hpp>
#include <cstring>
#include "GeMM.hpp"

// Helper to pack ternary matrix (values: -1, 0, +1)
static void pack_ternary(uint8_t* Ap, uint8_t* Am, const int8_t* A, uint32_t m, uint32_t n) {
    const uint32_t rowBytes = n / 8;
    std::memset(Ap, 0, m * rowBytes);
    std::memset(Am, 0, m * rowBytes);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            uint32_t byteIdx = i * rowBytes + j / 8;
            if (A[i * n + j] == 1) Ap[byteIdx] |= bit;
            else if (A[i * n + j] == -1) Am[byteIdx] |= bit;
        }
    }
}

// Helper to pack binary matrix (values: -1, +1)
static void pack_binary(uint8_t* Bb, const int8_t* B, uint32_t n, uint32_t k) {
    const uint32_t colBytes = n / 8;
    std::memset(Bb, 0, k * colBytes);
    for (uint32_t j = 0; j < k; ++j) {
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t bit = static_cast<uint8_t>(1u << (i & 7));
            uint32_t byteIdx = j * colBytes + i / 8;
            if (B[i * k + j] == -1) Bb[byteIdx] |= bit;
        }
    }
}

TEST_CASE("Naive GeMM basic 64x64") {
    // Use 64x64 matrices (n must be multiple of 64 for bit-packing)
    const uint32_t m = 64, n = 64, k = 64;
    const uint32_t rowBytesA = n / 8;
    const uint32_t colBytesB = n / 8;

    // Simple test: identity-like matrices
    // A: diagonal of +1, rest 0
    // B: all +1
    std::vector<int8_t> A(m * n, 0);
    std::vector<int8_t> B(n * k, 1);
    for (uint32_t i = 0; i < std::min(m, n); ++i) {
        A[i * n + i] = 1;
    }

    auto Ap = std::make_unique<uint8_t[]>(m * rowBytesA);
    auto Am = std::make_unique<uint8_t[]>(m * rowBytesA);
    auto Bb = std::make_unique<uint8_t[]>(k * colBytesB);

    pack_ternary(Ap.get(), Am.get(), A.data(), m, n);
    pack_binary(Bb.get(), B.data(), n, k);

    TilingParams tp = {64, 64, 128, 16, 8};
    int32_t* C = GemmTBN_Blocked(Ap.get(), Am.get(), Bb.get(), m, n, k, tp);

    // With diagonal A and all-ones B, result should be row of 1s where A has +1
    REQUIRE(C[0] == 1);  // First row has +1 at position 0
    REQUIRE(C[65] == 1); // Second row has +1 at position 1
    REQUIRE(C[128] == 1);

    delete[] C;
}
