#pragma once
#include <cstdint>

struct TilingParams {
    std::uint32_t mblk;
    std::uint32_t nblk;
    std::uint32_t kblk;
    std::uint32_t mmk;
    std::uint32_t nmk;
};

// A - m x n (ternary), B - n x k (binary); result - m x k
// A is encoded as two bit-planes: Ap (positive) and Am (negative)
// B is encoded as single bit-plane
// TilingParams ignored - this is naive implementation
std::int32_t* GemmTBN_Blocked(
    const std::uint8_t* Ap, const std::uint8_t* Am, const std::uint8_t* B,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    const TilingParams& tp
);
