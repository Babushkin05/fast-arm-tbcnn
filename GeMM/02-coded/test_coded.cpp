// tbn_pack_unpack.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>

// Pack ternary A (m x n, values in {-1,0,1}) into two bit-planes row-major.
// Returns pair {Ap, Am}, each size m*(n/8) bytes.
inline std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
PackTernaryRowMajor(const std::vector<int8_t>& A, uint32_t m, uint32_t n)
{
    if ((m % 8) || (n % 8)) throw std::invalid_argument("PackTernary: m,n must be multiples of 8");
    if (A.size() != static_cast<size_t>(m) * n) throw std::invalid_argument("PackTernary: size mismatch");

    const uint32_t rowBytes = n / 8;
    std::vector<uint8_t> Ap(m * rowBytes, 0), Am(m * rowBytes, 0);

    for (uint32_t i = 0; i < m; ++i) {
        auto* rowP = &Ap[i * rowBytes];
        auto* rowM = &Am[i * rowBytes];
        for (uint32_t j = 0; j < n; ++j) {
            int8_t v = A[i * n + j];
            uint32_t byteIdx = j / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            if (v == 1)       rowP[byteIdx] |= bit;    // (1,0)
            else if (v == -1) rowM[byteIdx] |= bit;    // (0,1)
            else if (v == 0)  ;                        // (0,0)
            else throw std::invalid_argument("PackTernary: A must be -1,0,1");
        }
    }
    return {std::move(Ap), std::move(Am)};
}

// Pack binary B (n x k, values in {-1,1}) into one bit-plane column-major.
// Encoding: bit 0 => +1, bit 1 => -1. Size k*(n/8) bytes.
inline std::vector<uint8_t>
PackBinaryColMajor(const std::vector<int8_t>& B, uint32_t n, uint32_t k)
{
    if ((n % 8) || (k % 8)) throw std::invalid_argument("PackBinary: n,k must be multiples of 8");
    if (B.size() != static_cast<size_t>(n) * k) throw std::invalid_argument("PackBinary: size mismatch");

    const uint32_t colBytes = n / 8;
    std::vector<uint8_t> Bb(k * colBytes, 0);

    for (uint32_t j = 0; j < k; ++j) {
        auto* col = &Bb[j * colBytes];
        for (uint32_t r = 0; r < n; ++r) {
            int8_t v = B[r * k + j];
            uint32_t byteIdx = r / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (r & 7));
            if (v == 1)       ;             // +1 => 0 bit
            else if (v == -1) col[byteIdx] |= bit; // -1 => 1 bit
            else throw std::invalid_argument("PackBinary: B must be -1 or 1");
        }
    }
    return Bb;
}

// Unpack ternary C (m x k) from two bit-planes row-major into int8 {-1,0,1}.
inline std::vector<int8_t>
UnpackTernaryRowMajor(const std::vector<uint8_t>& Cplus,
                      const std::vector<uint8_t>& Cminus,
                      uint32_t m, uint32_t k)
{
    if ((m % 8) || (k % 8)) throw std::invalid_argument("UnpackTernary: m,k must be multiples of 8");
    const uint32_t rowBytes = k / 8;
    if (Cplus.size() != static_cast<size_t>(m) * rowBytes ||
        Cminus.size() != static_cast<size_t>(m) * rowBytes)
        throw std::invalid_argument("UnpackTernary: size mismatch");

    std::vector<int8_t> C(m * k, 0);
    for (uint32_t i = 0; i < m; ++i) {
        const auto* rowP = &Cplus[i * rowBytes];
        const auto* rowM = &Cminus[i * rowBytes];
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t byteIdx = j / 8;
            uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            bool p = (rowP[byteIdx] & bit) != 0;
            bool mbit = (rowM[byteIdx] & bit) != 0;
            if (p && !mbit)        C[i * k + j] = +1;
            else if (!p && mbit)   C[i * k + j] = -1;
            else if (!p && !mbit)  C[i * k + j] = 0;
            else throw std::runtime_error("UnpackTernary: invalid (1,1) state");
        }
    }
    return C;
}

// Reference check: compute naive int32 Cref = A * B and compare signs with unpacked C.
// A: m x n in {-1,0,1}, B: n x k in {-1,1}, Cunpacked: m x k in {-1,0,1}.
inline bool
CheckTBNResult(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
               const std::vector<int8_t>& Cunpacked,
               uint32_t m, uint32_t n, uint32_t k)
{
    if (A.size() != static_cast<size_t>(m) * n ||
        B.size() != static_cast<size_t>(n) * k ||
        Cunpacked.size() != static_cast<size_t>(m) * k)
        throw std::invalid_argument("CheckTBNResult: size mismatch");

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            int32_t acc = 0;
            for (uint32_t t = 0; t < n; ++t) {
                acc += static_cast<int32_t>(A[i * n + t]) *
                       static_cast<int32_t>(B[t * k + j]);
            }
            int8_t sign;
            if (acc > 0) sign = +1;
            else if (acc < 0) sign = -1;
            else sign = 0;

            if (sign != Cunpacked[i * k + j]) return false;
        }
    }
    return true;
}
