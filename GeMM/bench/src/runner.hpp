#pragma once

#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include <functional>
#include "matrix_gen.hpp"

namespace bench {

/// Tiling parameters for blocked GeMM
struct TilingParams {
    std::uint32_t mblk;
    std::uint32_t nblk;
    std::uint32_t kblk;
    std::uint32_t mmk;
    std::uint32_t nmk;
};

/// Benchmark configuration
struct BenchConfig {
    std::string device;
    std::string impl;
    std::vector<MatrixType> matrix_types;
    std::vector<std::uint32_t> sizes;  // m = n = k for simplicity
    std::uint32_t runs;
    std::uint32_t warmup;
    std::uint32_t pause_ms;
    TilingParams tiling;
    std::string output_file;
};

/// High-resolution timer
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

/// Sleep for specified milliseconds
void sleep_ms(std::uint32_t ms);

/// Calculate GFLOPS for GeMM (m x n) * (n x k)
/// Each output element requires n multiplications and n-1 additions ≈ 2n ops
double calculate_gflops(std::uint32_t m, std::uint32_t n, std::uint32_t k, double time_ms);

} // namespace bench
