#pragma once

#include "../runtime/tensor.hpp"
#include "../runtime/types.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"

namespace tbn {

// Standard GeMM (General Matrix Multiplication)
Tensor gemm(const Tensor& A, const Tensor& B, const Tensor* C = nullptr,
            float alpha = 1.0f, float beta = 0.0f, bool transA = false, bool transB = false);

// Ternary GeMM - A is ternary, B is float
Tensor gemm_ternary(const Tensor& A_ternary, const Tensor& B,
                    const Tensor* C = nullptr, float alpha = 1.0f, float beta = 0.0f,
                    bool transA = false, bool transB = false);

// Binary GeMM - A is binary, B is float
Tensor gemm_binary(const Tensor& A_binary, const Tensor& B,
                   const Tensor* C = nullptr, float alpha = 1.0f, float beta = 0.0f,
                   bool transA = false, bool transB = false);

// Mixed precision GeMM - both A and B can be quantized
Tensor gemm_mixed(const Tensor& A, const Tensor& B,
                  const Tensor* C = nullptr, float alpha = 1.0f, float beta = 0.0f,
                  bool transA = false, bool transB = false);

// Batch GeMM - batched matrix multiplication
Tensor batch_gemm(const std::vector<Tensor>& A_batch, const std::vector<Tensor>& B_batch,
                  const std::vector<Tensor>* C_batch = nullptr,
                  float alpha = 1.0f, float beta = 0.0f,
                  bool transA = false, bool transB = false);

// Strided batch GeMM - more efficient for uniform batch sizes
Tensor strided_batch_gemm(const Tensor& A, const Tensor& B, const Tensor* C = nullptr,
                          int64_t batch_size = 1, int64_t strideA = 0, int64_t strideB = 0,
                          float alpha = 1.0f, float beta = 0.0f,
                          bool transA = false, bool transB = false);

// Optimized GeMM implementations
namespace impl {
    // Naive implementation for reference
    Tensor gemm_naive(const Tensor& A, const Tensor& B, const Tensor* C,
                      float alpha, float beta, bool transA, bool transB);

    // Cache-optimized implementation
    Tensor gemm_blocked(const Tensor& A, const Tensor& B, const Tensor* C,
                        float alpha, float beta, bool transA, bool transB);

    // SIMD-optimized implementation (ARM NEON)
    Tensor gemm_neon(const Tensor& A, const Tensor& B, const Tensor* C,
                     float alpha, float beta, bool transA, bool transB);

    // Ternary-weight GeMM with bit-packing
    Tensor gemm_ternary_packed(const Tensor& A_packed, const Tensor& B,
                               const Tensor* C, float alpha, float beta,
                               bool transA, bool transB);

    // Binary-weight GeMM with bit-packing
    Tensor gemm_binary_packed(const Tensor& A_packed, const Tensor& B,
                              const Tensor* C, float alpha, float beta,
                              bool transA, bool transB);
}

// Performance tuning parameters
struct GemmParams {
    int tile_size_m = 64;
    int tile_size_n = 64;
    int tile_size_k = 32;
    int unroll_factor = 4;
    bool use_neon = true;
    bool prefetch = true;

    GemmParams() = default;
};

// Advanced GeMM with performance tuning
Tensor gemm_tuned(const Tensor& A, const Tensor& B, const Tensor* C = nullptr,
                  const GemmParams& params = GemmParams());

// Auto-tuning - find best parameters for given sizes
GemmParams auto_tune_gemm(int64_t M, int64_t N, int64_t K);

// Helper function to convert shape to string
std::string shape_to_string(const Shape& shape);

} // namespace tbn