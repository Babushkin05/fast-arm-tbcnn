#include "tbn/operators/gemm.hpp"
#include "tbn/utils/errors.hpp"
#include "tbn/utils/logging.hpp"
#include "tbn/runtime/types.hpp"
#include "tbn/memory/packed_weights.hpp"
#include "tbn/quantization/quantizer.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace tbn {

// Forward declarations for optimized implementations
namespace impl {

    // Naive reference implementation
    Tensor gemm_naive(const Tensor& A, const Tensor& B, const Tensor* C,
                      float alpha, float beta, bool transA, bool transB) {
        const float* A_data = A.typed_data<float>();
        const float* B_data = B.typed_data<float>();
        const float* C_data = C ? C->typed_data<float>() : nullptr;

        int64_t M = transA ? A.shape().dims[1] : A.shape().dims[0];
        int64_t K = transA ? A.shape().dims[0] : A.shape().dims[1];
        int64_t N = transB ? B.shape().dims[0] : B.shape().dims[1];

        Tensor result(Shape{M, N}, DataType::FLOAT32);
        float* result_data = result.typed_data<float>();

        // Initialize with beta * C if provided
        if (C_data && beta != 0.0f) {
            for (int64_t i = 0; i < M * N; ++i) {
                result_data[i] = beta * C_data[i];
            }
        } else {
            std::memset(result_data, 0, M * N * sizeof(float));
        }

        // Compute A * B
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t a_idx = transA ? (k * M + i) : (i * K + k);
                    int64_t b_idx = transB ? (j * K + k) : (k * N + j);
                    sum += A_data[a_idx] * B_data[b_idx];
                }
                result_data[i * N + j] += alpha * sum;
            }
        }

        return result;
    }

    // Ternary GeMM with packed weights
    Tensor gemm_ternary_packed(const TernaryPackedWeights& A_packed, const Tensor& B,
                               const Tensor* C, float alpha, float beta,
                               bool transA, bool transB) {
        TBN_CHECK(!transA, NotImplementedError, "Transposed ternary matrices not implemented");
        TBN_CHECK(!transB, NotImplementedError, "Transposed B matrix not implemented");

        int64_t M = A_packed.shape().dims[0];
        int64_t K = A_packed.shape().dims[1];
        int64_t N = B.shape().dims[1];

        const float* B_data = B.typed_data<float>();
        const float* C_data = C ? C->typed_data<float>() : nullptr;

        Tensor result(Shape{M, N}, DataType::FLOAT32);
        float* result_data = result.typed_data<float>();

        // Initialize result
        if (C_data && beta != 0.0f) {
            for (int64_t i = 0; i < M * N; ++i) {
                result_data[i] = beta * C_data[i];
            }
        } else {
            std::memset(result_data, 0, M * N * sizeof(float));
        }

        // Compute with ternary weights
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    TernaryWeight weight = A_packed.get_weight(i * K + k);
                    float dequantized = dequantize_ternary(weight);
                    sum += dequantized * B_data[k * N + j];
                }
                result_data[i * N + j] += alpha * sum;
            }
        }

        return result;
    }

    // Binary GeMM with packed weights
    Tensor gemm_binary_packed(const BinaryPackedWeights& A_packed, const Tensor& B,
                              const Tensor* C, float alpha, float beta,
                              bool transA, bool transB) {
        TBN_CHECK(!transA, NotImplementedError, "Transposed binary matrices not implemented");
        TBN_CHECK(!transB, NotImplementedError, "Transposed B matrix not implemented");

        int64_t M = A_packed.shape().dims[0];
        int64_t K = A_packed.shape().dims[1];
        int64_t N = B.shape().dims[1];

        const float* B_data = B.typed_data<float>();
        const float* C_data = C ? C->typed_data<float>() : nullptr;

        Tensor result(Shape{M, N}, DataType::FLOAT32);
        float* result_data = result.typed_data<float>();

        // Initialize result
        if (C_data && beta != 0.0f) {
            for (int64_t i = 0; i < M * N; ++i) {
                result_data[i] = beta * C_data[i];
            }
        } else {
            std::memset(result_data, 0, M * N * sizeof(float));
        }

        // Compute with binary weights
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    BinaryWeight weight = A_packed.get_weight(i * K + k);
                    float dequantized = dequantize_binary(weight);
                    sum += dequantized * B_data[k * N + j];
                }
                result_data[i * N + j] += alpha * sum;
            }
        }

        return result;
    }

    // Cache-optimized blocked implementation
    Tensor gemm_blocked(const Tensor& A, const Tensor& B, const Tensor* C,
                        float alpha, float beta, bool transA, bool transB) {
        const int64_t BLOCK_SIZE = 64;

        const float* A_data = A.typed_data<float>();
        const float* B_data = B.typed_data<float>();
        const float* C_data = C ? C->typed_data<float>() : nullptr;

        int64_t M = transA ? A.shape().dims[1] : A.shape().dims[0];
        int64_t K = transA ? A.shape().dims[0] : A.shape().dims[1];
        int64_t N = transB ? B.shape().dims[0] : B.shape().dims[1];

        Tensor result(Shape{M, N}, DataType::FLOAT32);
        float* result_data = result.typed_data<float>();

        // Initialize result
        if (C_data && beta != 0.0f) {
            for (int64_t i = 0; i < M * N; ++i) {
                result_data[i] = beta * C_data[i];
            }
        } else {
            std::memset(result_data, 0, M * N * sizeof(float));
        }

        // Blocked matrix multiplication
        for (int64_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
            int64_t i1 = std::min(i0 + BLOCK_SIZE, M);
            for (int64_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                int64_t j1 = std::min(j0 + BLOCK_SIZE, N);
                for (int64_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                    int64_t k1 = std::min(k0 + BLOCK_SIZE, K);

                    // Compute block
                    for (int64_t i = i0; i < i1; ++i) {
                        for (int64_t j = j0; j < j1; ++j) {
                            float sum = 0.0f;
                            for (int64_t k = k0; k < k1; ++k) {
                                int64_t a_idx = transA ? (k * M + i) : (i * K + k);
                                int64_t b_idx = transB ? (j * K + k) : (k * N + j);
                                sum += A_data[a_idx] * B_data[b_idx];
                            }
                            result_data[i * N + j] += alpha * sum;
                        }
                    }
                }
            }
        }

        return result;
    }
}

// Public API implementations

Tensor gemm(const Tensor& A, const Tensor& B, const Tensor* C,
            float alpha, float beta, bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm: A=" + shape_to_string(A.shape()) +
                  " B=" + shape_to_string(B.shape()) +
                  " alpha=" + std::to_string(alpha) +
                  " beta=" + std::to_string(beta) +
                  " transA=" + std::to_string(transA) +
                  " transB=" + std::to_string(transB));

    // Validate inputs
    TBN_CHECK(A.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Gemm requires float32 input A");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Gemm requires float32 input B");
    TBN_CHECK(A.shape().dims.size() == 2, InvalidShapeError,
              "Gemm requires 2D tensor A");
    TBN_CHECK(B.shape().dims.size() == 2, InvalidShapeError,
              "Gemm requires 2D tensor B");

    int64_t M = transA ? A.shape().dims[1] : A.shape().dims[0];
    int64_t K = transA ? A.shape().dims[0] : A.shape().dims[1];
    int64_t N = transB ? B.shape().dims[0] : B.shape().dims[1];

    TBN_CHECK(K == (transB ? B.shape().dims[1] : B.shape().dims[0]),
              InvalidShapeError,
              "Matrix dimensions incompatible for multiplication");

    if (C) {
        TBN_CHECK(C->shape().dims.size() == 2, InvalidShapeError,
                  "C must be 2D tensor");
        TBN_CHECK(C->shape().dims[0] == M && C->shape().dims[1] == N,
                  InvalidShapeError,
                  "C shape must match output shape");
    }

    // Use blocked implementation for better cache performance
    return impl::gemm_blocked(A, B, C, alpha, beta, transA, transB);
}

Tensor gemm_ternary(const Tensor& A_ternary, const Tensor& B,
                    const Tensor* C, float alpha, float beta,
                    bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm_ternary: A=" + shape_to_string(A_ternary.shape()) +
                  " B=" + shape_to_string(B.shape()));

    TBN_CHECK(A_ternary.dtype() == DataType::TERNARY, InvalidArgumentError,
              "A must be ternary tensor");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "B must be float32 tensor");

    // Convert ternary tensor to packed format for efficiency
    TernaryPackedWeights packed_weights(A_ternary.shape());
    const TernaryWeight* ternary_data = A_ternary.typed_data<TernaryWeight>();

    // Pack the weights
    for (int64_t i = 0; i < A_ternary.num_elements(); ++i) {
        packed_weights.set_weight(i, ternary_data[i]);
    }

    return impl::gemm_ternary_packed(packed_weights, B, C, alpha, beta, transA, transB);
}

Tensor gemm_binary(const Tensor& A_binary, const Tensor& B,
                   const Tensor* C, float alpha, float beta,
                   bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm_binary: A=" + shape_to_string(A_binary.shape()) +
                  " B=" + shape_to_string(B.shape()));

    TBN_CHECK(A_binary.dtype() == DataType::BINARY, InvalidArgumentError,
              "A must be binary tensor");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "B must be float32 tensor");

    // Convert binary tensor to packed format
    BinaryPackedWeights packed_weights(A_binary.shape());
    const BinaryWeight* binary_data = A_binary.typed_data<BinaryWeight>();

    // Pack the weights
    for (int64_t i = 0; i < A_binary.num_elements(); ++i) {
        packed_weights.set_weight(i, binary_data[i]);
    }

    return impl::gemm_binary_packed(packed_weights, B, C, alpha, beta, transA, transB);
}

Tensor gemm_mixed(const Tensor& A, const Tensor& B,
                  const Tensor* C, float alpha, float beta,
                  bool transA, bool transB) {
    // Route to appropriate implementation based on tensor types
    if (A.dtype() == DataType::TERNARY) {
        return gemm_ternary(A, B, C, alpha, beta, transA, transB);
    } else if (A.dtype() == DataType::BINARY) {
        return gemm_binary(A, B, C, alpha, beta, transA, transB);
    } else if (A.dtype() == DataType::FLOAT32) {
        return gemm(A, B, C, alpha, beta, transA, transB);
    } else {
        throw InvalidArgumentError("Unsupported data type for mixed precision GeMM");
    }
}

// Batch GeMM
Tensor batch_gemm(const std::vector<Tensor>& A_batch, const std::vector<Tensor>& B_batch,
                  const std::vector<Tensor>* C_batch,
                  float alpha, float beta,
                  bool transA, bool transB) {
    TBN_CHECK(A_batch.size() == B_batch.size(), InvalidArgumentError,
              "A and B batches must have same size");

    if (C_batch) {
        TBN_CHECK(C_batch->size() == A_batch.size(), InvalidArgumentError,
                  "C batch must have same size as A and B");
    }

    std::vector<Tensor> results;
    results.reserve(A_batch.size());

    for (size_t i = 0; i < A_batch.size(); ++i) {
        const Tensor* C_ptr = C_batch ? &(*C_batch)[i] : nullptr;
        results.push_back(gemm(A_batch[i], B_batch[i], C_ptr, alpha, beta, transA, transB));
    }

    // Stack results into single tensor
    // For now, return first result
    return results[0];
}

Tensor strided_batch_gemm(const Tensor& A, const Tensor& B, const Tensor* C,
                          int64_t batch_size, int64_t strideA, int64_t strideB,
                          float alpha, float beta,
                          bool transA, bool transB) {
    // Simplified implementation - extract batches and call batch_gemm
    std::vector<Tensor> A_batch, B_batch;

    for (int64_t i = 0; i < batch_size; ++i) {
        // Extract batch - this is simplified, real implementation would
        // properly slice the tensors
        A_batch.push_back(A);
        B_batch.push_back(B);
    }

    std::vector<Tensor> C_batch;
    if (C) {
        for (int64_t i = 0; i < batch_size; ++i) {
            C_batch.push_back(*C);
        }
    }

    return batch_gemm(A_batch, B_batch, C ? &C_batch : nullptr,
                      alpha, beta, transA, transB);
}

// Performance tuning
GemmParams auto_tune_gemm(int64_t M, int64_t N, int64_t K) {
    GemmParams params;

    // Simple heuristics based on matrix sizes
    if (M > 512 && N > 512 && K > 512) {
        params.tile_size_m = 128;
        params.tile_size_n = 128;
        params.tile_size_k = 64;
    } else if (M > 256 && N > 256 && K > 256) {
        params.tile_size_m = 64;
        params.tile_size_n = 64;
        params.tile_size_k = 32;
    }

    return params;
}

Tensor gemm_tuned(const Tensor& A, const Tensor& B, const Tensor* C,
                  const GemmParams& params) {
    // For now, just call the regular gemm
    // Future: use params to select optimal implementation
    return gemm(A, B, C, 1.0f, 0.0f, false, false);
}

// Helper function
static std::string shape_to_string(const Shape& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape.dims[i]);
    }
    result += "]";
    return result;
}

} // namespace tbn

// Notes for ARM NEON optimization:
// 1. Use vld1q_f32 for loading 4 floats at once
// 2. Use vfmaq_f32 for fused multiply-add
// 3. Use vmulq_f32 for vector multiplication
// 4. Process 4x4 blocks for better SIMD utilization
// 5. Use prefetch instructions for cache optimization
// 6. Align memory to 16-byte boundaries
// 7. Use inline assembly for critical loops
// 8. Consider using ARM Compute Library for complex operations

// Ternary optimization notes:
// 1. Use bit operations for weight extraction
// 2. Avoid branching in inner loops
// 3. Use lookup tables for dequantization
// 4. Pack multiple ternary values per byte
// 5. Use SIMD for parallel dequantization
// 6. Consider sign-bit manipulation for multiplication

// Binary optimization notes:
// 1. Use bit manipulation for weight extraction
// 2. Use conditional move instead of branches
// 3. Pack 8 binary values per byte
// 4. Use SIMD for parallel operations
// 5. Consider population count for efficient computation