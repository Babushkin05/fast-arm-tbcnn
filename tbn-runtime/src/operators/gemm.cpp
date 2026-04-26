#include "tbn/operators/gemm.hpp"
#include "tbn/utils/errors.hpp"
#include "tbn/utils/logging.hpp"
#include "tbn/runtime/types.hpp"
#include "tbn/memory/packed_weights.hpp"
#include "tbn/quantization/quantizer.hpp"

// Include the optimized GeMM engine
#include "../../../GeMM/05-final/GeMM.hpp"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <span>

namespace tbn {

// ============================================================================
// Helper: Convert Tensor data to span
// ============================================================================
template<typename T>
std::span<const T> make_span(const Tensor& tensor) {
    return std::span<const T>(tensor.typed_data<T>(), tensor.num_elements());
}

// ============================================================================
// Float × Float GEMM (blocked for cache efficiency)
// ============================================================================
namespace impl {

Tensor gemm_float_blocked(const Tensor& A, const Tensor& B, const Tensor* C,
                          float alpha, float beta, bool transA, bool transB) {
    const int64_t BLOCK_SIZE = 64;

    const float* A_data = A.typed_data<float>();
    const float* B_data = B.typed_data<float>();
    const float* C_data = C ? C->typed_data<float>() : nullptr;
    bool C_is_1d = C && C->shape().dims.size() == 1;

    int64_t M = transA ? A.shape().dims[1] : A.shape().dims[0];
    int64_t K = transA ? A.shape().dims[0] : A.shape().dims[1];
    int64_t N = transB ? B.shape().dims[0] : B.shape().dims[1];

    Tensor result(Shape{M, N}, DataType::FLOAT32);
    float* result_data = result.typed_data<float>();

    // Initialize result
    if (C_data && beta != 0.0f) {
        if (C_is_1d) {
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    result_data[i * N + j] = beta * C_data[j];
                }
            }
        } else {
            for (int64_t i = 0; i < M * N; ++i) {
                result_data[i] = beta * C_data[i];
            }
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

} // namespace impl

// ============================================================================
// Pad dimensions to meet GeMM requirements
// ============================================================================
struct PaddedDimensions {
    uint32_t m, n, k;
    uint32_t m_padded, n_padded, k_padded;

    PaddedDimensions(uint32_t m_, uint32_t n_, uint32_t k_)
        : m(m_), n(n_), k(k_) {
        // GeMM requirements:
        // - TernaryMatrix: rows multiple of 8, cols multiple of 64
        // - BinaryMatrix: rows multiple of 64, cols multiple of 8
        // - TilingParams: n multiple of 128, m multiple of mmk (16), k multiple of nmk (8)
        m_padded = ((m + 15) / 16) * 16;    // multiple of 16 (mmk)
        k_padded = ((k + 63) / 64) * 64;    // multiple of 64 (TernaryMatrix cols, BinaryMatrix rows)
        n_padded = ((n + 127) / 128) * 128; // multiple of 128 (TilingParams n)
    }
};

// ============================================================================
// Pad float matrix with zeros
// ============================================================================
Tensor pad_matrix(const Tensor& tensor, uint32_t target_rows, uint32_t target_cols) {
    uint32_t rows = static_cast<uint32_t>(tensor.shape().dims[0]);
    uint32_t cols = static_cast<uint32_t>(tensor.shape().dims[1]);

    if (rows == target_rows && cols == target_cols) {
        return tensor;
    }

    Tensor padded(Shape{target_rows, target_cols}, DataType::FLOAT32);
    float* dst = padded.typed_data<float>();
    const float* src = tensor.typed_data<float>();

    std::memset(dst, 0, static_cast<size_t>(target_rows) * target_cols * sizeof(float));

    for (uint32_t i = 0; i < rows; ++i) {
        std::memcpy(dst + i * target_cols, src + i * cols, cols * sizeof(float));
    }

    return padded;
}

// ============================================================================
// Quantize float activations to ternary
// ============================================================================
std::vector<int8_t> quantize_to_ternary(const float* data, size_t count,
                                         float threshold_low, float threshold_high) {
    std::vector<int8_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        if (val < threshold_low) {
            result[i] = -1;
        } else if (val > threshold_high) {
            result[i] = +1;
        } else {
            result[i] = 0;
        }
    }
    return result;
}

// ============================================================================
// Quantize float weights to binary
// ============================================================================
std::vector<int8_t> quantize_to_binary(const float* data, size_t count) {
    std::vector<int8_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = (data[i] >= 0) ? +1 : -1;
    }
    return result;
}

// ============================================================================
// Convert BinaryWeight tensor to int8_t array
// ============================================================================
std::vector<int8_t> convert_binary_weights(const Tensor& tensor) {
    int64_t size = tensor.num_elements();
    std::vector<int8_t> result(size);

    const BinaryWeight* data = tensor.typed_data<BinaryWeight>();
    for (int64_t i = 0; i < size; ++i) {
        result[i] = (data[i] == BINARY_ONE) ? +1 : -1;
    }

    return result;
}

// ============================================================================
// Convert TernaryWeight tensor to int8_t array
// ============================================================================
std::vector<int8_t> convert_ternary_weights(const Tensor& tensor) {
    int64_t size = tensor.num_elements();
    std::vector<int8_t> result(size);

    const TernaryWeight* data = tensor.typed_data<TernaryWeight>();
    for (int64_t i = 0; i < size; ++i) {
        result[i] = static_cast<int8_t>(data[i]);
    }

    return result;
}

// ============================================================================
// Optimized Ternary × Binary GEMM using GeMM engine
// ============================================================================
Tensor gemm_ternary_binary_optimized(
    const Tensor& A_ternary,
    const Tensor& B_binary,
    float scale,
    float threshold_low = -0.1f,
    float threshold_high = 0.1f
) {
    uint32_t M = static_cast<uint32_t>(A_ternary.shape().dims[0]);
    uint32_t K = static_cast<uint32_t>(A_ternary.shape().dims[1]);
    uint32_t N = static_cast<uint32_t>(B_binary.shape().dims[1]);

    // Calculate padded dimensions
    PaddedDimensions dims(M, N, K);

    TBN_LOG_DEBUG("gemm_ternary_binary_optimized: M=" + std::to_string(M) +
                  " K=" + std::to_string(K) + " N=" + std::to_string(N));
    TBN_LOG_DEBUG("Padded: m=" + std::to_string(dims.m_padded) +
                  " k=" + std::to_string(dims.k_padded) + " n=" + std::to_string(dims.n_padded));

    // Convert and pad A (ternary activations)
    std::vector<int8_t> a_data;
    if (A_ternary.dtype() == DataType::TERNARY) {
        a_data = convert_ternary_weights(A_ternary);
    } else if (A_ternary.dtype() == DataType::FLOAT32) {
        // Quantize float to ternary
        Tensor a_padded = pad_matrix(A_ternary, dims.m_padded, dims.k_padded);
        a_data = quantize_to_ternary(
            a_padded.typed_data<float>(),
            static_cast<size_t>(dims.m_padded) * dims.k_padded,
            threshold_low, threshold_high
        );
    } else {
        throw InvalidArgumentError("A must be TERNARY or FLOAT32");
    }

    // Pad a_data if needed
    if (dims.m != dims.m_padded || dims.k != dims.k_padded) {
        std::vector<int8_t> padded(static_cast<size_t>(dims.m_padded) * dims.k_padded, 0);
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < K; ++j) {
                padded[i * dims.k_padded + j] = a_data[i * K + j];
            }
        }
        a_data = std::move(padded);
    }

    // Convert and pad B (binary weights)
    std::vector<int8_t> b_data;
    if (B_binary.dtype() == DataType::BINARY) {
        b_data = convert_binary_weights(B_binary);
    } else if (B_binary.dtype() == DataType::FLOAT32) {
        b_data = quantize_to_binary(
            B_binary.typed_data<float>(),
            B_binary.num_elements()
        );
    } else {
        throw InvalidArgumentError("B must be BINARY or FLOAT32");
    }

    // Pad b_data if needed (K rows, N cols)
    std::vector<int8_t> b_padded(static_cast<size_t>(dims.k_padded) * dims.n_padded, 1);
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            b_padded[i * dims.n_padded + j] = b_data[i * N + j];
        }
    }

    // Pack matrices for GeMM engine
    ::tbn::TernaryMatrix a_packed = ::tbn::TernaryMatrix::pack(
        std::span<const int8_t>(a_data.data(), a_data.size()),
        dims.m_padded, dims.k_padded
    );

    ::tbn::BinaryMatrix b_packed = ::tbn::BinaryMatrix::pack(
        std::span<const int8_t>(b_padded.data(), b_padded.size()),
        dims.k_padded, dims.n_padded
    );

    // Run optimized GeMM
    ::tbn::TilingParams params = ::tbn::TilingParams::default_128x128();
    ::tbn::GemmEngine engine;
    ::tbn::Int32Matrix result = engine.compute(a_packed.view(), b_packed.view(), params);

    // Extract valid region and apply scale
    Tensor output(Shape{M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            out_data[i * N + j] = static_cast<float>(result.at(i, j)) * scale;
        }
    }

    return output;
}

// ============================================================================
// Public API: Standard Float × Float GEMM
// ============================================================================
Tensor gemm(const Tensor& A, const Tensor& B, const Tensor* C,
            float alpha, float beta, bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm: A=" + shape_to_string(A.shape()) +
                  " B=" + shape_to_string(B.shape()));

    TBN_CHECK(A.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "gemm requires float32 input A");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "gemm requires float32 input B");

    return impl::gemm_float_blocked(A, B, C, alpha, beta, transA, transB);
}

// ============================================================================
// Public API: Ternary × Float GEMM
// ============================================================================
Tensor gemm_ternary(const Tensor& A_ternary, const Tensor& B,
                    const Tensor* C, float alpha, float beta,
                    bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm_ternary: A=" + shape_to_string(A_ternary.shape()) +
                  " B=" + shape_to_string(B.shape()));

    TBN_CHECK(!transA, NotImplementedError, "Transposed ternary A not supported");
    TBN_CHECK(!transB, NotImplementedError, "Transposed B not supported");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "B must be float32");

    // Convert B to binary and use optimized path
    std::vector<int8_t> b_binary(B.num_elements());
    const float* b_data = B.typed_data<float>();
    for (int64_t i = 0; i < B.num_elements(); ++i) {
        b_binary[i] = (b_data[i] >= 0) ? +1 : -1;
    }

    // Create binary tensor
    Tensor B_binary(B.shape(), DataType::BINARY);
    BinaryWeight* bw = B_binary.typed_data<BinaryWeight>();
    for (int64_t i = 0; i < B.num_elements(); ++i) {
        bw[i] = (b_binary[i] >= 0) ? BINARY_ONE : BINARY_ZERO;
    }

    // Use optimized path
    Tensor result = gemm_ternary_binary_optimized(A_ternary, B_binary, 1.0f);

    // Apply alpha and add C
    float* result_data = result.typed_data<float>();
    int64_t M = result.shape().dims[0];
    int64_t N = result.shape().dims[1];

    for (int64_t i = 0; i < M * N; ++i) {
        result_data[i] *= alpha;
    }

    if (C && beta != 0.0f) {
        const float* C_data = C->typed_data<float>();
        for (int64_t i = 0; i < M * N; ++i) {
            result_data[i] += beta * C_data[i];
        }
    }

    return result;
}

// ============================================================================
// Public API: Binary × Float GEMM
// ============================================================================
Tensor gemm_binary(const Tensor& A_binary, const Tensor& B,
                   const Tensor* C, float alpha, float beta,
                   bool transA, bool transB) {
    TBN_LOG_DEBUG("gemm_binary: A=" + shape_to_string(A_binary.shape()) +
                  " B=" + shape_to_string(B.shape()));

    TBN_CHECK(!transA, NotImplementedError, "Transposed binary A not supported");
    TBN_CHECK(!transB, NotImplementedError, "Transposed B not supported");
    TBN_CHECK(A_binary.dtype() == DataType::BINARY, InvalidArgumentError,
              "A must be binary");
    TBN_CHECK(B.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "B must be float32");

    // Convert A_binary to ternary (binary is subset of ternary)
    Tensor A_ternary(A_binary.shape(), DataType::TERNARY);
    const BinaryWeight* bw = A_binary.typed_data<BinaryWeight>();
    TernaryWeight* tw = A_ternary.typed_data<TernaryWeight>();

    for (int64_t i = 0; i < A_binary.num_elements(); ++i) {
        tw[i] = (bw[i] == BINARY_ONE) ? TERNARY_PLUS_ONE : TERNARY_MINUS_ONE;
    }

    // Convert B to binary for optimized path
    Tensor B_binary(B.shape(), DataType::BINARY);
    BinaryWeight* bwb = B_binary.typed_data<BinaryWeight>();
    const float* b_data = B.typed_data<float>();
    for (int64_t i = 0; i < B.num_elements(); ++i) {
        bwb[i] = (b_data[i] >= 0) ? BINARY_ONE : BINARY_ZERO;
    }

    // Use optimized path
    Tensor result = gemm_ternary_binary_optimized(A_ternary, B_binary, 1.0f);

    // Apply alpha and add C
    float* result_data = result.typed_data<float>();
    int64_t M = result.shape().dims[0];
    int64_t N = result.shape().dims[1];

    for (int64_t i = 0; i < M * N; ++i) {
        result_data[i] *= alpha;
    }

    if (C && beta != 0.0f) {
        const float* C_data = C->typed_data<float>();
        for (int64_t i = 0; i < M * N; ++i) {
            result_data[i] += beta * C_data[i];
        }
    }

    return result;
}

// ============================================================================
// Public API: Mixed precision GEMM
// ============================================================================
Tensor gemm_mixed(const Tensor& A, const Tensor& B,
                  const Tensor* C, float alpha, float beta,
                  bool transA, bool transB) {
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

// ============================================================================
// Batch GEMM
// ============================================================================
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

    return results[0];
}

// ============================================================================
// Strided Batch GEMM
// ============================================================================
Tensor strided_batch_gemm(const Tensor& A, const Tensor& B, const Tensor* C,
                          int64_t batch_size, int64_t strideA, int64_t strideB,
                          float alpha, float beta,
                          bool transA, bool transB) {
    // Simplified implementation
    return gemm(A, B, C, alpha, beta, transA, transB);
}

// ============================================================================
// Auto-tuning
// ============================================================================
GemmParams auto_tune_gemm(int64_t M, int64_t N, int64_t K) {
    GemmParams params;
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
    return gemm(A, B, C, 1.0f, 0.0f, false, false);
}

// ============================================================================
// Helper
// ============================================================================
std::string shape_to_string(const Shape& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape.dims[i]);
    }
    result += "]";
    return result;
}

} // namespace tbn
