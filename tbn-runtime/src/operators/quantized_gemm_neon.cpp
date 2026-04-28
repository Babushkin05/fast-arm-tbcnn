#include "../../include/tbn/operators/quantized_gemm.hpp"
#include "../../include/tbn/operators/gemm.hpp"
#include "../../include/tbn/utils/errors.hpp"
#include "../../include/tbn/utils/logging.hpp"

// Include the optimized GeMM from 05-final
#include "../../../GeMM/05-final/GeMM.hpp"

#include <vector>
#include <cstring>
#include <span>

namespace tbn {

// Forward declarations
Tensor qlinear_matmul_binary_float(const Tensor& a, const Tensor& b_binary_float,
                                   float scale, const TilingParams& params);
Tensor qlinear_matmul_binary_blocked(const Tensor& a, const Tensor& b_binary,
                                     float scale, const TilingParams& params,
                                     float threshold_low, float threshold_high);

// ============================================================================
// Configuration
// ============================================================================
constexpr int64_t GEMM_THRESHOLD = 64;  // Use blocked GeMM for matrices >= 64

// Flag to control activation quantization in blocked GeMM
// Set to false for Float × Binary (keep activations as float)
constexpr bool QUANTIZE_ACTIVATIONS_BY_DEFAULT = false;

// Default thresholds for float → ternary quantization (when enabled)
constexpr float DEFAULT_THRESHOLD_LOW = -0.1f;
constexpr float DEFAULT_THRESHOLD_HIGH = 0.1f;

// ============================================================================
// Helper: Pad dimensions to meet GeMM requirements
// ============================================================================
struct PaddedDimensions {
    uint32_t m, n, k;
    uint32_t m_padded, n_padded, k_padded;
    bool needs_padding;

    PaddedDimensions(uint32_t m_, uint32_t n_, uint32_t k_)
        : m(m_), n(n_), k(k_) {
        // GeMM.cpp expects:
        // - m (A.rows): multiple of mmk (16)
        // - n (A.cols = our k): multiple of 128 for NEON
        // - k (B.cols = our n): multiple of nmk (8)
        //
        // Also TernaryMatrix/BinaryMatrix requirements:
        // - TernaryMatrix: rows multiple of 8, cols multiple of 64
        // - BinaryMatrix: rows multiple of 64, cols multiple of 8
        m_padded = ((m + 15) / 16) * 16;     // multiple of 16 (mmk)
        k_padded = ((k + 127) / 128) * 128;  // multiple of 128 (GeMM's "n" requirement)
        n_padded = ((n + 7) / 8) * 8;        // multiple of 8 (nmk and BinaryMatrix cols)
        needs_padding = (m != m_padded) || (n != n_padded) || (k != k_padded);
    }
};

// ============================================================================
// Naive implementation (fallback for small matrices)
// ============================================================================
Tensor qlinear_matmul_binary_naive(
    const Tensor& a,
    const Tensor& b_binary,
    float scale
) {
    int64_t M = a.shape().dims[0];
    int64_t K = a.shape().dims[1];
    int64_t N = b_binary.shape().dims[1];

    Tensor output({M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();
    const float* a_data = a.typed_data<float>();
    const BinaryWeight* b_data = b_binary.typed_data<BinaryWeight>();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // Binary: 0 = -1, 1 = +1
                float sign = (b_data[k * N + j] == BINARY_ONE) ? 1.0f : -1.0f;
                sum += a_data[i * K + k] * sign;
            }
            out_data[i * N + j] = sum * scale;
        }
    }

    return output;
}

// ============================================================================
// Quantize float to ternary on-the-fly during packing
// ============================================================================
std::vector<int8_t> quantize_float_to_ternary(
    const float* data,
    size_t count,
    float threshold_low = DEFAULT_THRESHOLD_LOW,
    float threshold_high = DEFAULT_THRESHOLD_HIGH
) {
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
// Pad float matrix with zeros
// ============================================================================
Tensor pad_float_matrix(
    const Tensor& tensor,
    uint32_t target_rows,
    uint32_t target_cols
) {
    uint32_t rows = static_cast<uint32_t>(tensor.shape().dims[0]);
    uint32_t cols = static_cast<uint32_t>(tensor.shape().dims[1]);

    if (rows == target_rows && cols == target_cols) {
        return tensor;
    }

    Tensor padded({target_rows, target_cols}, DataType::FLOAT32);
    float* dst = padded.typed_data<float>();
    const float* src = tensor.typed_data<float>();

    std::memset(dst, 0, static_cast<size_t>(target_rows) * target_cols * sizeof(float));

    for (uint32_t i = 0; i < rows; ++i) {
        std::memcpy(dst + i * target_cols, src + i * cols, cols * sizeof(float));
    }

    return padded;
}

// ============================================================================
// Convert binary tensor to int8_t array for GeMM
// BinaryWeight encoding: BINARY_ZERO=0 → -1, BINARY_ONE=1 → +1
// ============================================================================
std::vector<int8_t> convert_binary_to_int8(const Tensor& b_binary) {
    int64_t size = b_binary.num_elements();
    std::vector<int8_t> result(size);

    const BinaryWeight* b_data = b_binary.typed_data<BinaryWeight>();
    for (int64_t i = 0; i < size; ++i) {
        // BINARY_ZERO(0) → -1, BINARY_ONE(1) → +1
        result[i] = (b_data[i] == BINARY_ONE) ? +1 : -1;
    }

    return result;
}

// ============================================================================
// Blocked GeMM implementation using GeMM/05-final
// NOTE: This quantizes activations to ternary, which may reduce accuracy
// For Float × Binary without quantization, use naive path instead
// ============================================================================
Tensor qlinear_matmul_binary_blocked(
    const Tensor& a,
    const Tensor& b_binary,
    float scale,
    const TilingParams& params,
    float threshold_low = DEFAULT_THRESHOLD_LOW,
    float threshold_high = DEFAULT_THRESHOLD_HIGH
) {
    uint32_t M = static_cast<uint32_t>(a.shape().dims[0]);
    uint32_t K = static_cast<uint32_t>(a.shape().dims[1]);
    uint32_t N = static_cast<uint32_t>(b_binary.shape().dims[1]);

    // Calculate padded dimensions
    PaddedDimensions dims(M, N, K);

    TBN_LOG_DEBUG("qlinear_matmul_binary_blocked: M=" + std::to_string(M) +
                  " K=" + std::to_string(K) + " N=" + std::to_string(N));
    TBN_LOG_DEBUG("Padded: m=" + std::to_string(dims.m_padded) +
                  " k=" + std::to_string(dims.k_padded) + " n=" + std::to_string(dims.n_padded));

    // Pad matrices if needed
    Tensor a_padded = pad_float_matrix(a, dims.m_padded, dims.k_padded);

    // Fused: quantize and pack A directly (no intermediate int8_t buffer)
    TernaryMatrix a_packed = TernaryMatrix::pack_from_float(
        a_padded.typed_data<float>(),
        dims.m_padded,
        dims.k_padded,
        threshold_low,
        threshold_high
    );

    // Convert B to int8_t binary format (still needed for weights, but this is one-time)
    // Need to pad B to (k_padded × n_padded)
    std::vector<int8_t> b_int8(static_cast<size_t>(dims.k_padded) * dims.n_padded, 1);  // default +1

    const BinaryWeight* b_data = b_binary.typed_data<BinaryWeight>();
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            // BINARY_ZERO(0) → -1, BINARY_ONE(1) → +1
            b_int8[i * dims.n_padded + j] = (b_data[i * N + j] == BINARY_ONE) ? +1 : -1;
        }
    }

    // Pack B for GeMM
    BinaryMatrix b_packed = BinaryMatrix::pack(
        std::span<const int8_t>(b_int8.data(), b_int8.size()),
        dims.k_padded,
        dims.n_padded
    );

    // Run GeMM with provided tiling parameters
    GemmEngine engine;
    Int32Matrix result = engine.compute(a_packed.view(), b_packed.view(), params);

    // Extract the valid (non-padded) region and apply scale
    // Use memcpy for contiguous rows when no padding on N dimension
    Tensor output({M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();

    if (N == dims.n_padded) {
        // No padding on N — copy rows with memcpy
        for (uint32_t i = 0; i < M; ++i) {
            const std::int32_t* src = result.data().data() + i * dims.n_padded;
            float* dst = out_data + i * N;
            for (uint32_t j = 0; j < N; ++j) {
                dst[j] = static_cast<float>(src[j]) * scale;
            }
        }
    } else {
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                out_data[i * N + j] = static_cast<float>(result.at(i, j)) * scale;
            }
        }
    }

    return output;
}

// ============================================================================
// Blocked GeMM with pre-packed BinaryMatrix (cached weights)
// Skips the B-side int8 conversion + BinaryMatrix::pack
// ============================================================================
Tensor qlinear_matmul_binary_blocked_packed(
    const Tensor& a,
    const BinaryMatrix& b_packed,
    uint32_t n_orig,
    float scale,
    const TilingParams& params,
    float threshold_low,
    float threshold_high
) {
    uint32_t M = static_cast<uint32_t>(a.shape().dims[0]);
    uint32_t K = static_cast<uint32_t>(a.shape().dims[1]);
    uint32_t N = n_orig;
    uint32_t k_padded = b_packed.rows();
    uint32_t n_padded = b_packed.cols();

    // Compute m_padded (depends on activation, varies per inference)
    uint32_t m_padded = ((M + 15) / 16) * 16;

    TBN_LOG_DEBUG("qlinear_matmul_binary_blocked_packed: M=" + std::to_string(M) +
                  " K=" + std::to_string(K) + " N=" + std::to_string(N) +
                  " (pre-packed B: k_pad=" + std::to_string(k_padded) +
                  " n_pad=" + std::to_string(n_padded) + ")");

    // Pad and quantize A (activation — still per-inference)
    Tensor a_padded = pad_float_matrix(a, m_padded, k_padded);
    TernaryMatrix a_packed = TernaryMatrix::pack_from_float(
        a_padded.typed_data<float>(),
        m_padded,
        k_padded,
        threshold_low,
        threshold_high
    );

    // B is already packed — skip int8 conversion and BinaryMatrix::pack!

    // Run GeMM with pre-packed B
    GemmEngine engine;
    Int32Matrix result = engine.compute(a_packed.view(), b_packed.view(), params);

    // Extract the valid (non-padded) region and apply scale
    Tensor output({M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();

    if (N == n_padded) {
        for (uint32_t i = 0; i < M; ++i) {
            const std::int32_t* src = result.data().data() + i * n_padded;
            float* dst = out_data + i * N;
            for (uint32_t j = 0; j < N; ++j) {
                dst[j] = static_cast<float>(src[j]) * scale;
            }
        }
    } else {
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                out_data[i * N + j] = static_cast<float>(result.at(i, j)) * scale;
            }
        }
    }

    return output;
}

// ============================================================================
// GEMM with both A and B pre-packed — zero conversions
// ============================================================================
Tensor qlinear_matmul_binary_blocked_prepacked(
    const TernaryMatrix& a_packed,
    const BinaryMatrix& b_packed,
    uint32_t m_orig, uint32_t n_orig,
    float scale,
    const TilingParams& params
) {
    uint32_t M = m_orig;
    uint32_t N = n_orig;
    uint32_t n_padded = b_packed.cols();

    // Both matrices already packed — skip all conversions
    GemmEngine engine;
    Int32Matrix result = engine.compute(a_packed.view(), b_packed.view(), params);

    // Extract the valid region
    Tensor output({M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();

    if (N == n_padded) {
        for (uint32_t i = 0; i < M; ++i) {
            const std::int32_t* src = result.data().data() + i * n_padded;
            float* dst = out_data + i * N;
            for (uint32_t j = 0; j < N; ++j) {
                dst[j] = static_cast<float>(src[j]) * scale;
            }
        }
    } else {
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                out_data[i * N + j] = static_cast<float>(result.at(i, j)) * scale;
            }
        }
    }

    return output;
}

// ============================================================================
// Helper: Check if float weights are already binary (all values are -1 or +1)
// ============================================================================
bool is_binary_float_weights(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        if (val != -1.0f && val != 1.0f && val != 0.0f) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Helper: Convert binary float weights to int8_t directly (no quantization needed)
// ============================================================================
std::vector<int8_t> convert_binary_float_to_int8(const float* data, size_t count) {
    std::vector<int8_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        // 0.0f is treated as +1 (or could be -1)
        result[i] = (data[i] >= 0.0f) ? +1 : -1;
    }
    return result;
}

// ============================================================================
// Main entry point for binary weights
// ============================================================================
Tensor qlinear_matmul_binary(
    const Tensor& a,
    const Tensor& b_binary,
    float scale,
    const TilingParams& params
) {
    TBN_LOG_DEBUG("qlinear_matmul_binary: a_shape=" + shape_to_string(a.shape()) +
                  " b_shape=" + shape_to_string(b_binary.shape()) +
                  " scale=" + std::to_string(scale));

    // Validate inputs
    TBN_CHECK(a.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Input A must be FLOAT32, got " + std::to_string(static_cast<int>(a.dtype())));

    const auto& a_shape = a.shape();
    const auto& b_shape = b_binary.shape();

    if (a_shape.dims.size() != 2 || b_shape.dims.size() != 2) {
        throw InvalidShapeError("Both inputs must be 2D matrices");
    }

    int64_t M = a_shape.dims[0];
    int64_t K = a_shape.dims[1];
    int64_t K_b = b_shape.dims[0];
    int64_t N = b_shape.dims[1];

    if (K != K_b) {
        throw InvalidShapeError(
            "Matrix dimensions incompatible: A cols (" + std::to_string(K) +
            ") != B rows (" + std::to_string(K_b) + ")");
    }

    // Check if weights are float but already binary
    if (b_binary.dtype() == DataType::FLOAT32) {
        const float* b_data = b_binary.typed_data<float>();
        if (is_binary_float_weights(b_data, b_binary.num_elements())) {
            TBN_LOG_DEBUG("Detected pre-quantized binary float weights - using optimized path");
            return qlinear_matmul_binary_float(a, b_binary, scale, params);
        } else {
            // Need to quantize float weights to binary
            TBN_LOG_DEBUG("Quantizing float weights to binary");
            return qlinear_matmul_binary_blocked(a, b_binary, scale, params);
        }
    }

    TBN_CHECK(b_binary.dtype() == DataType::BINARY, InvalidArgumentError,
              "Input B must be BINARY or FLOAT32 with binary values");

    TBN_LOG_DEBUG("Using blocked GeMM implementation");
    return qlinear_matmul_binary_blocked(a, b_binary, scale, params);
}

// ============================================================================
// Optimized path for pre-quantized binary float weights (no re-quantization)
// ============================================================================
Tensor qlinear_matmul_binary_float(
    const Tensor& a,
    const Tensor& b_binary_float,
    float scale,
    const TilingParams& params
) {
    // This function assumes b_binary_float contains only -1.0f, 0.0f, or +1.0f values
    // It skips the quantization step entirely

    uint32_t M = static_cast<uint32_t>(a.shape().dims[0]);
    uint32_t K = static_cast<uint32_t>(a.shape().dims[1]);
    uint32_t N = static_cast<uint32_t>(b_binary_float.shape().dims[1]);

    PaddedDimensions dims(M, N, K);

    TBN_LOG_DEBUG("qlinear_matmul_binary_float: M=" + std::to_string(M) +
                  " K=" + std::to_string(K) + " N=" + std::to_string(N));

    // Pad A matrix
    Tensor a_padded = pad_float_matrix(a, dims.m_padded, dims.k_padded);

    // Quantize activations to ternary (this is still needed)
    std::vector<int8_t> a_ternary = quantize_float_to_ternary(
        a_padded.typed_data<float>(),
        static_cast<size_t>(dims.m_padded) * dims.k_padded
    );

    // Convert B directly without re-quantization (just type conversion)
    std::vector<int8_t> b_int8(static_cast<size_t>(dims.k_padded) * dims.n_padded, 1);
    const float* b_data = b_binary_float.typed_data<float>();

    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            // Direct conversion: -1.0f -> -1, 0.0f or +1.0f -> +1
            b_int8[i * dims.n_padded + j] = (b_data[i * N + j] >= 0.0f) ? +1 : -1;
        }
    }

    // Pack matrices for GeMM
    TernaryMatrix a_packed = TernaryMatrix::pack(
        std::span<const int8_t>(a_ternary.data(), a_ternary.size()),
        dims.m_padded, dims.k_padded
    );

    BinaryMatrix b_packed = BinaryMatrix::pack(
        std::span<const int8_t>(b_int8.data(), b_int8.size()),
        dims.k_padded, dims.n_padded
    );

    // Run GeMM
    GemmEngine engine;
    Int32Matrix result = engine.compute(a_packed.view(), b_packed.view(), params);

    // Extract valid region
    Tensor output({M, N}, DataType::FLOAT32);
    float* out_data = output.typed_data<float>();

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            out_data[i * N + j] = static_cast<float>(result.at(i, j)) * scale;
        }
    }

    return output;
}

// ============================================================================
// Ternary weights - convert to binary and use binary path
// This quantizes ternary {-1, 0, +1} to binary {-1, +1}
// Zero values are converted to +1 (or could use -1, configurable)
// ============================================================================
Tensor qlinear_matmul_ternary(
    const Tensor& a,
    const Tensor& b_ternary,
    float scale,
    const TilingParams& params
) {
    TBN_LOG_DEBUG("qlinear_matmul_ternary: a_shape=" + shape_to_string(a.shape()) +
                  " b_shape=" + shape_to_string(b_ternary.shape()));

    TBN_CHECK(a.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Input A must be FLOAT32");
    TBN_CHECK(b_ternary.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Input B must be TERNARY");

    // Convert ternary to binary
    // -1 → -1 (keep as is)
    //  0 → +1 (or -1, we choose +1 as default)
    // +1 → +1 (keep as is)
    int64_t K = b_ternary.shape().dims[0];
    int64_t N = b_ternary.shape().dims[1];

    Tensor b_binary({K, N}, DataType::BINARY);
    BinaryWeight* b_bin_data = b_binary.typed_data<BinaryWeight>();
    const TernaryWeight* b_tern_data = b_ternary.typed_data<TernaryWeight>();

    for (int64_t i = 0; i < K * N; ++i) {
        // Map ternary to binary: -1→0 (means -1 in binary), 0→1, +1→1
        b_bin_data[i] = (b_tern_data[i] == TERNARY_MINUS_ONE) ? BINARY_ZERO : BINARY_ONE;
    }

    // Use binary path
    return qlinear_matmul_binary(a, b_binary, scale, params);
}

// ============================================================================
// Quantize any tensor to binary weights
// Used when loading ONNX models with non-binary weights
// ============================================================================
Tensor quantize_to_binary(const Tensor& weights, float threshold) {
    TBN_LOG_DEBUG("Quantizing weights to binary with threshold=" + std::to_string(threshold));

    Tensor binary_weights(weights.shape(), DataType::BINARY);
    BinaryWeight* dst = binary_weights.typed_data<BinaryWeight>();

    if (weights.dtype() == DataType::FLOAT32) {
        const float* src = weights.typed_data<float>();
        for (int64_t i = 0; i < weights.num_elements(); ++i) {
            dst[i] = (src[i] > threshold) ? BINARY_ONE : BINARY_ZERO;
        }
    } else if (weights.dtype() == DataType::INT8) {
        const int8_t* src = weights.typed_data<int8_t>();
        for (int64_t i = 0; i < weights.num_elements(); ++i) {
            dst[i] = (src[i] > 0) ? BINARY_ONE : BINARY_ZERO;
        }
    } else {
        throw InvalidArgumentError("Unsupported dtype for binary quantization");
    }

    return binary_weights;
}

// ============================================================================
// Quantize any tensor to ternary weights
// ============================================================================
Tensor quantize_to_ternary(
    const Tensor& weights,
    float threshold_low,
    float threshold_high
) {
    TBN_LOG_INFO("Quantizing weights to ternary with thresholds=[" +
                 std::to_string(threshold_low) + ", " + std::to_string(threshold_high) + "]");

    Tensor ternary_weights(weights.shape(), DataType::TERNARY);
    TernaryWeight* dst = ternary_weights.typed_data<TernaryWeight>();

    if (weights.dtype() == DataType::FLOAT32) {
        const float* src = weights.typed_data<float>();
        for (int64_t i = 0; i < weights.num_elements(); ++i) {
            if (src[i] < threshold_low) {
                dst[i] = TERNARY_MINUS_ONE;
            } else if (src[i] > threshold_high) {
                dst[i] = TERNARY_PLUS_ONE;
            } else {
                dst[i] = TERNARY_ZERO;
            }
        }
    } else {
        throw InvalidArgumentError("Unsupported dtype for ternary quantization");
    }

    return ternary_weights;
}

} // namespace tbn
