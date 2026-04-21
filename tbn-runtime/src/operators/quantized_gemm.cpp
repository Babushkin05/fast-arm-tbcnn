#include "../../include/tbn/operators/quantized_gemm.hpp"
#include "../../include/tbn/operators/gemm.hpp"
#include "../../include/tbn/utils/errors.hpp"
#include "../../include/tbn/utils/logging.hpp"
#include "../../include/tbn/memory/packed_weights.hpp"
#include "../../include/tbn/quantization/quantizer.hpp"

using tbn::Tensor;
using tbn::Shape;
using tbn::DataType;
using tbn::TernaryPackedWeights;
using tbn::BinaryPackedWeights;
#include <cmath>
#include <algorithm>
#include <vector>

namespace tbn {

// Helper function to extract quantization parameters
MatMulQuantizationConfig extract_matmul_quantization_config(
    const Tensor& a_scale, const Tensor& a_zero_point,
    const Tensor& b_scale, const Tensor& b_zero_point,
    float y_scale, const Tensor& y_zero_point) {

    MatMulQuantizationConfig config;

    // Extract A quantization parameters
    if (a_scale.shape().dims.size() == 0) {
        // Per-tensor
        config.a_scales.push_back(a_scale.typed_data<float>()[0]);
        config.a_zero_points.push_back(a_zero_point.typed_data<int8_t>()[0]);
    } else {
        // Per-channel/row
        int64_t num_scales = a_scale.shape().size();
        const float* a_scale_data = a_scale.typed_data<float>();
        const int8_t* a_zp_data = a_zero_point.typed_data<int8_t>();

        for (int64_t i = 0; i < num_scales; ++i) {
            config.a_scales.push_back(a_scale_data[i]);
            config.a_zero_points.push_back(a_zp_data[i]);
        }
    }

    // Extract B quantization parameters
    if (b_scale.shape().dims.size() == 0) {
        // Per-tensor
        config.b_scales.push_back(b_scale.typed_data<float>()[0]);
        config.b_zero_points.push_back(b_zero_point.typed_data<int8_t>()[0]);
    } else {
        // Per-channel/column
        int64_t num_scales = b_scale.shape().size();
        const float* b_scale_data = b_scale.typed_data<float>();
        const int8_t* b_zp_data = b_zero_point.typed_data<int8_t>();

        for (int64_t i = 0; i < num_scales; ++i) {
            config.b_scales.push_back(b_scale_data[i]);
            config.b_zero_points.push_back(b_zp_data[i]);
        }
    }

    // Extract output quantization parameters
    config.y_scale = y_scale;
    config.y_zero_point = y_zero_point.typed_data<int8_t>()[0];

    return config;
}

// Standard quantized matrix multiplication
Tensor qlinear_matmul(const Tensor& a, const Tensor& a_scale, const Tensor& a_zero_point,
                     const Tensor& b, const Tensor& b_scale, const Tensor& b_zero_point,
                     float y_scale, const Tensor& y_zero_point) {
    TBN_LOG_DEBUG("qlinear_matmul: a_shape=" + shape_to_string(a.shape()) +
                  " b_shape=" + shape_to_string(b.shape()) +
                  " y_scale=" + std::to_string(y_scale));

    // Validate inputs
    TBN_CHECK(a.dtype() == DataType::INT8 || a.dtype() == DataType::UINT8, InvalidArgumentError,
              "QLinearMatMul requires int8/uint8 input A");
    TBN_CHECK(b.dtype() == DataType::INT8 || b.dtype() == DataType::UINT8, InvalidArgumentError,
              "QLinearMatMul requires int8/uint8 input B");

    // Extract quantization configuration
    auto config = extract_matmul_quantization_config(a_scale, a_zero_point,
                                                    b_scale, b_zero_point,
                                                    y_scale, y_zero_point);

    // Use naive implementation for now
    return impl::qlinear_matmul_naive(a, config.a_scales[0], config.a_zero_points[0],
                                     b, config.b_scales[0], config.b_zero_points[0],
                                     config.y_scale, config.y_zero_point);
}

// Naive reference implementation
namespace impl {

Tensor qlinear_matmul_naive(const Tensor& a, float a_scale, int8_t a_zero_point,
                           const Tensor& b, float b_scale, int8_t b_zero_point,
                           float y_scale, int8_t y_zero_point) {
    // Get matrix dimensions
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();

    TBN_CHECK(a_shape.dims.size() == 2, InvalidShapeError, "Input A must be 2D matrix");
    TBN_CHECK(b_shape.dims.size() == 2, InvalidShapeError, "Input B must be 2D matrix");

    int64_t M = a_shape.dims[0];
    int64_t K = a_shape.dims[1];
    int64_t N = b_shape.dims[1];

    TBN_CHECK(K == b_shape.dims[0], InvalidShapeError,
              "Matrix dimensions incompatible for multiplication");

    // Create output tensor
    Tensor output(Shape{M, N}, DataType::INT8);
    int8_t* output_data = output.typed_data<int8_t>();

    // Get input pointers
    const int8_t* a_data = a.typed_data<int8_t>();
    const int8_t* b_data = b.typed_data<int8_t>();

    // Perform matrix multiplication
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            // Initialize accumulator
            int32_t acc = 0;

            // Compute dot product
            for (int64_t k = 0; k < K; ++k) {
                int8_t a_val = a_data[i * K + k];
                int8_t b_val = b_data[k * N + j];

                // Dequantize and multiply
                float a_deq = (a_val - a_zero_point) * a_scale;
                float b_deq = (b_val - b_zero_point) * b_scale;
                acc += static_cast<int32_t>(a_deq * b_deq / (a_scale * b_scale));
            }

            // Requantize output
            float output_float = acc * (a_scale * b_scale) / y_scale;
            int32_t output_int = static_cast<int32_t>(std::round(output_float)) + y_zero_point;

            // Clamp to int8 range
            output_int = std::max<int32_t>(std::min<int32_t>(output_int, 127), -128);
            output_data[i * N + j] = static_cast<int8_t>(output_int);
        }
    }

    return output;
}

} // namespace impl

// Ternary-weight quantized MatMul
Tensor qlinear_matmul_ternary(const Tensor& a,
                             const Tensor& b_ternary,
                             float b_scale) {
    TBN_LOG_DEBUG("qlinear_matmul_ternary: a_shape=" + shape_to_string(a.shape()) +
                  " b_shape=" + shape_to_string(b_ternary.shape()) +
                  " b_scale=" + std::to_string(b_scale));

    // Validate inputs
    TBN_CHECK(a.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Ternary MatMul requires float32 input A");
    TBN_CHECK(b_ternary.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Ternary MatMul requires ternary weights B");

    // Convert ternary tensor to packed weights
    TernaryPackedWeights packed_weights(b_ternary.shape());
    const TernaryWeight* ternary_data = b_ternary.typed_data<TernaryWeight>();

    // Pack the weights
    for (int64_t i = 0; i < b_ternary.num_elements(); ++i) {
        packed_weights.set_weight(i, ternary_data[i]);
    }

    return impl::qlinear_matmul_ternary_packed(a, packed_weights, b_scale);
}

// Binary-weight quantized MatMul
Tensor qlinear_matmul_binary(const Tensor& a,
                            const Tensor& b_binary,
                            float b_scale) {
    TBN_LOG_DEBUG("qlinear_matmul_binary: a_shape=" + shape_to_string(a.shape()) +
                  " b_shape=" + shape_to_string(b_binary.shape()) +
                  " b_scale=" + std::to_string(b_scale));

    // Validate inputs
    TBN_CHECK(a.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Binary MatMul requires float32 input A");
    TBN_CHECK(b_binary.dtype() == DataType::BINARY, InvalidArgumentError,
              "Binary MatMul requires binary weights B");

    // Convert binary tensor to packed weights
    BinaryPackedWeights packed_weights(b_binary.shape());
    const BinaryWeight* binary_data = b_binary.typed_data<BinaryWeight>();

    // Pack the weights
    for (int64_t i = 0; i < b_binary.num_elements(); ++i) {
        packed_weights.set_weight(i, binary_data[i]);
    }

    return impl::qlinear_matmul_binary_packed(a, packed_weights, b_scale);
}

// Placeholder implementations for packed operations
namespace impl {

Tensor qlinear_matmul_ternary_packed(const Tensor& a,
                                     const TernaryPackedWeights& b_packed,
                                     float b_scale) {
    TBN_LOG_WARNING("Ternary packed MatMul not fully implemented, using naive approach");

    // For now, dequantize and use regular GeMM
    // TODO: Implement proper ternary matrix multiplication
    tbn::Tensor b_dequantized(b_packed.shape(), tbn::DataType::FLOAT32);
    float* b_data = b_dequantized.typed_data<float>();

    for (int64_t i = 0; i < b_packed.shape().size(); ++i) {
        TernaryWeight weight = b_packed.get_weight(i);
        b_data[i] = dequantize_ternary(weight) * b_scale;
    }

    return gemm(a, b_dequantized);
}

Tensor qlinear_matmul_binary_packed(const Tensor& a,
                                    const BinaryPackedWeights& b_packed,
                                    float b_scale) {
    TBN_LOG_WARNING("Binary packed MatMul not fully implemented, using naive approach");

    // For now, dequantize and use regular GeMM
    // TODO: Implement proper binary matrix multiplication
    Tensor b_dequantized = Tensor(b_packed.shape(), DataType::FLOAT32);
    float* b_data = b_dequantized.typed_data<float>();

    for (int64_t i = 0; i < b_packed.shape().size(); ++i) {
        BinaryWeight weight = b_packed.get_weight(i);
        b_data[i] = dequantize_binary(weight) * b_scale;
    }

    return gemm(a, b_dequantized);
}

} // namespace impl

} // namespace tbn

// Helper functions for ONNX integration
// These will be used by the ONNX parser to create QLinearMatMul nodes
Tensor quantize_linear_matmul(const Tensor& a, const Tensor& b,
                              float a_scale, float b_scale,
                              float y_scale, int8_t y_zero_point) {
    // This function would be called by the ONNX parser to create a QLinearMatMul node
    // For now, just call regular gemm with dequantized weights

    // Dequantize B matrix
    Tensor b_float(b.shape(), DataType::FLOAT32);
    const int8_t* b_data = b.typed_data<int8_t>();
    float* b_float_data = b_float.typed_data<float>();

    for (int64_t i = 0; i < b.num_elements(); ++i) {
        b_float_data[i] = (b_data[i] - 0) * b_scale; // Assuming zero_point is 0
    }

    return gemm(a, b_float);
}

// Version for ternary/binary weights
Tensor quantize_linear_matmul_ternary(const Tensor& a, const Tensor& b_ternary,
                                      float b_scale) {
    return qlinear_matmul_ternary(a, b_ternary, b_scale);
}

Tensor quantize_linear_matmul_binary(const Tensor& a, const Tensor& b_binary,
                                     float b_scale) {
    return qlinear_matmul_binary(a, b_binary, b_scale);
}

// Export functions for ONNX integration
// These will be registered in the ONNX operator mapping
Tensor onnx_qlinear_matmul(const Tensor& a, const Tensor& a_scale, const Tensor& a_zero_point,
                          const Tensor& b, const Tensor& b_scale, const Tensor& b_zero_point,
                          float y_scale, const Tensor& y_zero_point) {
    return qlinear_matmul(a, a_scale, a_zero_point,
                         b, b_scale, b_zero_point,
                         y_scale, y_zero_point);
}

Tensor onnx_qlinear_matmul_ternary(const Tensor& a, const Tensor& b_ternary, float b_scale) {
    return qlinear_matmul_ternary(a, b_ternary, b_scale);
}

Tensor onnx_qlinear_matmul_binary(const Tensor& a, const Tensor& b_binary, float b_scale) {
    return qlinear_matmul_binary(a, b_binary, b_scale);
}

// Notes for ARM NEON optimization:
// 1. Use SIMD instructions for parallel dequantization
// 2. Process 4x4 or 8x8 blocks for better cache utilization
// 3. Use fused multiply-add for accumulation
// 4. Implement bit manipulation for ternary/binary weights
// 5. Consider using ARM Compute Library for complex operations
// 6. Use lookup tables for dequantization when possible
// 7. Implement zero-point correction efficiently

// Performance optimization notes:
// 1. Pre-compute scale ratios to avoid division in inner loops
// 2. Use bit-packed weights for memory efficiency
// 3. Implement blocked matrix multiplication
// 4. Use prefetch instructions for cache optimization
// 5. Consider mixed precision for intermediate results
// 6. Implement parallel execution for large matrices
// 7. Use memory pools to avoid allocations in hot paths

// Testing strategy:
// 1. Compare results with reference float implementation
// 2. Test edge cases (zero scales, extreme values)
// 3. Verify quantization/dequantization round-trip
// 4. Test per-channel quantization
// 5. Benchmark against existing implementations
// 6. Test with real ONNX models
// 7. Verify numerical accuracy within acceptable bounds