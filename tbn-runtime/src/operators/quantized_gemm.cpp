#include "../../include/tbn/operators/quantized_gemm.hpp"
#include "../../include/tbn/operators/gemm.hpp"
#include "../../include/tbn/utils/errors.hpp"
#include "../../include/tbn/utils/logging.hpp"
#include "../../include/tbn/memory/packed_weights.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

namespace tbn {

// ============================================================================
// Helper function to extract quantization parameters
// ============================================================================
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

// ============================================================================
// Standard quantized matrix multiplication (int8 x int8)
// ============================================================================
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

    // Use naive implementation
    return impl::qlinear_matmul_naive(a, config.a_scales[0], config.a_zero_points[0],
                                     b, config.b_scales[0], config.b_zero_points[0],
                                     config.y_scale, config.y_zero_point);
}

// ============================================================================
// Naive reference implementation for int8 x int8
// ============================================================================
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

// ============================================================================
// Packed weight implementations (fallback via dequantization)
// ============================================================================

Tensor qlinear_matmul_ternary_packed(const Tensor& a,
                                     const TernaryPackedWeights& b_packed,
                                     float b_scale) {
    TBN_LOG_INFO("Using packed ternary matrix multiplication");

    // Unpack weights to tensor format
    Tensor b_ternary(b_packed.shape(), DataType::TERNARY);
    TernaryWeight* b_data = b_ternary.typed_data<TernaryWeight>();

    for (int64_t i = 0; i < b_packed.size(); ++i) {
        b_data[i] = b_packed.get_weight(i);
    }

    // Call the main implementation
    return qlinear_matmul_ternary(a, b_ternary, b_scale, TilingParams::default_128x128());
}

Tensor qlinear_matmul_binary_packed(const Tensor& a,
                                    const BinaryPackedWeights& b_packed,
                                    float b_scale) {
    TBN_LOG_INFO("Using packed binary matrix multiplication");

    // Unpack weights to tensor format
    Tensor b_binary(b_packed.shape(), DataType::BINARY);
    BinaryWeight* b_data = b_binary.typed_data<BinaryWeight>();

    for (int64_t i = 0; i < b_packed.size(); ++i) {
        b_data[i] = b_packed.get_weight(i);
    }

    // Call the main implementation
    return qlinear_matmul_binary(a, b_binary, b_scale, TilingParams::default_128x128());
}

} // namespace impl

// ============================================================================
// ONNX integration helpers
// ============================================================================

Tensor quantize_linear_matmul(const Tensor& a, const Tensor& b,
                              float a_scale, float b_scale,
                              float y_scale, int8_t y_zero_point) {
    // Dequantize B matrix
    Tensor b_float(b.shape(), DataType::FLOAT32);
    const int8_t* b_data = b.typed_data<int8_t>();
    float* b_float_data = b_float.typed_data<float>();

    for (int64_t i = 0; i < b.num_elements(); ++i) {
        b_float_data[i] = (b_data[i] - 0) * b_scale;
    }

    return gemm(a, b_float);
}

Tensor quantize_linear_matmul_ternary(const Tensor& a, const Tensor& b_ternary,
                                      float b_scale) {
    return qlinear_matmul_ternary(a, b_ternary, b_scale, TilingParams::default_128x128());
}

Tensor quantize_linear_matmul_binary(const Tensor& a, const Tensor& b_binary,
                                     float b_scale) {
    return qlinear_matmul_binary(a, b_binary, b_scale, TilingParams::default_128x128());
}

// ONNX operator wrappers
Tensor onnx_qlinear_matmul(const Tensor& a, const Tensor& a_scale, const Tensor& a_zero_point,
                          const Tensor& b, const Tensor& b_scale, const Tensor& b_zero_point,
                          float y_scale, const Tensor& y_zero_point) {
    return qlinear_matmul(a, a_scale, a_zero_point,
                         b, b_scale, b_zero_point,
                         y_scale, y_zero_point);
}

Tensor onnx_qlinear_matmul_ternary(const Tensor& a, const Tensor& b_ternary, float b_scale) {
    return qlinear_matmul_ternary(a, b_ternary, b_scale, TilingParams::default_128x128());
}

Tensor onnx_qlinear_matmul_binary(const Tensor& a, const Tensor& b_binary, float b_scale) {
    return qlinear_matmul_binary(a, b_binary, b_scale, TilingParams::default_128x128());
}

} // namespace tbn
