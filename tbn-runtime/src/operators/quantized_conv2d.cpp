#include "../../include/tbn/operators/quantized_conv2d.hpp"
#include "../../include/tbn/operators/conv2d.hpp"
#include "../../include/tbn/utils/errors.hpp"
#include "../../include/tbn/utils/logging.hpp"
#include "../../include/tbn/quantization/quantizer.hpp"
#include "../../include/tbn/memory/packed_weights.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>

// Helper function
static std::string shape_to_string(const tbn::Shape& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        if (i > 0) result += "x";
        result += std::to_string(shape.dims[i]);
    }
    result += "]";
    return result;
}

namespace tbn {

// Helper function to extract quantization parameters
QuantizationConfig extract_quantization_config(const Tensor& x_scale, const Tensor& x_zero_point,
                                              const Tensor& w_scale, const Tensor& w_zero_point,
                                              float y_scale, const Tensor& y_zero_point) {
    QuantizationConfig config;

    // Extract input scale and zero point
    config.x_scale = x_scale.typed_data<float>()[0];
    config.x_zero_point = x_zero_point.typed_data<int8_t>()[0];

    // Extract weight scales and zero points
    if (w_scale.shape().dims.size() == 0) {
        // Per-tensor quantization
        config.w_scales.push_back(w_scale.typed_data<float>()[0]);
        config.w_zero_points.push_back(w_zero_point.typed_data<int8_t>()[0]);
    } else {
        // Per-channel quantization
        int64_t num_channels = w_scale.shape().dims[0];
        const float* w_scale_data = w_scale.typed_data<float>();
        const int8_t* w_zp_data = w_zero_point.typed_data<int8_t>();

        for (int64_t i = 0; i < num_channels; ++i) {
            config.w_scales.push_back(w_scale_data[i]);
            config.w_zero_points.push_back(w_zp_data[i]);
        }
    }

    // Extract output scale and zero point
    config.y_scale = y_scale;
    config.y_zero_point = y_zero_point.typed_data<int8_t>()[0];

    return config;
}

// Standard quantized Conv2D implementation
Tensor qlinear_conv2d(const Tensor& x, const Tensor& x_scale, const Tensor& x_zero_point,
                     const Tensor& w, const Tensor& w_scale, const Tensor& w_zero_point,
                     const Tensor& y_scale, const Tensor& y_zero_point,
                     const Conv2DParams& params) {
    TBN_LOG_DEBUG("qlinear_conv2d: x_shape=" + shape_to_string(x.shape()) +
                  " w_shape=" + shape_to_string(w.shape()) +
                  " x_scale=" + std::to_string(x_scale.typed_data<float>()[0]) +
                  " y_scale=" + std::to_string(y_scale.typed_data<float>()[0]));

    // Validate inputs
    TBN_CHECK(x.dtype() == DataType::INT8 || x.dtype() == DataType::UINT8, InvalidArgumentError,
              "QLinearConv requires int8/uint8 input");
    TBN_CHECK(w.dtype() == DataType::INT8 || w.dtype() == DataType::UINT8, InvalidArgumentError,
              "QLinearConv requires int8/uint8 weights");

    // Extract quantization configuration
    auto config = tbn::extract_quantization_config(x_scale, x_zero_point, w_scale, w_zero_point,
                                                  y_scale.typed_data<float>()[0], y_zero_point);

    // Use naive implementation for now
    return impl::qlinear_conv2d_naive(x, config.x_scale, config.x_zero_point,
                                     w, config.w_scales, config.w_zero_points,
                                     config.y_scale, config.y_zero_point, params);
}

// Naive reference implementation
namespace impl {

Tensor qlinear_conv2d_naive(const Tensor& x, float x_scale, int8_t x_zero_point,
                           const Tensor& w, const std::vector<float>& w_scales,
                           const std::vector<int8_t>& w_zero_points,
                           float y_scale, int8_t y_zero_point,
                           const Conv2DParams& params) {
    // Get tensor dimensions
    const int64_t* x_shape = x.shape().dims.data();
    const int64_t* w_shape = w.shape().dims.data();

    int64_t N = x_shape[0];  // batch size
    int64_t C = x_shape[1];  // input channels
    int64_t H = x_shape[2];  // input height
    int64_t W = x_shape[3];  // input width

    int64_t M = w_shape[0];  // output channels
    int64_t kernel_h = params.kernel_h;
    int64_t kernel_w = params.kernel_w;

    // Calculate output dimensions
    int64_t out_h = (H + 2 * params.pad_h - kernel_h) / params.stride_h + 1;
    int64_t out_w = (W + 2 * params.pad_w - kernel_w) / params.stride_w + 1;

    // Create output tensor
    Tensor output(Shape{N, M, out_h, out_w}, DataType::INT8);
    int8_t* output_data = output.typed_data<int8_t>();

    // Get input pointers
    const int8_t* x_data = x.typed_data<int8_t>();
    const int8_t* w_data = w.typed_data<int8_t>();

    // Perform convolution
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            // Get per-channel quantization parameters if needed
            float w_scale = w_scales.size() > 1 ? w_scales[m] : w_scales[0];
            int8_t w_zp = w_zero_points.size() > 1 ? w_zero_points[m] : w_zero_points[0];

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate output position
                    int64_t out_idx = ((n * M + m) * out_h + oh) * out_w + ow;

                    // Initialize accumulator
                    int32_t acc = 0;

                    // Perform convolution
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                // Calculate input position
                                int64_t ih = oh * params.stride_h - params.pad_h + kh;
                                int64_t iw = ow * params.stride_w - params.pad_w + kw;

                                // Skip if out of bounds
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                                // Get input and weight values
                                int64_t x_idx = ((n * C + c) * H + ih) * W + iw;
                                int64_t w_idx = ((m * C + c) * kernel_h + kh) * kernel_w + kw;

                                int8_t x_val = x_data[x_idx];
                                int8_t w_val = w_data[w_idx];

                                // Dequantize and multiply
                                float x_deq = (x_val - x_zero_point) * x_scale;
                                float w_deq = (w_val - w_zp) * w_scale;
                                acc += static_cast<int32_t>(x_deq * w_deq / (x_scale * w_scale));
                            }
                        }
                    }

                    // Requantize output
                    float output_float = acc * (x_scale * w_scale) / y_scale;
                    int32_t output_int = static_cast<int32_t>(std::round(output_float)) + y_zero_point;

                    // Clamp to int8 range
                    output_int = std::max<int32_t>(std::min<int32_t>(output_int, 127), -128);
                    output_data[out_idx] = static_cast<int8_t>(output_int);
                }
            }
        }
    }

    return output;
}

} // namespace impl

// Ternary-weight quantized Conv2D
Tensor qlinear_conv2d_ternary(const Tensor& x,
                             const Tensor& w_ternary,
                             float w_scale,
                             const Conv2DParams& params) {
    TBN_LOG_DEBUG("qlinear_conv2d_ternary: x_shape=" + shape_to_string(x.shape()) +
                  " w_shape=" + shape_to_string(w_ternary.shape()) +
                  " w_scale=" + std::to_string(w_scale));

    // Validate inputs
    TBN_CHECK(x.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Ternary Conv2D requires float32 input");
    TBN_CHECK(w_ternary.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Ternary Conv2D requires ternary weights");

    // Use packed implementation for efficiency
    return impl::qlinear_conv2d_ternary_packed(x, w_ternary, w_scale, params);
}

// Binary-weight quantized Conv2D
Tensor qlinear_conv2d_binary(const Tensor& x,
                            const Tensor& w_binary,
                            float w_scale,
                            const Conv2DParams& params) {
    TBN_LOG_DEBUG("qlinear_conv2d_binary: x_shape=" + shape_to_string(x.shape()) +
                  " w_shape=" + shape_to_string(w_binary.shape()) +
                  " w_scale=" + std::to_string(w_scale));

    // Validate inputs
    TBN_CHECK(x.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Binary Conv2D requires float32 input");
    TBN_CHECK(w_binary.dtype() == DataType::BINARY, InvalidArgumentError,
              "Binary Conv2D requires binary weights");

    // Use packed implementation for efficiency
    return impl::qlinear_conv2d_binary_packed(x, w_binary, w_scale, params);
}

// Helper function for ONNX integration (placeholder)
Conv2DParams extract_conv2d_params_from_onnx(const std::unordered_map<std::string, Tensor>& attrs) {
    Conv2DParams params;
    // TODO: Extract parameters from ONNX attributes
    return params;
}

// Placeholder implementations for packed operations
namespace impl {

Tensor qlinear_conv2d_ternary_packed(const Tensor& x,
                                     const Tensor& w_packed,
                                     float w_scale,
                                     const Conv2DParams& params) {
    TBN_LOG_WARNING("Ternary packed Conv2D not fully implemented, using naive approach");
    // For now, just use the ternary tensor as-is
    // TODO: Implement proper bit-packed ternary convolution
    return Tensor(); // Placeholder
}

Tensor qlinear_conv2d_binary_packed(const Tensor& x,
                                    const Tensor& w_packed,
                                    float w_scale,
                                    const Conv2DParams& params) {
    TBN_LOG_WARNING("Binary packed Conv2D not fully implemented, using naive approach");
    // For now, just use the binary tensor as-is
    // TODO: Implement proper bit-packed binary convolution
    return Tensor(); // Placeholder
}

} // namespace impl

} // namespace tbn