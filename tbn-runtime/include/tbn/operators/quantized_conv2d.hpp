#pragma once

#include "../runtime/tensor.hpp"
#include "../runtime/types.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "conv2d.hpp"

#include <unordered_map>

namespace tbn {

/**
 * @brief ONNX QLinearConv operator implementation
 *
 * Performs quantized 2D convolution with integer tensors.
 * Supports both per-tensor and per-channel quantization.
 *
 * Inputs:
 * - x: Quantized input tensor (N x C x H x W)
 * - x_scale: Scale for input (float, scalar)
 * - x_zero_point: Zero point for input (same type as x, scalar)
 * - w: Quantized weights tensor (M x C/group x kH x kW)
 * - w_scale: Scale for weights (float, scalar or 1D for per-channel)
 * - w_zero_point: Zero point for weights (same type as w, scalar or 1D)
 * - y_scale: Scale for output (float, scalar)
 * - y_zero_point: Zero point for output (same type as y, scalar)
 * - bias (optional): Quantized bias (int32, 1D)
 *
 * Output:
 * - y: Quantized output tensor
 */

// Standard quantized Conv2D (8-bit input/weights)
Tensor qlinear_conv2d(const Tensor& x, const Tensor& x_scale, const Tensor& x_zero_point,
                     const Tensor& w, const Tensor& w_scale, const Tensor& w_zero_point,
                     const Tensor& y_scale, const Tensor& y_zero_point,
                     const Conv2DParams& params = Conv2DParams());

// With bias
Tensor qlinear_conv2d_bias(const Tensor& x, float x_scale, const Tensor& x_zero_point,
                          const Tensor& w, const Tensor& w_scale, const Tensor& w_zero_point,
                          float y_scale, const Tensor& y_zero_point,
                          const Tensor& bias,
                          const Conv2DParams& params = Conv2DParams());

// Ternary-weight quantized Conv2D (ternary weights, float input)
Tensor qlinear_conv2d_ternary(const Tensor& x,
                             const Tensor& w_ternary,
                             float w_scale,
                             const Conv2DParams& params = Conv2DParams());

// Binary-weight quantized Conv2D (binary weights, float input)
Tensor qlinear_conv2d_binary(const Tensor& x,
                            const Tensor& w_binary,
                            float w_scale,
                            const Conv2DParams& params = Conv2DParams());

// Per-channel quantization support
Tensor qlinear_conv2d_per_channel(const Tensor& x, float x_scale, const Tensor& x_zero_point,
                                 const Tensor& w, const std::vector<float>& w_scales,
                                 const Tensor& w_zero_points,
                                 float y_scale, const Tensor& y_zero_point,
                                 const Conv2DParams& params = Conv2DParams());

// Optimized implementations
namespace impl {
    // Naive reference implementation
    Tensor qlinear_conv2d_naive(const Tensor& x, float x_scale, int8_t x_zero_point,
                               const Tensor& w, const std::vector<float>& w_scales,
                               const std::vector<int8_t>& w_zero_points,
                               float y_scale, int8_t y_zero_point,
                               const Conv2DParams& params);

    // ARM NEON optimized implementation
    Tensor qlinear_conv2d_neon(const Tensor& x, float x_scale, int8_t x_zero_point,
                              const Tensor& w, const std::vector<float>& w_scales,
                              const std::vector<int8_t>& w_zero_points,
                              float y_scale, int8_t y_zero_point,
                              const Conv2DParams& params);

    // Ternary-weight optimized implementation
    Tensor qlinear_conv2d_ternary_packed(const Tensor& x,
                                        const Tensor& w_packed,
                                        float w_scale,
                                        const Conv2DParams& params);

    // Binary-weight optimized implementation
    Tensor qlinear_conv2d_binary_packed(const Tensor& x,
                                       const Tensor& w_packed,
                                       float w_scale,
                                       const Conv2DParams& params);
}

// Helper function to convert ONNX QLinearConv attributes
Conv2DParams extract_conv2d_params_from_onnx(const std::unordered_map<std::string, Tensor>& attrs);

// Helper function to convert shape to string
std::string shape_to_string(const Shape& shape);

// Helper function to extract quantization parameters
struct QuantizationConfig {
    float x_scale;
    int8_t x_zero_point;
    std::vector<float> w_scales;  // Per-channel or single scale
    std::vector<int8_t> w_zero_points;
    float y_scale;
    int8_t y_zero_point;
};

QuantizationConfig extract_quantization_config(const Tensor& x_scale, const Tensor& x_zero_point,
                                              const Tensor& w_scale, const Tensor& w_zero_point,
                                              float y_scale, const Tensor& y_zero_point);

} // namespace tbn