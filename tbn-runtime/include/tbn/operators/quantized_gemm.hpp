#pragma once

#include "../runtime/tensor.hpp"
#include "../runtime/types.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "../memory/packed_weights.hpp"

namespace tbn {

/**
 * @brief ONNX QLinearMatMul operator implementation
 *
 * Performs quantized matrix multiplication with integer tensors.
 * Supports both per-tensor and per-channel quantization.
 *
 * Inputs:
 * - a: Quantized input tensor A (M x K)
 * - a_scale: Scale for input A (float, scalar or 1D)
 * - a_zero_point: Zero point for input A (same type as a, scalar or 1D)
 * - b: Quantized input tensor B (K x N)
 * - b_scale: Scale for input B (float, scalar or 1D)
 * - b_zero_point: Zero point for input B (same type as b, scalar or 1D)
 * - y_scale: Scale for output (float, scalar)
 * - y_zero_point: Zero point for output (same type as y, scalar)
 *
 * Output:
 * - y: Quantized output tensor (M x N)
 */

// Standard quantized matrix multiplication (8-bit input)
Tensor qlinear_matmul(const Tensor& a, const Tensor& a_scale, const Tensor& a_zero_point,
                     const Tensor& b, const Tensor& b_scale, const Tensor& b_zero_point,
                     float y_scale, const Tensor& y_zero_point);

// Ternary-weight quantized MatMul (ternary weights, float input)
Tensor qlinear_matmul_ternary(const Tensor& a,
                             const Tensor& b_ternary,
                             float b_scale);

// Binary-weight quantized MatMul (binary weights, float input)
Tensor qlinear_matmul_binary(const Tensor& a,
                            const Tensor& b_binary,
                            float b_scale);

// Per-channel quantization support
Tensor qlinear_matmul_per_channel(const Tensor& a,
                                 const std::vector<float>& a_scales,
                                 const std::vector<int8_t>& a_zero_points,
                                 const Tensor& b,
                                 const std::vector<float>& b_scales,
                                 const std::vector<int8_t>& b_zero_points,
                                 float y_scale, const Tensor& y_zero_point);

// Batch matrix multiplication support
Tensor qlinear_batch_matmul(const std::vector<Tensor>& a_batch,
                           const std::vector<Tensor>& b_batch,
                           float a_scale, float b_scale,
                           float y_scale, const Tensor& y_zero_point);

// Optimized implementations
namespace impl {
    // Naive reference implementation
    Tensor qlinear_matmul_naive(const Tensor& a, float a_scale, int8_t a_zero_point,
                               const Tensor& b, float b_scale, int8_t b_zero_point,
                               float y_scale, int8_t y_zero_point);

    // ARM NEON optimized implementation
    Tensor qlinear_matmul_neon(const Tensor& a, float a_scale, int8_t a_zero_point,
                              const Tensor& b, float b_scale, int8_t b_zero_point,
                              float y_scale, int8_t y_zero_point);

    // Ternary-weight optimized implementation
    Tensor qlinear_matmul_ternary_packed(const Tensor& a,
                                        const TernaryPackedWeights& b_packed,
                                        float b_scale);

    // Binary-weight optimized implementation
    Tensor qlinear_matmul_binary_packed(const Tensor& a,
                                       const BinaryPackedWeights& b_packed,
                                       float b_scale);
}

// Helper function to extract quantization parameters from ONNX inputs
struct MatMulQuantizationConfig {
    std::vector<float> a_scales;
    std::vector<int8_t> a_zero_points;
    std::vector<float> b_scales;
    std::vector<int8_t> b_zero_points;
    float y_scale;
    int8_t y_zero_point;
};

MatMulQuantizationConfig extract_matmul_quantization_config(
    const Tensor& a_scale, const Tensor& a_zero_point,
    const Tensor& b_scale, const Tensor& b_zero_point,
    float y_scale, const Tensor& y_zero_point);

} // namespace tbn