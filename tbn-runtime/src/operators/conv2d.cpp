#include "tbn/operators/conv2d.hpp"
#include "tbn/operators/gemm.hpp"
#include <cstring>
#include <algorithm>

namespace tbn {

// Standard Conv2D - naive implementation with support for stride, padding, dilation
Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor* bias,
              const Conv2DParams& params) {
    // Validate inputs
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "Conv2D input must be 4D (NCHW), got " + shape_to_string(input.shape()));
    TBN_CHECK(weights.shape().dims.size() == 4, InvalidShapeError,
              "Conv2D weights must be 4D (MCHW), got " + shape_to_string(weights.shape()));
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Conv2D requires float32 input");
    TBN_CHECK(weights.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Conv2D requires float32 weights");

    // Input dimensions (NCHW)
    int64_t N = input.shape().dims[0];  // batch
    int64_t C = input.shape().dims[1];  // input channels
    int64_t H = input.shape().dims[2];  // height
    int64_t W = input.shape().dims[3];  // width

    // Weight dimensions (MCHW where M = output channels)
    int64_t M = weights.shape().dims[0];  // output channels
    int64_t C_w = weights.shape().dims[1];  // weight input channels
    int64_t kH = weights.shape().dims[2];  // kernel height
    int64_t kW = weights.shape().dims[3];  // kernel width

    // Use kernel size from weights if not specified in params
    int64_t kernel_h = (params.kernel_h > 0) ? params.kernel_h : kH;
    int64_t kernel_w = (params.kernel_w > 0) ? params.kernel_w : kW;

    // Validate channel consistency
    TBN_CHECK(C == C_w, InvalidShapeError,
              "Input channels (" + std::to_string(C) + ") must match weight channels (" +
              std::to_string(C_w) + ")");

    // Calculate output dimensions
    // out = (in + 2*pad - dilation*(kernel-1) - 1) / stride + 1
    int64_t out_h = (H + 2 * params.pad_h - params.dilation_h * (kernel_h - 1) - 1) /
                     params.stride_h + 1;
    int64_t out_w = (W + 2 * params.pad_w - params.dilation_w * (kernel_w - 1) - 1) /
                     params.stride_w + 1;

    TBN_CHECK(out_h > 0 && out_w > 0, InvalidShapeError,
              "Invalid output dimensions: " + std::to_string(out_h) + "x" + std::to_string(out_w));

    // Allocate output tensor
    Shape output_shape{N, M, out_h, out_w};
    Tensor output(output_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    const float* weight_data = weights.typed_data<float>();
    float* output_data = output.typed_data<float>();

    // Initialize with bias if provided
    if (bias) {
        TBN_CHECK(bias->shape().dims.size() == 1, InvalidShapeError,
                  "Bias must be 1D");
        TBN_CHECK(bias->shape().dims[0] == M, InvalidShapeError,
                  "Bias size must match output channels");
        const float* bias_data = bias->typed_data<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t out_idx = ((n * M + m) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = bias_data[m];
                    }
                }
            }
        }
    } else {
        std::memset(output_data, 0, output.num_elements() * sizeof(float));
    }

    // Naive convolution: 7 nested loops
    // N, M, out_h, out_w, C, kH, kW
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float acc = output_data[((n * M + m) * out_h + oh) * out_w + ow];

                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                // Input position with stride, padding, dilation
                                int64_t ih = oh * params.stride_h - params.pad_h +
                                             kh * params.dilation_h;
                                int64_t iw = ow * params.stride_w - params.pad_w +
                                             kw * params.dilation_w;

                                // Bounds check (implicit zero-padding)
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue;
                                }

                                // Index calculations (NCHW layout)
                                int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                int64_t w_idx = ((m * C + c) * kernel_h + kh) * kernel_w + kw;

                                acc += input_data[in_idx] * weight_data[w_idx];
                            }
                        }
                    }

                    output_data[((n * M + m) * out_h + oh) * out_w + ow] = acc;
                }
            }
        }
    }

    TBN_LOG_DEBUG("Conv2D: " + shape_to_string(input.shape()) + " -> " +
                  shape_to_string(output.shape()));

    return output;
}

// Ternary Conv2D - optimized for ternary weights
Tensor conv2d_ternary(const Tensor& input, const Tensor& ternary_weights,
                      const Tensor* bias, const Conv2DParams& params) {
    // For now, use standard conv2d with unpacked weights
    // TODO: Implement optimized version with packed weights
    TBN_LOG_WARNING("conv2d_ternary: using standard implementation (not optimized)");

    TBN_CHECK(ternary_weights.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Ternary Conv2D requires ternary weights");

    // Convert ternary weights to float for now
    // In production, would use packed weight access
    const auto& tw = ternary_weights;
    Shape float_shape = tw.shape();
    Tensor float_weights(float_shape, DataType::FLOAT32);

    const int8_t* tw_data = tw.typed_data<int8_t>();
    float* fw_data = float_weights.typed_data<float>();

    for (int64_t i = 0; i < tw.num_elements(); ++i) {
        fw_data[i] = static_cast<float>(tw_data[i]);  // -1, 0, +1
    }

    return conv2d(input, float_weights, bias, params);
}

// Binary Conv2D - optimized for binary weights
Tensor conv2d_binary(const Tensor& input, const Tensor& binary_weights,
                     const Tensor* bias, const Conv2DParams& params) {
    // For now, use standard conv2d with unpacked weights
    // TODO: Implement optimized version with packed weights
    TBN_LOG_WARNING("conv2d_binary: using standard implementation (not optimized)");

    TBN_CHECK(binary_weights.dtype() == DataType::BINARY, InvalidArgumentError,
              "Binary Conv2D requires binary weights");

    // Convert binary weights to float for now
    const uint8_t* bw_data = binary_weights.typed_data<uint8_t>();
    Shape float_shape = binary_weights.shape();
    Tensor float_weights(float_shape, DataType::FLOAT32);
    float* fw_data = float_weights.typed_data<float>();

    for (int64_t i = 0; i < binary_weights.num_elements(); ++i) {
        fw_data[i] = (bw_data[i] == 0) ? -1.0f : 1.0f;
    }

    return conv2d(input, float_weights, bias, params);
}

// Grouped convolution
Tensor conv2d_grouped(const Tensor& input, const Tensor& weights,
                      const Tensor* bias, int64_t groups,
                      const Conv2DParams& params) {
    TBN_CHECK(groups > 0, InvalidArgumentError, "Groups must be positive");
    TBN_CHECK(input.shape().dims[1] % groups == 0, InvalidShapeError,
              "Input channels must be divisible by groups");

    if (groups == 1) {
        return conv2d(input, weights, bias, params);
    }

    // TODO: Implement grouped convolution
    throw NotImplementedError("Grouped convolution not yet implemented");
}

// Depthwise convolution
Tensor depthwise_conv2d(const Tensor& input, const Tensor& weights,
                        const Tensor* bias, const Conv2DParams& params) {
    // Depthwise is grouped conv with groups = input_channels
    int64_t groups = input.shape().dims[1];
    return conv2d_grouped(input, weights, bias, groups, params);
}

// Dilated convolution
Tensor dilated_conv2d(const Tensor& input, const Tensor& weights,
                      const Tensor* bias, int64_t dilation,
                      const Conv2DParams& params) {
    Conv2DParams new_params = params;
    new_params.dilation_h = dilation;
    new_params.dilation_w = dilation;
    return conv2d(input, weights, bias, new_params);
}

// Transposed convolution
Tensor conv2d_transpose(const Tensor& input, const Tensor& weights,
                        const Tensor* bias, const Conv2DParams& params) {
    throw NotImplementedError("Transposed convolution not yet implemented");
}

// Winograd convolution
Tensor conv2d_winograd(const Tensor& input, const Tensor& weights,
                       const Tensor* bias, const Conv2DParams& params) {
    throw NotImplementedError("Winograd convolution not yet implemented");
}

// Im2Col helper - converts input to column matrix for GEMM-based convolution
namespace impl {

Tensor im2col(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
              int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
              int64_t dilation_h, int64_t dilation_w) {
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "im2col input must be 4D (NCHW)");

    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Output dimensions
    int64_t out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Column matrix shape: (N * out_h * out_w) x (C * kernel_h * kernel_w)
    int64_t col_rows = N * out_h * out_w;
    int64_t col_cols = C * kernel_h * kernel_w;

    Shape col_shape{col_rows, col_cols};
    Tensor col(col_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    float* col_data = col.typed_data<float>();

    // Fill column matrix
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                int64_t row_idx = (n * out_h + oh) * out_w + ow;

                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
                            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;

                            int64_t col_idx = (c * kernel_h + kh) * kernel_w + kw;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                col_data[row_idx * col_cols + col_idx] = input_data[in_idx];
                            } else {
                                col_data[row_idx * col_cols + col_idx] = 0.0f;  // zero-padding
                            }
                        }
                    }
                }
            }
        }
    }

    return col;
}

Tensor col2im(const Tensor& col, const Shape& input_shape,
              int64_t kernel_h, int64_t kernel_w,
              int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
              int64_t dilation_h, int64_t dilation_w) {
    TBN_CHECK(input_shape.dims.size() == 4, InvalidShapeError,
              "col2im input_shape must be 4D (NCHW)");
    TBN_CHECK(col.shape().dims.size() == 2, InvalidShapeError,
              "col2im col must be 2D");

    int64_t N = input_shape.dims[0];
    int64_t C = input_shape.dims[1];
    int64_t H = input_shape.dims[2];
    int64_t W = input_shape.dims[3];

    Tensor output(input_shape, DataType::FLOAT32);
    std::memset(output.data(), 0, output.num_elements() * sizeof(float));

    float* output_data = output.typed_data<float>();
    const float* col_data = col.typed_data<float>();

    int64_t out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t col_cols = C * kernel_h * kernel_w;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                int64_t row_idx = (n * out_h + oh) * out_w + ow;

                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
                            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t col_idx = (c * kernel_h + kh) * kernel_w + kw;
                                int64_t out_idx = ((n * C + c) * H + ih) * W + iw;
                                output_data[out_idx] += col_data[row_idx * col_cols + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

} // namespace impl

} // namespace tbn
