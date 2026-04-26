#include "tbn/operators/conv2d.hpp"
#include "tbn/operators/gemm.hpp"
#include "tbn/operators/quantized_gemm.hpp"
#include <cstring>
#include <algorithm>

namespace tbn {

namespace {

// Reshape 4D weights [M, C, kH, kW] to 2D [M, C*kH*kW] (for float weights)
Tensor reshape_weights_to_2d(const Tensor& weights) {
    int64_t M = weights.shape().dims[0];
    int64_t C = weights.shape().dims[1];
    int64_t kH = weights.shape().dims[2];
    int64_t kW = weights.shape().dims[3];
    int64_t K = C * kH * kW;

    Shape shape_2d{M, K};
    Tensor reshaped(shape_2d, weights.dtype());

    // Just copy data - it's already in row-major order
    std::memcpy(reshaped.data(), weights.data(),
                weights.num_elements() * sizeof(float));

    return reshaped;
}

// Reshape 4D binary weights [M, C, kH, kW] to 2D [M, C*kH*kW] (for binary weights)
Tensor reshape_binary_weights_to_2d(const Tensor& weights) {
    int64_t M = weights.shape().dims[0];
    int64_t C = weights.shape().dims[1];
    int64_t kH = weights.shape().dims[2];
    int64_t kW = weights.shape().dims[3];
    int64_t K = C * kH * kW;

    Shape shape_2d{M, K};
    Tensor reshaped(shape_2d, DataType::BINARY);

    // Just copy data - it's already in row-major order
    std::memcpy(reshaped.data(), weights.data(),
                weights.num_elements() * sizeof(BinaryWeight));

    return reshaped;
}

// Reshape output from [N*out_h*out_w, M] to [N, M, out_h, out_w]
// NOTE: This requires actual data reordering, not just memcpy!
Tensor reshape_output_to_4d(const Tensor& output_2d, int64_t N, int64_t M,
                            int64_t out_h, int64_t out_w) {
    Shape shape_4d{N, M, out_h, out_w};
    Tensor output(shape_4d, DataType::FLOAT32);

    // Transform from [N*out_h*out_w, M] to [N, M, out_h, out_w]
    // 2D layout: output_2d[n*out_h*out_w + oh*out_w + ow, m]
    // 4D layout: output[n, m, oh, ow]
    const float* src = output_2d.typed_data<float>();
    float* dst = output.typed_data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    int64_t src_idx = (n * out_h * out_w + oh * out_w + ow) * M + m;
                    int64_t dst_idx = ((n * M + m) * out_h + oh) * out_w + ow;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }

    return output;
}

} // anonymous namespace

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
    TBN_LOG_INFO("conv2d_ternary: converting ternary to binary for optimized path");

    TBN_CHECK(ternary_weights.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Ternary Conv2D requires ternary weights");

    // Convert ternary to binary: {-1, 0, +1} -> {-1, +1}
    // Zero values are mapped to +1 (or -1, doesn't matter much for inference)
    Shape binary_shape = ternary_weights.shape();
    Tensor binary_weights(binary_shape, DataType::BINARY);

    const int8_t* tw_data = ternary_weights.typed_data<int8_t>();
    BinaryWeight* bw_data = binary_weights.typed_data<BinaryWeight>();

    for (int64_t i = 0; i < ternary_weights.num_elements(); ++i) {
        // Map: -1 -> BINARY_ZERO (0), 0 or +1 -> BINARY_ONE (1)
        // This preserves the sign: negative weights stay negative in binary
        bw_data[i] = (tw_data[i] < 0) ? BINARY_ZERO : BINARY_ONE;
    }

    return conv2d_binary(input, binary_weights, bias, params);
}

// Binary Conv2D - optimized for binary weights
Tensor conv2d_binary(const Tensor& input, const Tensor& binary_weights,
                     const Tensor* bias, const Conv2DParams& params) {
    // Validate inputs
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "Conv2D input must be 4D (NCHW)");
    TBN_CHECK(binary_weights.shape().dims.size() == 4, InvalidShapeError,
              "Conv2D weights must be 4D (MCHW)");
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Conv2D requires float32 input");
    TBN_CHECK(binary_weights.dtype() == DataType::BINARY, InvalidArgumentError,
              "Binary Conv2D requires binary weights");

    // Input dimensions (NCHW)
    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Weight dimensions
    int64_t M = binary_weights.shape().dims[0];
    int64_t C_w = binary_weights.shape().dims[1];
    int64_t kH = binary_weights.shape().dims[2];
    int64_t kW = binary_weights.shape().dims[3];

    // Validate
    TBN_CHECK(C == C_w, InvalidShapeError,
              "Input channels must match weight channels");

    // Calculate output dimensions
    int64_t out_h = (H + 2 * params.pad_h - params.dilation_h * (kH - 1) - 1) /
                     params.stride_h + 1;
    int64_t out_w = (W + 2 * params.pad_w - params.dilation_w * (kW - 1) - 1) /
                     params.stride_w + 1;

    // Step 1: im2col - transform input to column matrix
    // [N, C, H, W] -> [N*out_h*out_w, C*kH*kW]
    Tensor col = impl::im2col(input, kH, kW,
                              params.stride_h, params.stride_w,
                              params.pad_h, params.pad_w,
                              params.dilation_h, params.dilation_w);

    // Step 2: Reshape weights from [M, C, kH, kW] to [M, C*kH*kW]
    Tensor weights_2d = reshape_binary_weights_to_2d(binary_weights);

    // Step 3: Optimized matrix multiplication using GeMM
    // col: [N*out_h*out_w, C*kH*kW] (float)
    // weights: [M, C*kH*kW] (binary)
    // Need to transpose weights for multiplication: col @ weights^T
    // But qlinear_matmul_binary expects A[M,K] @ B[K,N]
    // So we need: col[N*out, K] @ weights[K, M] = output[N*out, M]

    // Transpose weights: [M, K] -> [K, M]
    int64_t K = C * kH * kW;
    Shape weights_T_shape{K, M};
    Tensor weights_T(weights_T_shape, DataType::BINARY);

    const BinaryWeight* w_data = binary_weights.typed_data<BinaryWeight>();
    BinaryWeight* wT_data = weights_T.typed_data<BinaryWeight>();

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            wT_data[k * M + m] = w_data[m * K + k];
        }
    }

    // Now: col [N*out, K] @ weights_T [K, M] = output [N*out, M]
    Tensor result_2d = qlinear_matmul_binary(col, weights_T, 1.0f);

    // Step 4: Reshape output to [N, M, out_h, out_w]
    Tensor output = reshape_output_to_4d(result_2d, N, M, out_h, out_w);

    // Step 5: Add bias
    if (bias) {
        TBN_CHECK(bias->shape().dims.size() == 1, InvalidShapeError,
                  "Bias must be 1D");
        TBN_CHECK(bias->shape().dims[0] == M, InvalidShapeError,
                  "Bias size must match output channels");
        const float* bias_data = bias->typed_data<float>();
        float* output_data = output.typed_data<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t idx = ((n * M + m) * out_h + oh) * out_w + ow;
                        output_data[idx] += bias_data[m];
                    }
                }
            }
        }
    }

    TBN_LOG_DEBUG("Conv2D binary (optimized): " + shape_to_string(input.shape()) +
                 " -> " + shape_to_string(output.shape()));

    return output;
}

// Binary Conv2D with per-channel scales - for binary-scaled weights like ±5.85
Tensor conv2d_binary_with_scales(const Tensor& input, const Tensor& binary_weights,
                                  const Tensor* bias, const Conv2DParams& params,
                                  const std::vector<float>& channel_scales) {
    // First run standard binary convolution
    Tensor result = conv2d_binary(input, binary_weights, nullptr, params);

    // Apply per-channel scales
    int64_t N = result.shape().dims[0];
    int64_t M = result.shape().dims[1];  // output channels
    int64_t out_h = result.shape().dims[2];
    int64_t out_w = result.shape().dims[3];

    TBN_CHECK(static_cast<int64_t>(channel_scales.size()) == M, InvalidArgumentError,
              "channel_scales size must match output channels");

    float* result_data = result.typed_data<float>();

    // Apply scale per output channel
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            float scale = channel_scales[m];
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    int64_t idx = ((n * M + m) * out_h + oh) * out_w + ow;
                    result_data[idx] *= scale;
                }
            }
        }
    }

    // Add bias after scaling
    if (bias) {
        TBN_CHECK(bias->shape().dims.size() == 1, InvalidShapeError, "Bias must be 1D");
        TBN_CHECK(bias->shape().dims[0] == M, InvalidShapeError,
                  "Bias size must match output channels");
        const float* bias_data = bias->typed_data<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t idx = ((n * M + m) * out_h + oh) * out_w + ow;
                        result_data[idx] += bias_data[m];
                    }
                }
            }
        }
    }

    return result;
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
