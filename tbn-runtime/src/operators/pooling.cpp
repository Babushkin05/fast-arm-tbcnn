#include "tbn/operators/pooling.hpp"
#include "tbn/operators/gemm.hpp"
#include <cstring>
#include <algorithm>
#include <limits>

namespace tbn {

// MaxPool2D - returns maximum value in each window
Tensor maxpool2d(const Tensor& input, const Pool2DParams& params) {
    // Validate input
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "MaxPool2D input must be 4D (NCHW), got " + shape_to_string(input.shape()));
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "MaxPool2D requires float32 input");

    // Input dimensions (NCHW)
    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Calculate output dimensions
    // Standard: out = floor((in + 2*pad - kernel) / stride) + 1
    // With ceil_mode: out = ceil((in + 2*pad - kernel) / stride) + 1
    int64_t out_h, out_w;

    if (params.ceil_mode) {
        out_h = (H + 2 * params.pad_h - params.kernel_h + params.stride_h - 1) / params.stride_h + 1;
        out_w = (W + 2 * params.pad_w - params.kernel_w + params.stride_w - 1) / params.stride_w + 1;
    } else {
        out_h = (H + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
        out_w = (W + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;
    }

    // Allocate output tensor
    Shape output_shape{N, C, out_h, out_w};
    Tensor output(output_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    float* output_data = output.typed_data<float>();

    // Initialize output to very negative value
    for (int64_t i = 0; i < output.num_elements(); ++i) {
        output_data[i] = -std::numeric_limits<float>::infinity();
    }

    // Max pooling
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int64_t kh = 0; kh < params.kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < params.kernel_w; ++kw) {
                            // Input position
                            int64_t ih = oh * params.stride_h - params.pad_h + kh * params.dilation_h;
                            int64_t iw = ow * params.stride_w - params.pad_w + kw * params.dilation_w;

                            // Bounds check
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                max_val = std::max(max_val, input_data[in_idx]);
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
                    output_data[out_idx] = max_val;
                }
            }
        }
    }

    TBN_LOG_DEBUG("MaxPool2D: " + shape_to_string(input.shape()) + " -> " +
                  shape_to_string(output.shape()));

    return output;
}

// AvgPool2D - returns average value in each window
Tensor avgpool2d(const Tensor& input, const Pool2DParams& params) {
    // Validate input
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "AvgPool2D input must be 4D (NCHW), got " + shape_to_string(input.shape()));
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "AvgPool2D requires float32 input");

    // Input dimensions (NCHW)
    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Calculate output dimensions
    int64_t out_h, out_w;
    if (params.ceil_mode) {
        out_h = (H + 2 * params.pad_h - params.kernel_h + params.stride_h - 1) / params.stride_h + 1;
        out_w = (W + 2 * params.pad_w - params.kernel_w + params.stride_w - 1) / params.stride_w + 1;
    } else {
        out_h = (H + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
        out_w = (W + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;
    }

    // Allocate output tensor
    Shape output_shape{N, C, out_h, out_w};
    Tensor output(output_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    float* output_data = output.typed_data<float>();

    // Average pooling
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int64_t kh = 0; kh < params.kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < params.kernel_w; ++kw) {
                            int64_t ih = oh * params.stride_h - params.pad_h + kh * params.dilation_h;
                            int64_t iw = ow * params.stride_w - params.pad_w + kw * params.dilation_w;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                sum += input_data[in_idx];
                                count++;
                            } else if (params.count_include_pad) {
                                // Count padding as zeros
                                count++;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
                    output_data[out_idx] = (count > 0) ? (sum / count) : 0.0f;
                }
            }
        }
    }

    TBN_LOG_DEBUG("AvgPool2D: " + shape_to_string(input.shape()) + " -> " +
                  shape_to_string(output.shape()));

    return output;
}

// GlobalMaxPool2D - pool over entire spatial dimensions
Tensor global_maxpool2d(const Tensor& input) {
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "GlobalMaxPool2D input must be 4D (NCHW)");

    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Output shape: [N, C, 1, 1]
    Shape output_shape{N, C, 1, 1};
    Tensor output(output_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    float* output_data = output.typed_data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    max_val = std::max(max_val, input_data[idx]);
                }
            }
            output_data[n * C + c] = max_val;
        }
    }

    return output;
}

// GlobalAvgPool2D - pool over entire spatial dimensions
Tensor global_avgpool2d(const Tensor& input) {
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "GlobalAvgPool2D input must be 4D (NCHW)");

    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    // Output shape: [N, C, 1, 1]
    Shape output_shape{N, C, 1, 1};
    Tensor output(output_shape, DataType::FLOAT32);

    const float* input_data = input.typed_data<float>();
    float* output_data = output.typed_data<float>();

    int64_t spatial_size = H * W;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    sum += input_data[idx];
                }
            }
            output_data[n * C + c] = sum / spatial_size;
        }
    }

    return output;
}

// MaxPool2D with indices
PoolWithIndices maxpool2d_with_indices(const Tensor& input, const Pool2DParams& params) {
    TBN_CHECK(input.shape().dims.size() == 4, InvalidShapeError,
              "MaxPool2D input must be 4D (NCHW)");
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "MaxPool2D requires float32 input");

    int64_t N = input.shape().dims[0];
    int64_t C = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    int64_t out_h = (H + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
    int64_t out_w = (W + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;

    Shape output_shape{N, C, out_h, out_w};
    Tensor output(output_shape, DataType::FLOAT32);
    Tensor indices(output_shape, DataType::INT64);

    const float* input_data = input.typed_data<float>();
    float* output_data = output.typed_data<float>();
    int64_t* indices_data = indices.typed_data<int64_t>();

    for (int64_t i = 0; i < output.num_elements(); ++i) {
        output_data[i] = -std::numeric_limits<float>::infinity();
    }

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t kh = 0; kh < params.kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < params.kernel_w; ++kw) {
                            int64_t ih = oh * params.stride_h - params.pad_h + kh;
                            int64_t iw = ow * params.stride_w - params.pad_w + kw;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                if (input_data[in_idx] > max_val) {
                                    max_val = input_data[in_idx];
                                    max_idx = in_idx;
                                }
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
                    output_data[out_idx] = max_val;
                    indices_data[out_idx] = max_idx;
                }
            }
        }
    }

    return {output, indices};
}

} // namespace tbn
