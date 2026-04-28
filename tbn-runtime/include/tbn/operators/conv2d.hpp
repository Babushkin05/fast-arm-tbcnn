#pragma once

#include "../runtime/tensor.hpp"
#include "../runtime/types.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "../../../../GeMM/05-final/GeMM.hpp"
#include <vector>

namespace tbn {

struct Conv2DParams {
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t groups;

    Conv2DParams()
        : kernel_h(3), kernel_w(3),
          stride_h(1), stride_w(1),
          pad_h(0), pad_w(0),
          dilation_h(1), dilation_w(1),
          groups(1) {}

    Conv2DParams(int64_t kh, int64_t kw, int64_t sh = 1, int64_t sw = 1,
                 int64_t ph = 0, int64_t pw = 0, int64_t dh = 1, int64_t dw = 1, int64_t g = 1)
        : kernel_h(kh), kernel_w(kw),
          stride_h(sh), stride_w(sw),
          pad_h(ph), pad_w(pw),
          dilation_h(dh), dilation_w(dw),
          groups(g) {}
};

// Standard Conv2D implementation
Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor* bias = nullptr,
              const Conv2DParams& params = Conv2DParams());

// Ternary Conv2D - optimized for ternary weights
Tensor conv2d_ternary(const Tensor& input, const Tensor& ternary_weights,
                      const Tensor* bias = nullptr, const Conv2DParams& params = Conv2DParams());

// Binary Conv2D - optimized for binary weights
Tensor conv2d_binary(const Tensor& input, const Tensor& binary_weights,
                     const Tensor* bias = nullptr, const Conv2DParams& params = Conv2DParams());

// Binary Conv2D with per-channel scales - for binary-scaled weights
Tensor conv2d_binary_with_scales(const Tensor& input, const Tensor& binary_weights,
                                  const Tensor* bias, const Conv2DParams& params,
                                  const std::vector<float>& channel_scales);

// Grouped convolution support
Tensor conv2d_grouped(const Tensor& input, const Tensor& weights,
                      const Tensor* bias = nullptr, int64_t groups = 1,
                      const Conv2DParams& params = Conv2DParams());

// Depthwise convolution
Tensor depthwise_conv2d(const Tensor& input, const Tensor& weights,
                        const Tensor* bias = nullptr, const Conv2DParams& params = Conv2DParams());

// Dilated/atrous convolution
Tensor dilated_conv2d(const Tensor& input, const Tensor& weights,
                      const Tensor* bias = nullptr, int64_t dilation = 1,
                      const Conv2DParams& params = Conv2DParams());

// Transposed convolution (deconvolution)
Tensor conv2d_transpose(const Tensor& input, const Tensor& weights,
                        const Tensor* bias = nullptr, const Conv2DParams& params = Conv2DParams());

// Winograd convolution (fast for small kernels)
Tensor conv2d_winograd(const Tensor& input, const Tensor& weights,
                       const Tensor* bias = nullptr, const Conv2DParams& params = Conv2DParams());

// Im2Col helper for convolution
namespace impl {
    Tensor im2col(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
                  int64_t dilation_h, int64_t dilation_w);

    // Fused: im2col + float-to-ternary quantization in single pass
    // Writes directly into TernaryMatrix bit-planes, no intermediate float buffer
    TernaryMatrix im2col_ternary_packed(
        const float* input, int64_t N, int64_t C, int64_t H, int64_t W,
        int64_t kernel_h, int64_t kernel_w,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dilation_h, int64_t dilation_w,
        uint32_t m_padded, uint32_t k_padded,
        float threshold_low, float threshold_high);

    Tensor col2im(const Tensor& col, const Shape& input_shape,
                  int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
                  int64_t dilation_h, int64_t dilation_w);
}

} // namespace tbn