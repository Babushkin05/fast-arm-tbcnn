#pragma once

#include "../runtime/tensor.hpp"
#include "../runtime/types.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"

namespace tbn {

struct Pool2DParams {
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t dilation_h;
    int64_t dilation_w;
    bool ceil_mode;           // Use ceil instead of floor for output size
    bool count_include_pad;   // Include padding in average calculation

    Pool2DParams()
        : kernel_h(2), kernel_w(2),
          stride_h(2), stride_w(2),
          pad_h(0), pad_w(0),
          dilation_h(1), dilation_w(1),
          ceil_mode(false),
          count_include_pad(false) {}

    Pool2DParams(int64_t kh, int64_t kw, int64_t sh = 2, int64_t sw = 2,
                 int64_t ph = 0, int64_t pw = 0)
        : kernel_h(kh), kernel_w(kw),
          stride_h(sh), stride_w(sw),
          pad_h(ph), pad_w(pw),
          dilation_h(1), dilation_w(1),
          ceil_mode(false),
          count_include_pad(false) {}
};

// MaxPool2D - returns maximum value in each window
Tensor maxpool2d(const Tensor& input, const Pool2DParams& params = Pool2DParams());

// AvgPool2D - returns average value in each window
Tensor avgpool2d(const Tensor& input, const Pool2DParams& params = Pool2DParams());

// GlobalMaxPool2D - pool over entire spatial dimensions
Tensor global_maxpool2d(const Tensor& input);

// GlobalAvgPool2D - pool over entire spatial dimensions
Tensor global_avgpool2d(const Tensor& input);

// MaxPool2D with indices (for backward pass)
struct PoolWithIndices {
    Tensor output;
    Tensor indices;  // Indices of max values
};

PoolWithIndices maxpool2d_with_indices(const Tensor& input, const Pool2DParams& params = Pool2DParams());

} // namespace tbn
