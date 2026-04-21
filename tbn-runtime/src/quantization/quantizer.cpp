#include "../../include/tbn/quantization/quantizer.hpp"
#include "../../include/tbn/utils/logging.hpp"
#include <cmath>
#include <vector>

namespace tbn {

// Ternary Quantizer Implementation
Tensor TernaryQuantizer::quantize(const Tensor& input) {
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Ternary quantization requires float32 input");

    const float* data = input.typed_data<float>();
    std::vector<TernaryWeight> quantized_data(input.num_elements());

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        float value = data[i];
        if (value < params_.threshold_low) {
            quantized_data[i] = TERNARY_MINUS_ONE;
        } else if (value > params_.threshold_high) {
            quantized_data[i] = TERNARY_PLUS_ONE;
        } else {
            quantized_data[i] = TERNARY_ZERO;
        }
    }

    return Tensor(input.shape(), DataType::TERNARY, quantized_data.data());
}

Tensor TernaryQuantizer::dequantize(const Tensor& input) {
    TBN_CHECK(input.dtype() == DataType::TERNARY, InvalidArgumentError,
              "Dequantization requires ternary input");

    const TernaryWeight* data = input.typed_data<TernaryWeight>();
    std::vector<float> dequantized_data(input.num_elements());

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        dequantized_data[i] = dequantize_ternary(data[i]);
    }

    return Tensor(input.shape(), DataType::FLOAT32, dequantized_data.data());
}

// Binary Quantizer Implementation
Tensor BinaryQuantizer::quantize(const Tensor& input) {
    TBN_CHECK(input.dtype() == DataType::FLOAT32, InvalidArgumentError,
              "Binary quantization requires float32 input");

    const float* data = input.typed_data<float>();
    std::vector<BinaryWeight> quantized_data(input.num_elements());

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        float value = data[i];
        if (value < params_.threshold) {
            quantized_data[i] = BINARY_ZERO;
        } else {
            quantized_data[i] = BINARY_ONE;
        }
    }

    return Tensor(input.shape(), DataType::BINARY, quantized_data.data());
}

Tensor BinaryQuantizer::dequantize(const Tensor& input) {
    TBN_CHECK(input.dtype() == DataType::BINARY, InvalidArgumentError,
              "Dequantization requires binary input");

    const BinaryWeight* data = input.typed_data<BinaryWeight>();
    std::vector<float> dequantized_data(input.num_elements());

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        dequantized_data[i] = dequantize_binary(data[i]);
    }

    return Tensor(input.shape(), DataType::FLOAT32, dequantized_data.data());
}

} // namespace tbn