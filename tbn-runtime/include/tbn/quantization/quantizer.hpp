#pragma once

#include "../runtime/types.hpp"
#include "../runtime/tensor.hpp"
#include "../utils/errors.hpp"

namespace tbn {

enum class QuantizationType {
    SYMMETRIC,
    ASYMMETRIC,
    TERNARY,
    BINARY
};

class Quantizer {
public:
    virtual ~Quantizer() = default;

    virtual Tensor quantize(const Tensor& input) = 0;
    virtual Tensor dequantize(const Tensor& input) = 0;

    virtual QuantizationType type() const = 0;
    virtual const QuantizationParams& params() const = 0;
};

class SymmetricQuantizer : public Quantizer {
private:
    QuantizationParams params_;

public:
    SymmetricQuantizer(float scale, int32_t zero_point = 0)
        : params_(scale, zero_point, 8) {}

    Tensor quantize(const Tensor& input) override;
    Tensor dequantize(const Tensor& input) override;

    QuantizationType type() const override { return QuantizationType::SYMMETRIC; }
    const QuantizationParams& params() const override { return params_; }
};

class TernaryQuantizer : public Quantizer {
private:
    TernaryQuantizationParams params_;

public:
    TernaryQuantizer(float threshold_low = -0.5f, float threshold_high = 0.5f)
        : params_() {
        params_.threshold_low = threshold_low;
        params_.threshold_high = threshold_high;
    }

    Tensor quantize(const Tensor& input) override;
    Tensor dequantize(const Tensor& input) override;

    QuantizationType type() const override { return QuantizationType::TERNARY; }
    const QuantizationParams& params() const override { return params_; }
};

class BinaryQuantizer : public Quantizer {
private:
    BinaryQuantizationParams params_;

public:
    BinaryQuantizer(float threshold = 0.0f)
        : params_() {
        params_.threshold = threshold;
    }

    Tensor quantize(const Tensor& input) override;
    Tensor dequantize(const Tensor& input) override;

    QuantizationType type() const override { return QuantizationType::BINARY; }
    const QuantizationParams& params() const override { return params_; }
};

// Factory functions
std::unique_ptr<Quantizer> create_quantizer(QuantizationType type, const QuantizationParams& params);

} // namespace tbn