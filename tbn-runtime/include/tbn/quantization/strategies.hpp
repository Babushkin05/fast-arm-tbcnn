#pragma once

#include "quantizer.hpp"
#include "../runtime/types.hpp"
#include "../runtime/tensor.hpp"

namespace tbn {

// Per-channel quantization strategy
class PerChannelQuantizer {
public:
    struct ChannelParams {
        float scale;
        int32_t zero_point;
        int64_t channel_axis;
    };

    static std::vector<ChannelParams> compute_channel_params(
        const Tensor& weights,
        int64_t channel_axis = 0
    );

    static Tensor quantize_per_channel(
        const Tensor& weights,
        const std::vector<ChannelParams>& params,
        int64_t channel_axis = 0
    );
};

// Per-tensor quantization strategy
class PerTensorQuantizer {
public:
    static QuantizationParams compute_params(
        const Tensor& tensor,
        int32_t bit_width = 8
    );

    static Tensor quantize(
        const Tensor& tensor,
        const QuantizationParams& params
    );
};

// Ternary quantization with learnable thresholds
class LearnableTernaryQuantizer {
private:
    float threshold_low_;
    float threshold_high_;
    float scale_;

public:
    LearnableTernaryQuantizer(float threshold_low = -0.5f, float threshold_high = 0.5f)
        : threshold_low_(threshold_low), threshold_high_(threshold_high), scale_(1.0f) {}

    void set_thresholds(float low, float high) {
        threshold_low_ = low;
        threshold_high_ = high;
    }

    void set_scale(float scale) {
        scale_ = scale;
    }

    Tensor quantize(const Tensor& input);

    // Learn thresholds from statistics
    void learn_thresholds(const Tensor& weights, float percentile = 0.05f);
};

// Binary quantization strategies
class BinaryQuantizationStrategy {
public:
    enum class Method {
        SIGN,           // Simple sign-based quantization
        THRESHOLD,      // Threshold-based quantization
        CHANNEL_WISE,   // Per-channel binary quantization
        TERNARY_TO_BINARY // Convert ternary to binary
    };

    static Tensor quantize(const Tensor& input, Method method = Method::SIGN);
};

// Mixed precision quantization
class MixedPrecisionQuantizer {
public:
    struct LayerConfig {
        std::string layer_name;
        QuantizationType type;
        QuantizationParams params;
    };

    MixedPrecisionQuantizer(const std::vector<LayerConfig>& configs);

    Tensor quantize_layer(const std::string& layer_name, const Tensor& input);

private:
    std::unordered_map<std::string, std::unique_ptr<Quantizer>> quantizers_;
};

} // namespace tbn