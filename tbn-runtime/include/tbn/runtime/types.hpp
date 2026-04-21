#pragma once

#include <stddef.h>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace tbn {

enum class DataType {
    FLOAT32,
    INT8,
    INT16,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    TERNARY,
    BINARY
};

enum class DeviceType {
    CPU,
    ARM_NEON,
    ARM_SVE
};

struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> dims_) : dims(dims_) {}
    Shape(const std::vector<int64_t>& dims_) : dims(dims_) {}

    int64_t size() const {
        int64_t s = 1;
        for (auto d : dims) s *= d;
        return s;
    }

    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }
};

struct QuantizationParams {
    float scale;
    int32_t zero_point;
    int32_t bit_width;

    QuantizationParams() : scale(1.0f), zero_point(0), bit_width(8) {}
    QuantizationParams(float s, int32_t z, int32_t bits = 8)
        : scale(s), zero_point(z), bit_width(bits) {}
};

struct TernaryQuantizationParams : QuantizationParams {
    float threshold_low;
    float threshold_high;

    TernaryQuantizationParams()
        : QuantizationParams(1.0f, 0, 2),
          threshold_low(-0.5f), threshold_high(0.5f) {}
};

struct BinaryQuantizationParams : QuantizationParams {
    float threshold;

    BinaryQuantizationParams()
        : QuantizationParams(1.0f, 0, 1),
          threshold(0.0f) {}
};

using TernaryWeight = int8_t;
using BinaryWeight = uint8_t;

constexpr TernaryWeight TERNARY_MINUS_ONE = -1;
constexpr TernaryWeight TERNARY_ZERO = 0;
constexpr TernaryWeight TERNARY_PLUS_ONE = 1;

constexpr BinaryWeight BINARY_ZERO = 0;
constexpr BinaryWeight BINARY_ONE = 1;

inline int8_t quantize_ternary(float value, float threshold_low = -0.5f, float threshold_high = 0.5f) {
    if (value < threshold_low) return TERNARY_MINUS_ONE;
    if (value > threshold_high) return TERNARY_PLUS_ONE;
    return TERNARY_ZERO;
}

inline uint8_t quantize_binary(float value, float threshold = 0.0f) {
    return value > threshold ? BINARY_ONE : BINARY_ZERO;
}

inline float dequantize_ternary(TernaryWeight weight, float scale = 1.0f) {
    return weight * scale;
}

inline float dequantize_binary(BinaryWeight weight, float scale = 1.0f) {
    return weight * scale;
}

} // namespace tbn