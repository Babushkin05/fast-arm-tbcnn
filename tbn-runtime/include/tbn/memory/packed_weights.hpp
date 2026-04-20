#pragma once

#include "../runtime/types.hpp"
#include "../runtime/tensor.hpp"
#include "../utils/errors.hpp"
#include <cstring>
#include <vector>

namespace tbn {

// Bit-packing for ternary weights (-1, 0, +1)
// Pack 4 ternary values into 1 byte (2 bits per value)
class TernaryPackedWeights {
private:
    std::vector<uint8_t> packed_data_;
    Shape original_shape_;
    int64_t num_elements_;

public:
    TernaryPackedWeights() = default;

    TernaryPackedWeights(const Shape& shape)
        : original_shape_(shape), num_elements_(shape.size()) {
        int64_t packed_size = (num_elements_ + 3) / 4; // 4 values per byte
        packed_data_.resize(packed_size, 0);
    }

    // Pack ternary weights
    void pack(const Tensor& ternary_weights);

    // Unpack to ternary tensor
    Tensor unpack() const;

    // Get individual weight value
    TernaryWeight get_weight(int64_t index) const {
        TBN_CHECK(index >= 0 && index < num_elements_, InvalidArgumentError, "Index out of range");

        int64_t byte_index = index / 4;
        int64_t bit_offset = (index % 4) * 2;

        uint8_t value = (packed_data_[byte_index] >> bit_offset) & 0x03;

        // Decode: 00=-1, 01=0, 10=+1, 11=reserved
        switch (value) {
            case 0: return TERNARY_MINUS_ONE;
            case 1: return TERNARY_ZERO;
            case 2: return TERNARY_PLUS_ONE;
            default: return TERNARY_ZERO; // Reserved value
        }
    }

    // Set individual weight value
    void set_weight(int64_t index, TernaryWeight weight) {
        TBN_CHECK(index >= 0 && index < num_elements_, InvalidArgumentError, "Index out of range");

        int64_t byte_index = index / 4;
        int64_t bit_offset = (index % 4) * 2;

        // Encode: -1=00, 0=01, +1=10
        uint8_t encoded;
        switch (weight) {
            case TERNARY_MINUS_ONE: encoded = 0; break;
            case TERNARY_ZERO: encoded = 1; break;
            case TERNARY_PLUS_ONE: encoded = 2; break;
            default: encoded = 1; // Default to zero
        }

        // Clear existing bits and set new value
        packed_data_[byte_index] &= ~(0x03 << bit_offset);
        packed_data_[byte_index] |= (encoded << bit_offset);
    }

    const Shape& shape() const { return original_shape_; }
    int64_t size() const { return num_elements_; }
    int64_t packed_size() const { return packed_data_.size(); }
    const uint8_t* data() const { return packed_data_.data(); }
    uint8_t* data() { return packed_data_.data(); }
};

// Bit-packing for binary weights (0, 1)
// Pack 8 binary values into 1 byte
class BinaryPackedWeights {
private:
    std::vector<uint8_t> packed_data_;
    Shape original_shape_;
    int64_t num_elements_;

public:
    BinaryPackedWeights() = default;

    BinaryPackedWeights(const Shape& shape)
        : original_shape_(shape), num_elements_(shape.size()) {
        int64_t packed_size = (num_elements_ + 7) / 8; // 8 values per byte
        packed_data_.resize(packed_size, 0);
    }

    // Pack binary weights
    void pack(const Tensor& binary_weights);

    // Unpack to binary tensor
    Tensor unpack() const;

    // Get individual weight value
    BinaryWeight get_weight(int64_t index) const {
        TBN_CHECK(index >= 0 && index < num_elements_, InvalidArgumentError, "Index out of range");

        int64_t byte_index = index / 8;
        int64_t bit_offset = index % 8;

        return (packed_data_[byte_index] >> bit_offset) & 0x01;
    }

    // Set individual weight value
    void set_weight(int64_t index, BinaryWeight weight) {
        TBN_CHECK(index >= 0 && index < num_elements_, InvalidArgumentError, "Index out of range");

        int64_t byte_index = index / 8;
        int64_t bit_offset = index % 8;

        if (weight == BINARY_ONE) {
            packed_data_[byte_index] |= (1 << bit_offset);
        } else {
            packed_data_[byte_index] &= ~(1 << bit_offset);
        }
    }

    const Shape& shape() const { return original_shape_; }
    int64_t size() const { return num_elements_; }
    int64_t packed_size() const { return packed_data_.size(); }
    const uint8_t* data() const { return packed_data_.data(); }
    uint8_t* data() { return packed_data_.data(); }
};

// Memory layout optimization for ternary/binary weights
class WeightLayoutOptimizer {
public:
    // Optimize memory layout for SIMD operations
    static TernaryPackedWeights optimize_ternary_layout(const Tensor& weights);
    static BinaryPackedWeights optimize_binary_layout(const Tensor& weights);

    // Align weights for cache efficiency
    static void align_for_cache(TernaryPackedWeights& weights, int64_t cache_line_size = 64);
    static void align_for_cache(BinaryPackedWeights& weights, int64_t cache_line_size = 64);

    // Pre-transpose for common operations
    static TernaryPackedWeights pre_transpose(const TernaryPackedWeights& weights);
    static BinaryPackedWeights pre_transpose(const BinaryPackedWeights& weights);
};

// Weight compression utilities
class WeightCompressor {
public:
    // Compress weights using various algorithms
    enum class Algorithm {
        NONE,
        RUN_LENGTH,
        HUFFMAN,
        LZ4
    };

    static std::vector<uint8_t> compress_ternary(const TernaryPackedWeights& weights, Algorithm algo);
    static std::vector<uint8_t> compress_binary(const BinaryPackedWeights& weights, Algorithm algo);

    static TernaryPackedWeights decompress_ternary(const std::vector<uint8_t>& data, Algorithm algo, const Shape& shape);
    static BinaryPackedWeights decompress_binary(const std::vector<uint8_t>& data, Algorithm algo, const Shape& shape);
};

} // namespace tbn