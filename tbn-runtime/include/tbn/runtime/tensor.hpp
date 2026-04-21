#pragma once

#include "types.hpp"
#include "../utils/errors.hpp"
#include <memory>
#include <cstring>
#include <vector>

namespace tbn {

class Tensor {
private:
    Shape shape_;
    DataType dtype_;
    DeviceType device_;
    std::shared_ptr<void> data_;
    size_t data_size_;
    QuantizationParams quantization_params_;

public:
    Tensor() = default;

    Tensor(const Shape& shape, DataType dtype, DeviceType device = DeviceType::CPU)
        : shape_(shape), dtype_(dtype), device_(device) {
        data_size_ = calculate_data_size();
        data_ = std::shared_ptr<void>(new uint8_t[data_size_], std::default_delete<uint8_t[]>());
        std::memset(data_.get(), 0, data_size_);
    }

    Tensor(const Shape& shape, DataType dtype, const void* data, DeviceType device = DeviceType::CPU)
        : shape_(shape), dtype_(dtype), device_(device) {
        data_size_ = calculate_data_size();
        data_ = std::shared_ptr<void>(new uint8_t[data_size_], std::default_delete<uint8_t[]>());
        std::memcpy(data_.get(), data, data_size_);
    }

    // Getters
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    size_t data_size() const { return data_size_; }

    const QuantizationParams& quantization_params() const { return quantization_params_; }
    void set_quantization_params(const QuantizationParams& params) { quantization_params_ = params; }

    // Shape operations
    int64_t num_elements() const { return shape_.size(); }
    int64_t dim(int idx) const {
        TBN_CHECK(idx >= 0 && idx < shape_.dims.size(), InvalidArgumentError, "Dimension index out of range");
        return shape_.dims[idx];
    }

    // Type-safe data access
    template<typename T>
    T* typed_data() {
        return static_cast<T*>(data_.get());
    }

    template<typename T>
    const T* typed_data() const {
        return static_cast<const T*>(data_.get());
    }

    // Copy operations
    Tensor copy() const {
        Tensor result(shape_, dtype_, device_);
        result.quantization_params_ = quantization_params_;
        std::memcpy(result.data_.get(), data_.get(), data_size_);
        return result;
    }

    Tensor to_device(DeviceType device) const {
        if (device == device_) {
            return copy();
        }
        // TODO: Implement device transfer
        throw NotImplementedError("Device transfer not implemented");
    }

private:
    size_t calculate_data_size() const {
        size_t element_size = 0;
        switch (dtype_) {
            case DataType::FLOAT32: element_size = 4; break;
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::TERNARY:
            case DataType::BINARY: element_size = 1; break;
            case DataType::INT16:
            case DataType::UINT16: element_size = 2; break;
            case DataType::INT32:
            case DataType::UINT32: element_size = 4; break;
            default:
                throw InvalidArgumentError("Unknown data type");
        }
        return shape_.size() * element_size;
    }
};

// Tensor creation helpers
inline Tensor make_tensor(const Shape& shape, const std::vector<float>& data) {
    TBN_CHECK(data.size() == shape.size(), InvalidShapeError, "Data size doesn't match shape");
    return Tensor(shape, DataType::FLOAT32, data.data());
}

inline Tensor make_tensor(const Shape& shape, const std::vector<TernaryWeight>& data) {
    TBN_CHECK(data.size() == shape.size(), InvalidShapeError, "Data size doesn't match shape");
    return Tensor(shape, DataType::TERNARY, data.data());
}

inline Tensor make_tensor(const Shape& shape, const std::vector<BinaryWeight>& data) {
    TBN_CHECK(data.size() == shape.size(), InvalidShapeError, "Data size doesn't match shape");
    return Tensor(shape, DataType::BINARY, data.data());
}

} // namespace tbn

// Hash function for Shape
namespace std {
    template<>
    struct hash<tbn::Shape> {
        size_t operator()(const tbn::Shape& shape) const {
            size_t hash = 0;
            for (int64_t dim : shape.dims) {
                hash ^= std::hash<int64_t>()(dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
} // namespace std