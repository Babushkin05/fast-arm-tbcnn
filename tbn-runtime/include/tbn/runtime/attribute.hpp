#pragma once

#include "tensor.hpp"
#include <variant>
#include <vector>
#include <string>
#include <unordered_map>

namespace tbn {

// Attribute value can be one of several types
struct AttributeValue {
    using Value = std::variant<
        int64_t,                    // INT
        float,                      // FLOAT
        std::string,                // STRING
        Tensor,                     // TENSOR
        std::vector<int64_t>,       // INTS
        std::vector<float>,         // FLOATS
        std::vector<std::string>    // STRINGS
    >;

    Value value;

    // Convenience constructors
    AttributeValue() : value(0) {}
    AttributeValue(int64_t v) : value(v) {}
    AttributeValue(int v) : value(static_cast<int64_t>(v)) {}
    AttributeValue(float v) : value(v) {}
    AttributeValue(double v) : value(static_cast<float>(v)) {}
    AttributeValue(const std::string& v) : value(v) {}
    AttributeValue(const char* v) : value(std::string(v)) {}
    AttributeValue(const Tensor& v) : value(v) {}
    AttributeValue(const std::vector<int64_t>& v) : value(v) {}
    AttributeValue(const std::vector<float>& v) : value(v) {}
    AttributeValue(const std::vector<std::string>& v) : value(v) {}

    // Type queries
    bool is_int() const { return std::holds_alternative<int64_t>(value); }
    bool is_float() const { return std::holds_alternative<float>(value); }
    bool is_string() const { return std::holds_alternative<std::string>(value); }
    bool is_tensor() const { return std::holds_alternative<Tensor>(value); }
    bool is_ints() const { return std::holds_alternative<std::vector<int64_t>>(value); }
    bool is_floats() const { return std::holds_alternative<std::vector<float>>(value); }
    bool is_strings() const { return std::holds_alternative<std::vector<std::string>>(value); }

    // Getters (throw if wrong type)
    int64_t as_int() const { return std::get<int64_t>(value); }
    float as_float() const { return std::get<float>(value); }
    const std::string& as_string() const { return std::get<std::string>(value); }
    const Tensor& as_tensor() const { return std::get<Tensor>(value); }
    const std::vector<int64_t>& as_ints() const { return std::get<std::vector<int64_t>>(value); }
    const std::vector<float>& as_floats() const { return std::get<std::vector<float>>(value); }
    const std::vector<std::string>& as_strings() const { return std::get<std::vector<std::string>>(value); }

    // Get with default
    int64_t as_int_or(int64_t default_val) const {
        return is_int() ? as_int() : default_val;
    }
    float as_float_or(float default_val) const {
        return is_float() ? as_float() : default_val;
    }
    const std::string& as_string_or(const std::string& default_val) const {
        static const std::string empty;
        return is_string() ? as_string() : default_val;
    }
};

// Node attributes map
using NodeAttributes = std::unordered_map<std::string, AttributeValue>;

} // namespace tbn
