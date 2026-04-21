#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <cstring>

namespace tbn {

enum class ErrorCode {
    SUCCESS = 0,
    INVALID_ARGUMENT = -1,
    OUT_OF_MEMORY = -2,
    NOT_IMPLEMENTED = -3,
    INVALID_MODEL = -4,
    RUNTIME_ERROR = -5,
    QUANTIZATION_ERROR = -6,
    INVALID_SHAPE = -7,
    DEVICE_NOT_SUPPORTED = -8,
    ONNX_PARSE_ERROR = -9,
    OPERATOR_NOT_SUPPORTED = -10
};

class TBNError : public std::exception {
private:
    ErrorCode code_;
    std::string message_;

public:
    TBNError(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}

    TBNError(ErrorCode code, const char* message)
        : code_(code), message_(message) {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

    ErrorCode code() const noexcept {
        return code_;
    }
};

class InvalidArgumentError : public TBNError {
public:
    InvalidArgumentError(const std::string& msg)
        : TBNError(ErrorCode::INVALID_ARGUMENT, msg) {}
};

class OutOfMemoryError : public TBNError {
public:
    OutOfMemoryError(const std::string& msg)
        : TBNError(ErrorCode::OUT_OF_MEMORY, msg) {}
};

class NotImplementedError : public TBNError {
public:
    NotImplementedError(const std::string& msg)
        : TBNError(ErrorCode::NOT_IMPLEMENTED, msg) {}
};

class InvalidModelError : public TBNError {
public:
    InvalidModelError(const std::string& msg)
        : TBNError(ErrorCode::INVALID_MODEL, msg) {}
};

class QuantizationError : public TBNError {
public:
    QuantizationError(const std::string& msg)
        : TBNError(ErrorCode::QUANTIZATION_ERROR, msg) {}
};

class InvalidShapeError : public TBNError {
public:
    InvalidShapeError(const std::string& msg)
        : TBNError(ErrorCode::INVALID_SHAPE, msg) {}
};

class RuntimeError : public TBNError {
public:
    RuntimeError(const std::string& msg)
        : TBNError(ErrorCode::RUNTIME_ERROR, msg) {}
};

#define TBN_CHECK(condition, error_type, message) \
    if (!(condition)) { \
        throw error_type(message); \
    }

#define TBN_CHECK_NOT_NULL(ptr, error_type) \
    TBN_CHECK((ptr) != nullptr, error_type, #ptr " must not be null")

} // namespace tbn