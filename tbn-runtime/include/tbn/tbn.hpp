#pragma once

/**
 * @file tbn.hpp
 * @brief Main header for TBN Runtime
 *
 * This file includes all public APIs for the Ternary-Binary Network runtime.
 */

// Include all the necessary headers
#include <memory>
#include <string>
#include <cstddef>

// Forward declarations
namespace tbn {
    class TBNModel;
    class InferenceSession;
    class Tensor;
    class Logger;
}

// Include core components
#include "runtime/types.hpp"
#include "runtime/tensor.hpp"
#include "runtime/model.hpp"
#include "runtime/session.hpp"

#include "quantization/quantizer.hpp"
#include "quantization/strategies.hpp"

#include "operators/conv2d.hpp"
#include "operators/gemm.hpp"

#include "memory/packed_weights.hpp"

#include "utils/errors.hpp"
#include "utils/logging.hpp"

/**
 * @namespace tbn
 * @brief Ternary-Binary Network Runtime namespace
 */
namespace tbn {

/**
 * @brief Load a TBN model from ONNX file
 *
 * @param path Path to ONNX model file
 * @return TBNModel loaded model
 * @throws TBNError if loading fails
 */
inline TBNModel load_model(const std::string& path) {
    TBN_LOG_INFO("Loading model from: " + path);

    // TODO: Implement actual ONNX parsing
    auto model = TBNModel();
    model.set_producer("tbn-runtime", get_version());

    // Placeholder - would parse ONNX file
    throw NotImplementedError("ONNX model loading not implemented yet");
}

/**
 * @brief Load a TBN model from memory buffer
 *
 * @param data Pointer to ONNX model data
 * @param size Size of model data in bytes
 * @return TBNModel loaded model
 * @throws TBNError if loading fails
 */
inline TBNModel load_model_from_buffer(const void* data, size_t size) {
    TBN_LOG_INFO("Loading model from buffer, size: " + std::to_string(size));

    // TODO: Implement actual ONNX parsing
    auto model = TBNModel();
    model.set_producer("tbn-runtime", get_version());

    // Placeholder - would parse ONNX data
    throw NotImplementedError("ONNX buffer loading not implemented yet");
}

/**
 * @brief Get version information
 *
 * @return Version string
 */
inline const char* get_version() {
    return "0.1.0";
}

/**
 * @brief Create an inference session from a model
 *
 * @param model The TBN model
 * @param options Session configuration options
 * @return Inference session
 */
inline std::unique_ptr<InferenceSession> create_session(
    std::shared_ptr<TBNModel> model,
    const InferenceSession::Options& options = InferenceSession::Options()) {

    return create_inference_session(model, options);
}

} // namespace tbn

/**
 * @example simple_example.cpp
 *
 * This example shows how to load a model and run inference.
 */