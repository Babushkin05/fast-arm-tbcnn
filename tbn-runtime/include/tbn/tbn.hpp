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

#include "onnx_integration/onnx_parser.hpp"

/**
 * @namespace tbn
 * @brief Ternary-Binary Network Runtime namespace
 */
namespace tbn {

// Forward declare the main API functions - implementations are in model.hpp
TBNModel load_model(const std::string& path);
TBNModel load_model_from_buffer(const void* data, size_t size);
const char* get_version();

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