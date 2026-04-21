#include "../../include/tbn/runtime/model.hpp"
#include "../../include/tbn/utils/logging.hpp"
#include "../../include/tbn/utils/errors.hpp"
#include "../../include/tbn/onnx_integration/onnx_parser.hpp"
#include <string>

namespace tbn {

// Implementation of model loading functions
TBNModel load_model(const std::string& path) {
    TBN_LOG_INFO("Loading model from: " + path);

    // For now, use the ONNX integration
    return load_onnx_model(path);
}

TBNModel load_model_from_buffer(const void* data, size_t size) {
    TBN_LOG_INFO("Loading model from buffer, size: " + std::to_string(size));

    // For now, use the ONNX integration
    return load_onnx_model_from_buffer(data, size);
}

const char* get_version() {
    return "0.1.0";
}

} // namespace tbn