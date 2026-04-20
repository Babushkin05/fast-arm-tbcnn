#pragma once

#include "../runtime/model.hpp"
#include <memory>
#include <string>
#include <vector>
#include <string>
#include <string>

// Forward declare ONNX types to avoid including in header
namespace onnx {
    class ModelProto;
    class GraphProto;
    class NodeProto;
    class TensorProto;
    class ValueInfoProto;
}

namespace tbn {

class OnnxParserImpl;

class OnnxParser {
private:
    std::unique_ptr<OnnxParserImpl> impl_;

public:
    OnnxParser();
    ~OnnxParser();

    // Parse from file
    void parse_from_file(const std::string& path);

    // Parse from memory buffer
    void parse_from_buffer(const void* data, size_t size);

    // Get the parsed graph
    std::shared_ptr<ModelGraph> get_graph();

    // Get model metadata
    std::string get_producer_name() const;
    std::string get_producer_version() const;
};

// Convenience function to load ONNX model
inline TBNModel load_onnx_model(const std::string& path) {
    OnnxParser parser;
    parser.parse_from_file(path);

    auto graph = parser.get_graph();
    TBNModel model(graph);
    model.set_producer(parser.get_producer_name(), parser.get_producer_version());

    return model;
}

inline TBNModel load_onnx_model_from_buffer(const void* data, size_t size) {
    OnnxParser parser;
    parser.parse_from_buffer(data, size);

    auto graph = parser.get_graph();
    TBNModel model(graph);
    model.set_producer(parser.get_producer_name(), parser.get_producer_version());

    return model;
}

} // namespace tbn