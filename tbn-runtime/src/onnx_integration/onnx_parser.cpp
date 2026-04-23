#include "tbn/onnx_integration/onnx_parser.hpp"
#include "tbn/utils/errors.hpp"
#include "tbn/utils/logging.hpp"
#include "tbn/runtime/types.hpp"
#include "tbn/runtime/tensor.hpp"
#include "tbn/runtime/model.hpp"
#include "tbn/runtime/attribute.hpp"

// Standard headers
#include <cstdint>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <sstream>

#ifdef TBN_ONNX_RUNTIME_ENABLED
// Include ONNX headers only when ONNX support is enabled
#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>
#else
// Dummy types for compilation without ONNX
namespace onnx {
    class TensorProto;
    class ValueInfoProto;
    class ModelProto {
    public:
        bool ParseFromIstream(std::istream*) { return false; }
        bool ParseFromArray(const void*, int) { return false; }
        std::string producer_name() const { return ""; }
        std::string producer_version() const { return ""; }
        int64_t model_version() const { return 0; }
        struct Graph {
            struct Node {
                int node_size() const { return 0; }
            };
            int node_size() const { return 0; }
            int input_size() const { return 0; }
            int output_size() const { return 0; }
            int initializer_size() const { return 0; }
            int value_info_size() const { return 0; }
        };
        const Graph& graph() const { static Graph g; return g; }
    };
}
#endif

// Include standard headers
#include <cstdint>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iostream>

namespace tbn {

class OnnxParserImpl {
private:
    std::unique_ptr<onnx::ModelProto> model_proto_;
    std::unordered_map<std::string, const onnx::TensorProto*> initializers_;
    std::unordered_map<std::string, const onnx::ValueInfoProto*> value_info_;

public:
    OnnxParserImpl() : model_proto_(std::make_unique<onnx::ModelProto>()) {}

    void parse_from_file(const std::string& path) {
        TBN_LOG_INFO("Loading ONNX model from: " + path);

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw InvalidModelError("Failed to open ONNX file: " + path);
        }

        if (!model_proto_->ParseFromIstream(&input)) {
            throw InvalidModelError("Failed to parse ONNX model from: " + path);
        }

        TBN_LOG_INFO("Successfully loaded ONNX model");
        TBN_LOG_INFO("Model version: " + std::to_string(model_proto_->model_version()));
        TBN_LOG_INFO("Producer: " + model_proto_->producer_name() + " " + model_proto_->producer_version());

        extract_metadata();
    }

    void parse_from_buffer(const void* data, size_t size) {
        TBN_LOG_INFO("Parsing ONNX model from buffer, size: " + std::to_string(size));

        if (!model_proto_->ParseFromArray(data, static_cast<int>(size))) {
            throw InvalidModelError("Failed to parse ONNX model from buffer");
        }

        TBN_LOG_INFO("Successfully parsed ONNX model from buffer");
        extract_metadata();
    }

    std::shared_ptr<ModelGraph> convert_to_internal_graph() {
        auto graph = std::make_shared<ModelGraph>();
        const auto& onnx_graph = model_proto_->graph();

        TBN_LOG_INFO("Converting ONNX graph with " + std::to_string(onnx_graph.node_size()) + " nodes");

        // Process inputs
        for (const auto& input : onnx_graph.input()) {
            const std::string& name = input.name();
            graph->inputs.push_back(name);

            Shape shape = extract_shape(input);
            graph->value_info[name] = shape;
            value_info_[name] = &input;

            TBN_LOG_DEBUG("Input: " + name + " shape=" + shape_to_string(shape));
        }

        // Process outputs
        for (const auto& output : onnx_graph.output()) {
            const std::string& name = output.name();
            graph->outputs.push_back(name);

            Shape shape = extract_shape(output);
            graph->value_info[name] = shape;
            value_info_[name] = &output;

            TBN_LOG_DEBUG("Output: " + name + " shape=" + shape_to_string(shape));
        }

        // Process value_info (intermediate tensors)
        for (const auto& value_info : onnx_graph.value_info()) {
            const std::string& name = value_info.name();
            Shape shape = extract_shape(value_info);
            graph->value_info[name] = shape;
            value_info_[name] = &value_info;
        }

        // Process initializers (constant weights)
        for (const auto& init : onnx_graph.initializer()) {
            const std::string& name = init.name();
            Tensor tensor = convert_tensor(init);
            graph->initializers[name] = tensor;
            initializers_[name] = &init;

            TBN_LOG_DEBUG("Initializer: " + name + " shape=" + shape_to_string(tensor.shape()));
        }

        // Process nodes
        for (const auto& onnx_node : onnx_graph.node()) {
            ModelNode node = convert_node(onnx_node);
            graph->nodes.push_back(node);

            TBN_LOG_DEBUG("Node: " + node.name + " type=" + node.op_type);
        }

        TBN_LOG_INFO("Graph conversion complete");
        return graph;
    }

    const std::string& get_producer_name() const {
        return model_proto_->producer_name();
    }

    const std::string& get_producer_version() const {
        return model_proto_->producer_version();
    }

private:
    void extract_metadata() {
        const auto& onnx_graph = model_proto_->graph();

        // Build maps for quick lookup
        for (const auto& init : onnx_graph.initializer()) {
            initializers_[init.name()] = &init;
        }

        for (const auto& vi : onnx_graph.value_info()) {
            value_info_[vi.name()] = &vi;
        }

        // Add inputs/outputs to value_info
        for (const auto& input : onnx_graph.input()) {
            value_info_[input.name()] = &input;
        }

        for (const auto& output : onnx_graph.output()) {
            value_info_[output.name()] = &output;
        }
    }

    Shape extract_shape(const onnx::ValueInfoProto& value_info) {
        Shape shape;
        const auto& tensor_type = value_info.type().tensor_type();

        for (int i = 0; i < tensor_type.shape().dim_size(); ++i) {
            const auto& dim = tensor_type.shape().dim(i);
            if (dim.has_dim_value()) {
                shape.dims.push_back(dim.dim_value());
            } else {
                // Dynamic dimension - use 1 as placeholder
                shape.dims.push_back(1);
            }
        }

        return shape;
    }

    std::string shape_to_string(const Shape& shape) {
        std::string result = "[";
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(shape.dims[i]);
        }
        result += "]";
        return result;
    }

    Tensor convert_tensor(const onnx::TensorProto& tensor_proto) {
        Shape shape;
        for (int i = 0; i < tensor_proto.dims_size(); ++i) {
            shape.dims.push_back(tensor_proto.dims(i));
        }

        DataType dtype;
        switch (tensor_proto.data_type()) {
            case onnx::TensorProto::FLOAT:
                dtype = DataType::FLOAT32;
                break;
            case onnx::TensorProto::INT32:
                dtype = DataType::INT32;
                break;
            case onnx::TensorProto::INT64:
                dtype = DataType::INT64;
                break;
            case onnx::TensorProto::UINT8:
                dtype = DataType::UINT8;
                break;
            case onnx::TensorProto::INT8:
                dtype = DataType::INT8;
                break;
            default:
                TBN_LOG_WARNING("Unsupported tensor data type: " + std::to_string(tensor_proto.data_type()));
                dtype = DataType::FLOAT32;
        }

        Tensor tensor(shape, dtype);

        // Copy data based on type
        if (tensor_proto.has_raw_data()) {
            // Raw data
            const std::string& raw_data = tensor_proto.raw_data();
            std::memcpy(tensor.data(), raw_data.data(), raw_data.size());
        } else if (tensor_proto.float_data_size() > 0) {
            // Float data
            const float* src_data = tensor_proto.float_data().data();
            std::memcpy(tensor.data(), src_data, tensor_proto.float_data_size() * sizeof(float));
        } else if (tensor_proto.int32_data_size() > 0) {
            // Int32 data
            const int32_t* src_data = tensor_proto.int32_data().data();
            std::memcpy(tensor.data(), src_data, tensor_proto.int32_data_size() * sizeof(int32_t));
        } else if (tensor_proto.int64_data_size() > 0) {
            // Int64 data
            const int64_t* src_data = tensor_proto.int64_data().data();
            std::memcpy(tensor.data(), src_data, tensor_proto.int64_data_size() * sizeof(int64_t));
        }

        return tensor;
    }

    ModelNode convert_node(const onnx::NodeProto& onnx_node) {
        ModelNode node;
        node.name = onnx_node.name();
        if (node.name.empty()) {
            // Generate name if not provided
            node.name = onnx_node.op_type() + "_" + onnx_node.output(0);
        }

        node.op_type = onnx_node.op_type();

        // Copy inputs/outputs
        for (const auto& input : onnx_node.input()) {
            node.inputs.push_back(input);
        }

        for (const auto& output : onnx_node.output()) {
            node.outputs.push_back(output);
        }

        // Convert attributes
        for (const auto& attr : onnx_node.attribute()) {
            const std::string& attr_name = attr.name();

            switch (attr.type()) {
                case onnx::AttributeProto::INT:
                    node.attributes[attr_name] = AttributeValue(attr.i());
                    break;

                case onnx::AttributeProto::FLOAT:
                    node.attributes[attr_name] = AttributeValue(attr.f());
                    break;

                case onnx::AttributeProto::STRING:
                    node.attributes[attr_name] = AttributeValue(attr.s());
                    break;

                case onnx::AttributeProto::TENSOR:
                    node.attributes[attr_name] = AttributeValue(convert_tensor(attr.t()));
                    break;

                case onnx::AttributeProto::INTS: {
                    std::vector<int64_t> ints;
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        ints.push_back(attr.ints(i));
                    }
                    node.attributes[attr_name] = AttributeValue(ints);
                    break;
                }

                case onnx::AttributeProto::FLOATS: {
                    std::vector<float> floats;
                    for (int i = 0; i < attr.floats_size(); ++i) {
                        floats.push_back(attr.floats(i));
                    }
                    node.attributes[attr_name] = AttributeValue(floats);
                    break;
                }

                case onnx::AttributeProto::STRINGS: {
                    std::vector<std::string> strings;
                    for (int i = 0; i < attr.strings_size(); ++i) {
                        strings.push_back(attr.strings(i));
                    }
                    node.attributes[attr_name] = AttributeValue(strings);
                    break;
                }

                default:
                    TBN_LOG_WARNING("Unsupported attribute type: " + std::to_string(attr.type()) +
                                   " for attribute '" + attr_name + "'");
                    break;
            }
        }

        return node;
    }
};

// Public API implementation
OnnxParser::OnnxParser() : impl_(std::make_unique<OnnxParserImpl>()) {}

OnnxParser::~OnnxParser() = default;

void OnnxParser::parse_from_file(const std::string& path) {
    impl_->parse_from_file(path);
}

void OnnxParser::parse_from_buffer(const void* data, size_t size) {
    impl_->parse_from_buffer(data, size);
}

std::shared_ptr<ModelGraph> OnnxParser::get_graph() {
    return impl_->convert_to_internal_graph();
}

std::string OnnxParser::get_producer_name() const {
    return impl_->get_producer_name();
}

std::string OnnxParser::get_producer_version() const {
    return impl_->get_producer_version();
}

// Convenience functions implementation
TBNModel load_onnx_model(const std::string& path) {
    OnnxParser parser;
    parser.parse_from_file(path);

    auto graph = parser.get_graph();
    TBNModel model(graph);
    model.set_producer(parser.get_producer_name(), parser.get_producer_version());

    return model;
}

TBNModel load_onnx_model_from_buffer(const void* data, size_t size) {
    OnnxParser parser;
    parser.parse_from_buffer(data, size);

    auto graph = parser.get_graph();
    TBNModel model(graph);
    model.set_producer(parser.get_producer_name(), parser.get_producer_version());

    return model;
}

} // namespace tbn