#pragma once

#include "tensor.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>

namespace tbn {

struct ModelNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, Tensor> attributes;
};

struct ModelGraph {
    std::vector<ModelNode> nodes;
    std::unordered_map<std::string, Tensor> initializers;
    std::unordered_map<std::string, Shape> value_info;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

class TBNModel {
private:
    std::shared_ptr<ModelGraph> graph_;
    std::string model_path_;
    std::string producer_name_;
    std::string producer_version_;

public:
    TBNModel() : graph_(std::make_shared<ModelGraph>()) {}

    TBNModel(std::shared_ptr<ModelGraph> graph) : graph_(graph) {}

    // Graph access
    const ModelGraph& graph() const { return *graph_; }
    ModelGraph& graph() { return *graph_; }

    // Model metadata
    void set_producer(const std::string& name, const std::string& version) {
        producer_name_ = name;
        producer_version_ = version;
    }

    const std::string& producer_name() const { return producer_name_; }
    const std::string& producer_version() const { return producer_version_; }

    // Input/Output info
    const std::vector<std::string>& inputs() const { return graph_->inputs; }
    const std::vector<std::string>& outputs() const { return graph_->outputs; }

    bool has_input(const std::string& name) const {
        return std::find(graph_->inputs.begin(), graph_->inputs.end(), name) != graph_->inputs.end();
    }

    bool has_output(const std::string& name) const {
        return std::find(graph_->outputs.begin(), graph_->outputs.end(), name) != graph_->outputs.end();
    }

    const Shape& get_input_shape(const std::string& name) const {
        auto it = graph_->value_info.find(name);
        TBN_CHECK(it != graph_->value_info.end(), InvalidArgumentError,
                  "Input '" + name + "' not found in model");
        return it->second;
    }

    const Shape& get_output_shape(const std::string& name) const {
        auto it = graph_->value_info.find(name);
        TBN_CHECK(it != graph_->value_info.end(), InvalidArgumentError,
                  "Output '" + name + "' not found in model");
        return it->second;
    }

    // Model validation
    void validate() const {
        TBN_CHECK(!graph_->inputs.empty(), InvalidModelError, "Model has no inputs");
        TBN_CHECK(!graph_->outputs.empty(), InvalidModelError, "Model has no outputs");
        TBN_CHECK(!graph_->nodes.empty(), InvalidModelError, "Model has no computation nodes");

        // Check all inputs and outputs are defined
        for (const auto& input : graph_->inputs) {
            TBN_CHECK(graph_->value_info.count(input) > 0, InvalidModelError,
                      "Input '" + input + "' shape not defined");
        }

        for (const auto& output : graph_->outputs) {
            TBN_CHECK(graph_->value_info.count(output) > 0, InvalidModelError,
                      "Output '" + output + "' shape not defined");
        }

        // Check all node inputs/outputs exist
        for (const auto& node : graph_->nodes) {
            for (const auto& input : node.inputs) {
                TBN_CHECK(graph_->value_info.count(input) > 0 || graph_->initializers.count(input) > 0,
                          InvalidModelError, "Node input '" + input + "' not defined");
            }
            for (const auto& output : node.outputs) {
                TBN_CHECK(graph_->value_info.count(output) > 0,
                          InvalidModelError, "Node output '" + output + "' shape not defined");
            }
        }
    }

    // Session creation
    class Session {
    private:
        std::shared_ptr<ModelGraph> graph_;
        std::unordered_map<std::string, Tensor> input_tensors_;
        std::unordered_map<std::string, Tensor> output_tensors_;
        std::unordered_map<std::string, Tensor> intermediate_tensors_;

    public:
        Session(std::shared_ptr<ModelGraph> graph) : graph_(graph) {}

        void set_input(const std::string& name, const Tensor& tensor) {
            TBN_CHECK(graph_->value_info.count(name) > 0, InvalidArgumentError,
                      "Input '" + name + "' not found in model");
            TBN_CHECK(tensor.shape() == graph_->value_info.at(name), InvalidShapeError,
                      "Input shape mismatch for '" + name + "'");
            input_tensors_[name] = tensor;
        }

        Tensor get_input(const std::string& name) const {
            auto it = input_tensors_.find(name);
            TBN_CHECK(it != input_tensors_.end(), InvalidArgumentError,
                      "Input '" + name + "' not set");
            return it->second;
        }

        void run() {
            TBN_CHECK(input_tensors_.size() == graph_->inputs.size(), InvalidArgumentError,
                      "Not all inputs provided");

            // Clear previous outputs
            output_tensors_.clear();
            intermediate_tensors_.clear();

            // Copy inputs to intermediate tensors
            for (const auto& pair : input_tensors_) {
                intermediate_tensors_[pair.first] = pair.second;
            }

            // Copy initializers to intermediate tensors
            for (const auto& pair : graph_->initializers) {
                intermediate_tensors_[pair.first] = pair.second;
            }

            // Execute nodes in order (simple sequential execution for now)
            for (const auto& node : graph_->nodes) {
                execute_node(node);
            }

            // Copy outputs
            for (const auto& output : graph_->outputs) {
                auto it = intermediate_tensors_.find(output);
                TBN_CHECK(it != intermediate_tensors_.end(), RuntimeError,
                          "Output '" + output + "' was not produced");
                output_tensors_[output] = it->second;
            }
        }

        Tensor get_output(const std::string& name) const {
            auto it = output_tensors_.find(name);
            TBN_CHECK(it != output_tensors_.end(), InvalidArgumentError,
                      "Output '" + name + "' not available");
            return it->second;
        }

    private:
        void execute_node(const ModelNode& node) {
            TBN_LOG_INFO("Executing node: " + node.name + " (" + node.op_type + ")");

            // This is a placeholder - actual implementation would delegate to operators
            if (node.op_type == "Conv") {
                execute_conv(node);
            } else if (node.op_type == "Gemm") {
                execute_gemm(node);
            } else if (node.op_type == "Relu") {
                execute_relu(node);
            } else {
                throw NotImplementedError("Operator '" + node.op_type + "' not implemented");
            }
        }

        void execute_conv(const ModelNode& node) {
            // Placeholder - would call tbn::Conv2D operator
            TBN_LOG_WARNING("Conv operator execution not fully implemented");

            // For now, just create a dummy output tensor
            for (const auto& output_name : node.outputs) {
                auto it = graph_->value_info.find(output_name);
                if (it != graph_->value_info.end()) {
                    intermediate_tensors_[output_name] = Tensor(it->second, DataType::FLOAT32);
                }
            }
        }

        void execute_gemm(const ModelNode& node) {
            // Placeholder - would call tbn::Gemm operator
            TBN_LOG_WARNING("Gemm operator execution not fully implemented");

            // For now, just create a dummy output tensor
            for (const auto& output_name : node.outputs) {
                auto it = graph_->value_info.find(output_name);
                if (it != graph_->value_info.end()) {
                    intermediate_tensors_[output_name] = Tensor(it->second, DataType::FLOAT32);
                }
            }
        }

        void execute_relu(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() == 1, InvalidArgumentError, "ReLU expects 1 input");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "ReLU expects 1 output");

            const auto& input_name = node.inputs[0];
            const auto& output_name = node.outputs[0];

            auto it = intermediate_tensors_.find(input_name);
            TBN_CHECK(it != intermediate_tensors_.end(), RuntimeError, "Input tensor not found");

            const Tensor& input = it->second;
            Tensor output(input.shape(), input.dtype());

            // Simple CPU implementation
            const float* input_data = input.typed_data<float>();
            float* output_data = output.typed_data<float>();

            for (int64_t i = 0; i < input.num_elements(); ++i) {
                output_data[i] = std::max(0.0f, input_data[i]);
            }

            intermediate_tensors_[output_name] = output;
        }
    };

    Session create_session() {
        return Session(graph_);
    }
};

} // namespace tbn

// Include standard libraries
#include <cmath>
#include <algorithm>

namespace tbn {

// Forward declarations for the main API
TBNModel load_model(const std::string& path);
TBNModel load_model_from_buffer(const void* data, size_t size);
const char* get_version();

} // namespace tbn