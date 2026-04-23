#pragma once

#include "tensor.hpp"
#include "attribute.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "../operators/gemm.hpp"
#include "../operators/conv2d.hpp"
#include "../operators/quantized_gemm.hpp"
#include "../operators/pooling.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace tbn {

struct ModelNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    NodeAttributes attributes;  // Flexible attribute storage
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

            // Map ONNX operators to implementations
            if (node.op_type == "Conv") {
                execute_conv(node);
            } else if (node.op_type == "Gemm") {
                execute_gemm(node);
            } else if (node.op_type == "MatMul") {
                execute_matmul(node);
            } else if (node.op_type == "Add") {
                execute_add(node);
            } else if (node.op_type == "Relu") {
                execute_relu(node);
            } else if (node.op_type == "Reshape") {
                execute_reshape(node);
            } else if (node.op_type == "Flatten") {
                execute_flatten(node);
            } else if (node.op_type == "MaxPool") {
                execute_maxpool(node);
            } else if (node.op_type == "AveragePool" || node.op_type == "AvgPool") {
                execute_avgpool(node);
            } else if (node.op_type == "GlobalMaxPool") {
                execute_global_maxpool(node);
            } else if (node.op_type == "GlobalAveragePool" || node.op_type == "GlobalAvgPool") {
                execute_global_avgpool(node);
            } else {
                throw NotImplementedError("Operator '" + node.op_type + "' not implemented");
            }
        }

        void execute_conv(const ModelNode& node) {
            // ONNX Conv: Y = Conv(X, W, B)
            // X: input [N, C, H, W]
            // W: weights [M, C/group, kH, kW]
            // B: optional bias [M]
            TBN_CHECK(node.inputs.size() >= 2, InvalidArgumentError, "Conv expects at least 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Conv expects 1 output");

            const auto& X_name = node.inputs[0];
            const auto& W_name = node.inputs[1];

            auto it_X = intermediate_tensors_.find(X_name);
            auto it_W = intermediate_tensors_.find(W_name);
            TBN_CHECK(it_X != intermediate_tensors_.end(), RuntimeError, "Conv input X not found");
            TBN_CHECK(it_W != intermediate_tensors_.end(), RuntimeError, "Conv weights W not found");

            const Tensor& X = it_X->second;
            const Tensor& W = it_W->second;

            // Extract Conv2DParams from ONNX attributes
            Conv2DParams params;

            // Get kernel_shape from weights if not in attributes
            params.kernel_h = W.shape().dims[2];
            params.kernel_w = W.shape().dims[3];

            // Extract attributes
            auto attr_it = node.attributes.find("strides");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& strides = attr_it->second.as_ints();
                if (strides.size() >= 2) {
                    params.stride_h = strides[0];
                    params.stride_w = strides[1];
                }
            }

            attr_it = node.attributes.find("pads");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& pads = attr_it->second.as_ints();
                if (pads.size() >= 4) {
                    // ONNX pads: [begin_h, begin_w, end_h, end_w]
                    // We assume symmetric padding for now
                    params.pad_h = pads[0];
                    params.pad_w = pads[1];
                }
            }

            attr_it = node.attributes.find("dilations");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& dilations = attr_it->second.as_ints();
                if (dilations.size() >= 2) {
                    params.dilation_h = dilations[0];
                    params.dilation_w = dilations[1];
                }
            }

            attr_it = node.attributes.find("group");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                params.groups = attr_it->second.as_int();
            }

            // Optional bias
            const Tensor* B = nullptr;
            if (node.inputs.size() >= 3) {
                const auto& B_name = node.inputs[2];
                auto it_B = intermediate_tensors_.find(B_name);
                if (it_B != intermediate_tensors_.end()) {
                    B = &it_B->second;
                }
            }

            // Execute convolution with automatic quantization for optimization
            Tensor result;
            if (params.groups == 1) {
                // Check weight type and use optimized path when possible
                if (W.dtype() == DataType::BINARY) {
                    // Already binary - use optimized Conv2D
                    TBN_LOG_INFO("Conv: using optimized binary path");
                    result = conv2d_binary(X, W, B, params);
                } else if (W.dtype() == DataType::TERNARY) {
                    // Ternary weights - convert and use optimized path
                    TBN_LOG_INFO("Conv: using optimized ternary path");
                    result = conv2d_ternary(X, W, B, params);
                } else if (W.dtype() == DataType::FLOAT32) {
                    // Float weights - quantize to binary on-the-fly for optimization
                    TBN_LOG_INFO("Conv: quantizing float weights to binary for optimization");
                    Tensor binary_W = quantize_to_binary(W);
                    result = conv2d_binary(X, binary_W, B, params);
                } else {
                    // Fallback to naive implementation
                    result = conv2d(X, W, B, params);
                }
            } else {
                result = conv2d_grouped(X, W, B, params.groups, params);
            }

            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_maxpool(const ModelNode& node) {
            // ONNX MaxPool
            TBN_CHECK(node.inputs.size() >= 1, InvalidArgumentError, "MaxPool expects at least 1 input");
            TBN_CHECK(node.outputs.size() >= 1, InvalidArgumentError, "MaxPool expects at least 1 output");

            const auto& X_name = node.inputs[0];
            auto it_X = intermediate_tensors_.find(X_name);
            TBN_CHECK(it_X != intermediate_tensors_.end(), RuntimeError, "MaxPool input not found");

            const Tensor& X = it_X->second;

            // Extract Pool2DParams from ONNX attributes
            Pool2DParams params;

            auto attr_it = node.attributes.find("kernel_shape");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& kernel = attr_it->second.as_ints();
                if (kernel.size() >= 2) {
                    params.kernel_h = kernel[0];
                    params.kernel_w = kernel[1];
                }
            }

            attr_it = node.attributes.find("strides");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& strides = attr_it->second.as_ints();
                if (strides.size() >= 2) {
                    params.stride_h = strides[0];
                    params.stride_w = strides[1];
                }
            } else {
                // Default stride = kernel size if not specified
                params.stride_h = params.kernel_h;
                params.stride_w = params.kernel_w;
            }

            attr_it = node.attributes.find("pads");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& pads = attr_it->second.as_ints();
                if (pads.size() >= 4) {
                    params.pad_h = pads[0];
                    params.pad_w = pads[1];
                }
            }

            attr_it = node.attributes.find("ceil_mode");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                params.ceil_mode = (attr_it->second.as_int() != 0);
            }

            Tensor result = maxpool2d(X, params);
            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_avgpool(const ModelNode& node) {
            // ONNX AveragePool
            TBN_CHECK(node.inputs.size() >= 1, InvalidArgumentError, "AveragePool expects at least 1 input");
            TBN_CHECK(node.outputs.size() >= 1, InvalidArgumentError, "AveragePool expects at least 1 output");

            const auto& X_name = node.inputs[0];
            auto it_X = intermediate_tensors_.find(X_name);
            TBN_CHECK(it_X != intermediate_tensors_.end(), RuntimeError, "AveragePool input not found");

            const Tensor& X = it_X->second;

            Pool2DParams params;

            auto attr_it = node.attributes.find("kernel_shape");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& kernel = attr_it->second.as_ints();
                if (kernel.size() >= 2) {
                    params.kernel_h = kernel[0];
                    params.kernel_w = kernel[1];
                }
            }

            attr_it = node.attributes.find("strides");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& strides = attr_it->second.as_ints();
                if (strides.size() >= 2) {
                    params.stride_h = strides[0];
                    params.stride_w = strides[1];
                }
            } else {
                params.stride_h = params.kernel_h;
                params.stride_w = params.kernel_w;
            }

            attr_it = node.attributes.find("pads");
            if (attr_it != node.attributes.end() && attr_it->second.is_ints()) {
                const auto& pads = attr_it->second.as_ints();
                if (pads.size() >= 4) {
                    params.pad_h = pads[0];
                    params.pad_w = pads[1];
                }
            }

            attr_it = node.attributes.find("count_include_pad");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                params.count_include_pad = (attr_it->second.as_int() != 0);
            }

            Tensor result = avgpool2d(X, params);
            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_global_maxpool(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() >= 1, InvalidArgumentError, "GlobalMaxPool expects at least 1 input");
            TBN_CHECK(node.outputs.size() >= 1, InvalidArgumentError, "GlobalMaxPool expects at least 1 output");

            const auto& X_name = node.inputs[0];
            auto it_X = intermediate_tensors_.find(X_name);
            TBN_CHECK(it_X != intermediate_tensors_.end(), RuntimeError, "GlobalMaxPool input not found");

            Tensor result = global_maxpool2d(it_X->second);
            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_global_avgpool(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() >= 1, InvalidArgumentError, "GlobalAveragePool expects at least 1 input");
            TBN_CHECK(node.outputs.size() >= 1, InvalidArgumentError, "GlobalAveragePool expects at least 1 output");

            const auto& X_name = node.inputs[0];
            auto it_X = intermediate_tensors_.find(X_name);
            TBN_CHECK(it_X != intermediate_tensors_.end(), RuntimeError, "GlobalAveragePool input not found");

            Tensor result = global_avgpool2d(it_X->second);
            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_gemm(const ModelNode& node) {
            // ONNX Gemm: Y = alpha * A' * B' + beta * C
            // A' = transA ? A^T : A
            // B' = transB ? B^T : B
            TBN_CHECK(node.inputs.size() >= 2, InvalidArgumentError, "Gemm expects at least 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Gemm expects 1 output");

            const auto& A_name = node.inputs[0];
            const auto& B_name = node.inputs[1];

            auto it_A = intermediate_tensors_.find(A_name);
            auto it_B = intermediate_tensors_.find(B_name);
            TBN_CHECK(it_A != intermediate_tensors_.end(), RuntimeError, "Gemm input A not found");
            TBN_CHECK(it_B != intermediate_tensors_.end(), RuntimeError, "Gemm input B not found");

            const Tensor& A = it_A->second;
            const Tensor& B = it_B->second;

            // Get attributes with defaults
            float alpha = 1.0f;
            float beta = 1.0f;
            bool transA = false;
            bool transB = false;

            auto attr_it = node.attributes.find("alpha");
            if (attr_it != node.attributes.end() && attr_it->second.is_float()) {
                alpha = attr_it->second.as_float();
            }
            attr_it = node.attributes.find("beta");
            if (attr_it != node.attributes.end() && attr_it->second.is_float()) {
                beta = attr_it->second.as_float();
            }
            attr_it = node.attributes.find("transA");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                transA = attr_it->second.as_int() != 0;
            }
            attr_it = node.attributes.find("transB");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                transB = attr_it->second.as_int() != 0;
            }

            // Optional bias C
            const Tensor* C = nullptr;
            if (node.inputs.size() >= 3) {
                const auto& C_name = node.inputs[2];
                auto it_C = intermediate_tensors_.find(C_name);
                if (it_C != intermediate_tensors_.end()) {
                    C = &it_C->second;
                }
            }

            // Call GEMM implementation
            Tensor result = gemm(A, B, C, alpha, beta, transA, transB);
            intermediate_tensors_[node.outputs[0]] = result;
        }

        void execute_matmul(const ModelNode& node) {
            // ONNX MatMul: Y = A * B (no transposition, no alpha/beta)
            TBN_CHECK(node.inputs.size() == 2, InvalidArgumentError, "MatMul expects 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "MatMul expects 1 output");

            const auto& A_name = node.inputs[0];
            const auto& B_name = node.inputs[1];

            auto it_A = intermediate_tensors_.find(A_name);
            auto it_B = intermediate_tensors_.find(B_name);
            TBN_CHECK(it_A != intermediate_tensors_.end(), RuntimeError, "MatMul input A not found");
            TBN_CHECK(it_B != intermediate_tensors_.end(), RuntimeError, "MatMul input B not found");

            const Tensor& A = it_A->second;
            const Tensor& B = it_B->second;

            // For 2D tensors, use GEMM
            if (A.shape().dims.size() == 2 && B.shape().dims.size() == 2) {
                Tensor result = gemm(A, B, nullptr, 1.0f, 0.0f, false, false);
                intermediate_tensors_[node.outputs[0]] = result;
            } else {
                throw NotImplementedError("MatMul for non-2D tensors not implemented");
            }
        }

        void execute_add(const ModelNode& node) {
            // ONNX Add: Y = A + B (element-wise with broadcasting)
            TBN_CHECK(node.inputs.size() == 2, InvalidArgumentError, "Add expects 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Add expects 1 output");

            const auto& A_name = node.inputs[0];
            const auto& B_name = node.inputs[1];

            auto it_A = intermediate_tensors_.find(A_name);
            auto it_B = intermediate_tensors_.find(B_name);
            TBN_CHECK(it_A != intermediate_tensors_.end(), RuntimeError, "Add input A not found");
            TBN_CHECK(it_B != intermediate_tensors_.end(), RuntimeError, "Add input B not found");

            const Tensor& A = it_A->second;
            const Tensor& B = it_B->second;

            // Support simple broadcasting: 2D + 1D (bias case)
            if (A.shape().dims.size() == 2 && B.shape().dims.size() == 1) {
                // A is [M, N], B is [N] -> broadcast B across rows
                int64_t M = A.shape().dims[0];
                int64_t N = A.shape().dims[1];
                TBN_CHECK(B.shape().dims[0] == N, InvalidShapeError,
                          "Broadcasting: B size must match last dim of A");

                Tensor output(A.shape(), A.dtype());
                const float* A_data = A.typed_data<float>();
                const float* B_data = B.typed_data<float>();
                float* output_data = output.typed_data<float>();

                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        output_data[i * N + j] = A_data[i * N + j] + B_data[j];
                    }
                }
                intermediate_tensors_[node.outputs[0]] = output;
                return;
            }

            if (A.shape().dims.size() == 1 && B.shape().dims.size() == 2) {
                // B is [M, N], A is [N] -> broadcast A across rows
                int64_t M = B.shape().dims[0];
                int64_t N = B.shape().dims[1];
                TBN_CHECK(A.shape().dims[0] == N, InvalidShapeError,
                          "Broadcasting: A size must match last dim of B");

                Tensor output(B.shape(), B.dtype());
                const float* A_data = A.typed_data<float>();
                const float* B_data = B.typed_data<float>();
                float* output_data = output.typed_data<float>();

                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        output_data[i * N + j] = A_data[j] + B_data[i * N + j];
                    }
                }
                intermediate_tensors_[node.outputs[0]] = output;
                return;
            }

            // Same shape case
            TBN_CHECK(A.shape() == B.shape(), InvalidShapeError,
                      "Add requires same shape tensors or broadcastable shapes");

            Tensor output(A.shape(), A.dtype());
            const float* A_data = A.typed_data<float>();
            const float* B_data = B.typed_data<float>();
            float* output_data = output.typed_data<float>();

            for (int64_t i = 0; i < A.num_elements(); ++i) {
                output_data[i] = A_data[i] + B_data[i];
            }

            intermediate_tensors_[node.outputs[0]] = output;
        }

        void execute_reshape(const ModelNode& node) {
            // ONNX Reshape
            TBN_CHECK(node.inputs.size() >= 1, InvalidArgumentError, "Reshape expects at least 1 input");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Reshape expects 1 output");

            const auto& data_name = node.inputs[0];
            auto it_data = intermediate_tensors_.find(data_name);
            TBN_CHECK(it_data != intermediate_tensors_.end(), RuntimeError, "Reshape input not found");

            const Tensor& data = it_data->second;

            // Get shape from second input or attribute
            Shape new_shape;
            if (node.inputs.size() >= 2) {
                const auto& shape_name = node.inputs[1];
                auto it_shape = intermediate_tensors_.find(shape_name);
                if (it_shape != intermediate_tensors_.end()) {
                    const Tensor& shape_tensor = it_shape->second;
                    const int64_t* shape_data = shape_tensor.typed_data<int64_t>();
                    for (int i = 0; i < shape_tensor.num_elements(); ++i) {
                        new_shape.dims.push_back(shape_data[i]);
                    }
                }
            }

            // Handle -1 dimension (infer)
            int64_t known_size = 1;
            int unknown_dim = -1;
            for (size_t i = 0; i < new_shape.dims.size(); ++i) {
                if (new_shape.dims[i] == -1) {
                    unknown_dim = i;
                } else {
                    known_size *= new_shape.dims[i];
                }
            }
            if (unknown_dim >= 0) {
                new_shape.dims[unknown_dim] = data.num_elements() / known_size;
            }

            // Create reshaped tensor (shares data)
            Tensor output(new_shape, data.dtype());
            std::memcpy(output.data(), data.data(), data.num_elements() * sizeof(float));
            intermediate_tensors_[node.outputs[0]] = output;
        }

        void execute_flatten(const ModelNode& node) {
            // ONNX Flatten: flatten from axis dimension
            TBN_CHECK(node.inputs.size() == 1, InvalidArgumentError, "Flatten expects 1 input");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Flatten expects 1 output");

            const auto& input_name = node.inputs[0];
            auto it_input = intermediate_tensors_.find(input_name);
            TBN_CHECK(it_input != intermediate_tensors_.end(), RuntimeError, "Flatten input not found");

            const Tensor& input = it_input->second;

            // Get axis (default 1)
            int axis = 1;
            auto attr_it = node.attributes.find("axis");
            if (attr_it != node.attributes.end() && attr_it->second.is_int()) {
                axis = static_cast<int>(attr_it->second.as_int());
            }

            // Compute output shape
            int64_t outer_size = 1;
            int64_t inner_size = 1;
            for (int i = 0; i < axis; ++i) {
                outer_size *= input.shape().dims[i];
            }
            for (size_t i = axis; i < input.shape().dims.size(); ++i) {
                inner_size *= input.shape().dims[i];
            }

            Shape new_shape{outer_size, inner_size};
            Tensor output(new_shape, input.dtype());
            std::memcpy(output.data(), input.data(), input.num_elements() * sizeof(float));
            intermediate_tensors_[node.outputs[0]] = output;
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

namespace tbn {

// Forward declarations for the main API
TBNModel load_model(const std::string& path);
TBNModel load_model_from_buffer(const void* data, size_t size);
const char* get_version();

} // namespace tbn