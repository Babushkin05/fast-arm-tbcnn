#pragma once

#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "forward_declarations.hpp"
#include "types.hpp"
#include "tensor.hpp"
#include "model.hpp"
#include <memory>
#include <vector>
#include <cstring>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace tbn {

class InferenceSession {
public:
    struct Options {
        int num_threads;
        DeviceType device;
        bool enable_profiling;
        bool enable_memory_pool;
        size_t memory_pool_size;

        Options()
            : num_threads(1),
              device(DeviceType::CPU),
              enable_profiling(false),
              enable_memory_pool(true),
              memory_pool_size(16 * 1024 * 1024) {} // 16MB default
    };

private:
    std::shared_ptr<TBNModel> model_;
    Options options_;
    bool initialized_ = false;

    // Runtime state
    std::unordered_map<std::string, Tensor> input_tensors_;
    std::unordered_map<std::string, Tensor> output_tensors_;
    std::unordered_map<std::string, Tensor> intermediate_tensors_;

    // Memory pool for efficient allocation
    class MemoryPool {
    private:
        std::vector<std::shared_ptr<uint8_t>> blocks_;
        size_t block_size_;
        size_t total_allocated_ = 0;

    public:
        MemoryPool(size_t block_size) : block_size_(block_size) {}

        void* allocate(size_t size) {
            if (size > block_size_) {
                // Large allocation, create dedicated block
                auto block = std::shared_ptr<uint8_t>(new uint8_t[size], std::default_delete<uint8_t[]>());
                blocks_.push_back(block);
                total_allocated_ += size;
                return block.get();
            }

            // TODO: Implement proper pool allocation
            auto block = std::shared_ptr<uint8_t>(new uint8_t[block_size_], std::default_delete<uint8_t[]>());
            blocks_.push_back(block);
            total_allocated_ += block_size_;
            return block.get();
        }

        size_t total_allocated() const { return total_allocated_; }
    };

    std::unique_ptr<MemoryPool> memory_pool_;

public:
    InferenceSession(std::shared_ptr<TBNModel> model, const Options& options = Options())
        : model_(model), options_(options) {
        if (options_.enable_memory_pool) {
            memory_pool_ = std::make_unique<MemoryPool>(options_.memory_pool_size);
        }
    }

    void initialize() {
        TBN_CHECK(model_ != nullptr, InvalidArgumentError, "Model is null");

        // Validate model
        model_->validate();

        // Initialize memory pool if enabled
        if (memory_pool_) {
            TBN_LOG_INFO("Memory pool initialized with " + std::to_string(options_.memory_pool_size) + " bytes");
        }

        initialized_ = true;
        TBN_LOG_INFO("Inference session initialized");
    }

    void set_input(const std::string& name, const Tensor& tensor) {
        TBN_CHECK(initialized_, RuntimeError, "Session not initialized");
        TBN_CHECK(model_->has_input(name), InvalidArgumentError,
                  "Input '" + name + "' not found in model");
        TBN_CHECK(tensor.shape() == model_->get_input_shape(name), InvalidShapeError,
                  "Input shape mismatch for '" + name + "'");

        input_tensors_[name] = tensor;
        TBN_LOG_INFO("Set input: " + name + " shape=[" + format_shape(tensor.shape()) + "]");
    }

    Tensor get_input(const std::string& name) const {
        auto it = input_tensors_.find(name);
        TBN_CHECK(it != input_tensors_.end(), InvalidArgumentError,
                  "Input '" + name + "' not set");
        return it->second;
    }

    void run() {
        TBN_CHECK(initialized_, RuntimeError, "Session not initialized");
        TBN_CHECK(input_tensors_.size() == model_->inputs().size(), InvalidArgumentError,
                  "Not all inputs provided");

        auto start_time = std::chrono::high_resolution_clock::now();

        // Clear previous outputs
        output_tensors_.clear();
        intermediate_tensors_.clear();

        // Copy inputs to intermediate tensors
        for (const auto& pair : input_tensors_) {
            intermediate_tensors_[pair.first] = pair.second;
        }

        // Copy initializers to intermediate tensors
        for (const auto& pair : model_->graph().initializers) {
            intermediate_tensors_[pair.first] = pair.second;
        }

        // Execute nodes
        execute_graph();

        // Copy outputs
        for (const auto& output : model_->outputs()) {
            auto it = intermediate_tensors_.find(output);
            TBN_CHECK(it != intermediate_tensors_.end(), RuntimeError,
                      "Output '" + output + "' was not produced");
            output_tensors_[output] = it->second;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (options_.enable_profiling) {
            TBN_LOG_INFO("Inference completed in " + std::to_string(duration.count()) + " μs");
        }
    }

    Tensor get_output(const std::string& name) const {
        auto it = output_tensors_.find(name);
        TBN_CHECK(it != output_tensors_.end(), InvalidArgumentError,
                  "Output '" + name + "' not available");
        return it->second;
    }

    std::vector<std::string> get_input_names() const {
        return model_->inputs();
    }

    std::vector<std::string> get_output_names() const {
        return model_->outputs();
    }

    const Options& options() const { return options_; }

private:
    void execute_graph() {
        const auto& graph = model_->graph();

        // Simple sequential execution for now
        // TODO: Implement topological sort and parallel execution
        for (const auto& node : graph.nodes) {
            execute_node(node);
        }
    }

    void execute_node(const ModelNode& node) {
        TBN_LOG_INFO("Executing node: " + node.name + " (" + node.op_type + ")");

        // This would delegate to the actual operator implementations
        // For now, we'll use the model's internal execution
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
            auto shape = model_->get_output_shape(output_name);
            intermediate_tensors_[output_name] = Tensor(shape, DataType::FLOAT32);
        }
    }

    void execute_gemm(const ModelNode& node) {
        // Placeholder - would call tbn::Gemm operator
        TBN_LOG_WARNING("Gemm operator execution not fully implemented");

        // For now, just create a dummy output tensor
        for (const auto& output_name : node.outputs) {
            auto shape = model_->get_output_shape(output_name);
            intermediate_tensors_[output_name] = Tensor(shape, DataType::FLOAT32);
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

    std::string format_shape(const Shape& shape) {
        std::stringstream ss;
        ss << shape.dims[0];
        for (size_t i = 1; i < shape.dims.size(); ++i) {
            ss << "x" << shape.dims[i];
        }
        return ss.str();
    }
};

// Convenience function
inline std::unique_ptr<InferenceSession> create_inference_session(
    std::shared_ptr<TBNModel> model,
    const InferenceSession::Options& options = InferenceSession::Options()) {

    auto session = std::make_unique<InferenceSession>(model, options);
    session->initialize();
    return session;
}

} // namespace tbn