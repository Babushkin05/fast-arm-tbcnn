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
#include <unordered_map>

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

    std::unordered_map<std::string, Tensor> input_tensors_;
    std::unordered_map<std::string, Tensor> output_tensors_;

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
                auto block = std::shared_ptr<uint8_t>(new uint8_t[size], std::default_delete<uint8_t[]>());
                blocks_.push_back(block);
                total_allocated_ += size;
                return block.get();
            }
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
        model_->validate();

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

        // Delegate to TBNModel::Session which has full operator implementations
        auto session = model_->create_session();
        for (const auto& [name, tensor] : input_tensors_) {
            session.set_input(name, tensor);
        }
        session.run();

        output_tensors_.clear();
        for (const auto& output : model_->outputs()) {
            output_tensors_[output] = session.get_output(output);
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
