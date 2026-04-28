#pragma once

#include "tensor.hpp"
#include "attribute.hpp"
#include "../utils/errors.hpp"
#include "../utils/logging.hpp"
#include "../operators/gemm.hpp"
#include "../operators/conv2d.hpp"
#include "../operators/quantized_gemm.hpp"
#include "../operators/pooling.hpp"
#include "../../../../GeMM/05-final/GeMM.hpp"
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
        bool use_quantization_ = false;  // Enable binary weight quantization

        // Weight pre-packing caches (Phase 1: avoid per-inference packing)
        // Key: weight initializer name (W_name from node)
        // For pre-packed binary weights (pre-converted to BinaryMatrix)
        struct CachedBinaryWeights {
            BinaryMatrix b_packed;
            TilingParams tiling;
            std::uint32_t original_k{};
            std::uint32_t original_n{};
        };
        std::unordered_map<std::string, CachedBinaryWeights> cached_binary_weights_;

        // Cache for the "already_binary" check per weight tensor
        struct CachedChannelScales {
            std::vector<float> scales;
            bool already_binary{};
        };
        std::unordered_map<std::string, CachedChannelScales> cached_channel_scales_;

        // Cache for extracted binary weight tensors (avoid per-inference extraction)
        std::unordered_map<std::string, Tensor> cached_binary_tensors_;

        // Cache for pre-packed BinaryMatrix (avoids per-inference int8 conversion + packing)
        struct CachedPackedBinaryMatrix {
            BinaryMatrix matrix;
            uint32_t n_orig;     // original N (unpadded)
            TilingParams tiling;
        };
        std::unordered_map<std::string, CachedPackedBinaryMatrix> cached_packed_b_;

        // Pre-allocated buffers reused across inference (Phase 6)
        std::vector<std::uint8_t> im2col_buffer_;
        std::vector<float> packed_activation_buffer_;

    public:
        Session(std::shared_ptr<ModelGraph> graph) : graph_(graph) {}

        // Enable/disable quantization for faster inference
        void set_quantization(bool enable) { use_quantization_ = enable; }
        bool is_quantization_enabled() const { return use_quantization_; }

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
            // Count inputs that are NOT in initializers (actual user inputs)
            size_t user_input_count = 0;
            for (const auto& name : graph_->inputs) {
                if (graph_->initializers.find(name) == graph_->initializers.end()) {
                    user_input_count++;
                }
            }
            TBN_CHECK(input_tensors_.size() == user_input_count, InvalidArgumentError,
                      "Not all inputs provided (expected " + std::to_string(user_input_count) +
                      ", got " + std::to_string(input_tensors_.size()) + ")");

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
            TBN_LOG_DEBUG("Executing node: " + node.name + " (" + node.op_type + ")");

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
            } else if (node.op_type == "Greater") {
                execute_greater(node);
            } else if (node.op_type == "Less") {
                execute_less(node);
            } else if (node.op_type == "Cast") {
                execute_cast(node);
            } else if (node.op_type == "Sub") {
                execute_sub(node);
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

            // Handle auto_pad attribute (SAME_UPPER, SAME_LOWER, NOTSET)
            attr_it = node.attributes.find("auto_pad");
            if (attr_it != node.attributes.end() && attr_it->second.is_string()) {
                const std::string& auto_pad = attr_it->second.as_string();
                if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                    // Calculate padding for same output size
                    // pad = (kernel - 1) / 2 for stride=1, dilation=1
                    int64_t kernel_h = params.kernel_h > 0 ? params.kernel_h : W.shape().dims[2];
                    int64_t kernel_w = params.kernel_w > 0 ? params.kernel_w : W.shape().dims[3];
                    int64_t total_pad_h = (kernel_h - 1) * params.dilation_h;
                    int64_t total_pad_w = (kernel_w - 1) * params.dilation_w;
                    params.pad_h = total_pad_h / 2;
                    params.pad_w = total_pad_w / 2;
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
                    TBN_LOG_DEBUG("Conv: using optimized binary path");
                    result = conv2d_binary(X, W, B, params);
                } else if (W.dtype() == DataType::TERNARY) {
                    TBN_LOG_DEBUG("Conv: using optimized ternary path");
                    result = conv2d_ternary(X, W, B, params);
                } else if (W.dtype() == DataType::FLOAT32) {
                    if (use_quantization_) {
                        // Check cache for already_binary computation (Phase 1: skip per-inference detection)
                        auto cache_it = cached_channel_scales_.find(W_name);
                        if (cache_it == cached_channel_scales_.end()) {
                            // First inference: compute per-channel scales and check if binary
                            const float* w_data = W.typed_data<float>();
                            int64_t n_elems = W.num_elements();
                            int64_t n_channels = W.shape().dims[0];
                            int64_t elems_per_channel = n_elems / n_channels;

                            CachedChannelScales cached;
                            cached.scales.resize(n_channels);

                            for (int64_t c = 0; c < n_channels; ++c) {
                                float sum = 0.0f;
                                for (int64_t i = 0; i < elems_per_channel; ++i) {
                                    sum += std::abs(w_data[c * elems_per_channel + i]);
                                }
                                cached.scales[c] = sum / elems_per_channel;
                            }

                            cached.already_binary = true;
                            for (int64_t c = 0; c < n_channels && cached.already_binary; ++c) {
                                float scale = cached.scales[c];
                                int64_t near_scale_count = 0;
                                for (int64_t i = 0; i < elems_per_channel; ++i) {
                                    float val = w_data[c * elems_per_channel + i];
                                    float expected = (val >= 0) ? scale : -scale;
                                    if (std::abs(val - expected) < 0.01f * scale || std::abs(val) < 0.01f) {
                                        near_scale_count++;
                                    }
                                }
                                if (near_scale_count < elems_per_channel * 0.9f) {
                                    cached.already_binary = false;
                                }
                            }
                            cache_it = cached_channel_scales_.emplace(W_name, std::move(cached)).first;
                        }

                        if (cache_it->second.already_binary) {
                            TBN_LOG_DEBUG("Conv: extracting binary weights (cached)");

                            // Check if we already have the extracted binary tensor cached
                            auto bin_it = cached_binary_tensors_.find(W_name);
                            const Tensor* binary_W_ptr;
                            if (bin_it != cached_binary_tensors_.end()) {
                                binary_W_ptr = &bin_it->second;
                            } else {
                                // First time: extract binary signs into a tensor
                                Tensor binary_W(W.shape(), DataType::BINARY);
                                BinaryWeight* bw_data = binary_W.typed_data<BinaryWeight>();
                                const float* w_data = W.typed_data<float>();
                                int64_t n_channels = W.shape().dims[0];
                                int64_t elems_per_channel = W.num_elements() / n_channels;

                                for (int64_t c = 0; c < n_channels; ++c) {
                                    for (int64_t i = 0; i < elems_per_channel; ++i) {
                                        bw_data[c * elems_per_channel + i] =
                                            (w_data[c * elems_per_channel + i] >= 0) ? BINARY_ONE : BINARY_ZERO;
                                    }
                                }
                                bin_it = cached_binary_tensors_.emplace(W_name, std::move(binary_W)).first;
                                binary_W_ptr = &bin_it->second;
                            }

                            // Get weight dimensions for im2col + GEMM
                            const auto& w4d_shape = binary_W_ptr->shape();
                            int64_t M_out = w4d_shape.dims[0];
                            int64_t C_in = w4d_shape.dims[1];
                            int64_t kH_conv = w4d_shape.dims[2];
                            int64_t kW_conv = w4d_shape.dims[3];
                            int64_t K_conv = C_in * kH_conv * kW_conv;

                            // Output dims (computed before im2col for padding)
                            const auto& x_shape = X.shape();
                            int64_t N_batch = x_shape.dims[0];
                            int64_t H_in = x_shape.dims[2];
                            int64_t W_in = x_shape.dims[3];
                            int64_t out_h = (H_in + 2*params.pad_h - params.dilation_h*(kH_conv-1) - 1) / params.stride_h + 1;
                            int64_t out_w = (W_in + 2*params.pad_w - params.dilation_w*(kW_conv-1) - 1) / params.stride_w + 1;

                            // Get or create transposed+packed BinaryMatrix (cache key: W_name + "_t")
                            std::string packed_key = W_name + "_t";
                            auto packed_it = cached_packed_b_.find(packed_key);
                            if (packed_it == cached_packed_b_.end()) {
                                // Reshape 4D [M,C,kH,kW] -> 2D [M,K], transpose to [K,M], pack
                                uint32_t k_padded = ((static_cast<uint32_t>(K_conv) + 127) / 128) * 128;
                                uint32_t n_padded = ((static_cast<uint32_t>(M_out) + 7) / 8) * 8;

                                std::vector<int8_t> b_int8(static_cast<size_t>(k_padded) * n_padded, 1);
                                const BinaryWeight* w_bin_data = binary_W_ptr->typed_data<BinaryWeight>();
                                for (int64_t m = 0; m < M_out; ++m) {
                                    for (int64_t k = 0; k < K_conv; ++k) {
                                        // w_bin_data[m*K + k] -> transposed: b_int8[k * n_padded + m]
                                        b_int8[k * n_padded + m] =
                                            (w_bin_data[m * K_conv + k] == BINARY_ONE) ? +1 : -1;
                                    }
                                }

                                CachedPackedBinaryMatrix entry;
                                entry.matrix = BinaryMatrix::pack(
                                    std::span<const int8_t>(b_int8.data(), b_int8.size()),
                                    k_padded, n_padded);
                                entry.n_orig = static_cast<uint32_t>(M_out);
                                entry.tiling = tiling_config::get();
                                packed_it = cached_packed_b_.emplace(packed_key, std::move(entry)).first;
                            }

                            // Fused: im2col + float->ternary quantization in single pass
                            // Eliminates ~15MB intermediate float im2col buffer
                            const auto& tp = packed_it->second.tiling;
                            uint32_t m_orig = static_cast<uint32_t>(N_batch * out_h * out_w);
                            uint32_t m_padded = ((m_orig + tp.mmk - 1) / tp.mmk) * tp.mmk;
                            uint32_t k_padded = static_cast<uint32_t>(packed_it->second.matrix.rows());

                            const float* x_data = X.typed_data<float>();
                            TernaryMatrix a_packed = impl::im2col_ternary_packed(
                                x_data, N_batch, C_in, H_in, W_in,
                                kH_conv, kW_conv,
                                params.stride_h, params.stride_w,
                                params.pad_h, params.pad_w,
                                params.dilation_h, params.dilation_w,
                                m_padded, k_padded,
                                -0.1f, 0.1f);

                            // GEMM: both A and B pre-packed, zero conversions
                            Tensor result_2d = qlinear_matmul_binary_blocked_prepacked(
                                a_packed, packed_it->second.matrix,
                                m_orig, packed_it->second.n_orig,
                                1.0f, tp);

                            // Reshape to 4D [N, M, OH, OW]
                            Shape out4d_shape{N_batch, M_out, out_h, out_w};
                            Tensor result_tensor(out4d_shape, DataType::FLOAT32);
                            {
                                const float* src = result_2d.typed_data<float>();
                                float* dst = result_tensor.typed_data<float>();
                                for (int64_t n = 0; n < N_batch; ++n) {
                                    for (int64_t m = 0; m < M_out; ++m) {
                                        for (int64_t oh = 0; oh < out_h; ++oh) {
                                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                                dst[((n * M_out + m) * out_h + oh) * out_w + ow] =
                                                    src[((n * out_h + oh) * out_w + ow) * M_out + m];
                                            }
                                        }
                                    }
                                }
                            }

                            // Add bias
                            if (B) {
                                const float* bias_data = B->typed_data<float>();
                                float* out_data = result_tensor.typed_data<float>();
                                for (int64_t n = 0; n < N_batch; ++n) {
                                    for (int64_t m = 0; m < M_out; ++m) {
                                        float bv = bias_data[m];
                                        for (int64_t oh = 0; oh < out_h; ++oh) {
                                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                                int64_t idx = ((n * M_out + m) * out_h + oh) * out_w + ow;
                                                out_data[idx] += bv;
                                            }
                                        }
                                    }
                                }
                            }

                            // Apply per-channel scales
                            {
                                const auto& scales = cache_it->second.scales;
                                float* out_data = result_tensor.typed_data<float>();
                                for (int64_t n = 0; n < N_batch; ++n) {
                                    for (int64_t m = 0; m < M_out; ++m) {
                                        float s = scales[m];
                                        if (s != 1.0f) {
                                            for (int64_t oh = 0; oh < out_h; ++oh) {
                                                for (int64_t ow = 0; ow < out_w; ++ow) {
                                                    int64_t idx = ((n * M_out + m) * out_h + oh) * out_w + ow;
                                                    out_data[idx] *= s;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            result = result_tensor;
                        } else {
                            TBN_LOG_DEBUG("Conv: quantizing float weights to binary");
                            Tensor binary_W = quantize_to_binary(W);
                            result = conv2d_binary(X, binary_W, B, params);
                        }
                    } else {
                        result = conv2d(X, W, B, params);
                    }
                } else {
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
            Tensor result;

            // Check if quantization is enabled and weights are suitable
            if (use_quantization_ && B.dtype() == DataType::FLOAT32 && !transA && !transB) {
                // Check cache for already_binary (Phase 1: skip per-inference sampling)
                auto cache_it = cached_channel_scales_.find(B_name);
                bool already_binary;
                if (cache_it != cached_channel_scales_.end()) {
                    already_binary = true;  // if in cache, it passed the check before
                } else {
                    const float* b_data = B.typed_data<float>();
                    int64_t b_elems = B.num_elements();
                    already_binary = true;
                    int64_t sample_step = std::max(int64_t(1), b_elems / 1000);
                    for (int64_t i = 0; i < b_elems && already_binary; i += sample_step) {
                        float val = b_data[i];
                        if (val != -1.0f && val != 1.0f && val != 0.0f &&
                            val != 0.999f && val != -0.999f) {
                            already_binary = false;
                        }
                    }
                    if (already_binary) {
                        CachedChannelScales entry;
                        entry.already_binary = true;
                        cached_channel_scales_[B_name] = entry;
                    }
                }

                if (already_binary) {
                    TBN_LOG_DEBUG("Gemm: using pre-quantized binary weights (cached)");

                    // Check if we already have the extracted binary tensor cached
                    auto bin_it = cached_binary_tensors_.find(B_name);
                    const Tensor* binary_B_ptr;
                    if (bin_it != cached_binary_tensors_.end()) {
                        binary_B_ptr = &bin_it->second;
                    } else {
                        // First time: extract binary signs into a BINARY tensor
                        Tensor binary_B(B.shape(), DataType::BINARY);
                        BinaryWeight* bw_data = binary_B.typed_data<BinaryWeight>();
                        const float* b_float_data = B.typed_data<float>();
                        int64_t n_elements = B.num_elements();
                        for (int64_t i = 0; i < n_elements; ++i) {
                            bw_data[i] = (b_float_data[i] >= 0) ? BINARY_ONE : BINARY_ZERO;
                        }
                        bin_it = cached_binary_tensors_.emplace(B_name, std::move(binary_B)).first;
                        binary_B_ptr = &bin_it->second;
                    }

                    // Check for pre-packed BinaryMatrix cache
                    auto packed_it = cached_packed_b_.find(B_name);
                    if (packed_it == cached_packed_b_.end()) {
                        // First time: pack binary weights into GeMM BinaryMatrix
                        const Tensor& b_bin = *binary_B_ptr;
                        uint32_t K_w = static_cast<uint32_t>(b_bin.shape().dims[0]);
                        uint32_t N_w = static_cast<uint32_t>(b_bin.shape().dims[1]);
                        uint32_t k_padded = ((K_w + 127) / 128) * 128;
                        uint32_t n_padded = ((N_w + 7) / 8) * 8;

                        std::vector<int8_t> b_int8(static_cast<size_t>(k_padded) * n_padded, 1);
                        const BinaryWeight* b_data = b_bin.typed_data<BinaryWeight>();
                        for (uint32_t i = 0; i < K_w; ++i) {
                            for (uint32_t j = 0; j < N_w; ++j) {
                                b_int8[i * n_padded + j] = (b_data[i * N_w + j] == BINARY_ONE) ? +1 : -1;
                            }
                        }

                        CachedPackedBinaryMatrix entry;
                        entry.matrix = BinaryMatrix::pack(
                            std::span<const int8_t>(b_int8.data(), b_int8.size()),
                            k_padded, n_padded);
                        entry.n_orig = N_w;
                        entry.tiling = tiling_config::get();
                        packed_it = cached_packed_b_.emplace(B_name, std::move(entry)).first;
                    }

                    result = qlinear_matmul_binary_blocked_packed(
                        A, packed_it->second.matrix, packed_it->second.n_orig,
                        1.0f, packed_it->second.tiling);
                } else {
                    TBN_LOG_DEBUG("Gemm: quantizing weights on-the-fly");
                    Tensor binary_B = quantize_to_binary(B);
                    result = qlinear_matmul_binary(A, binary_B, 1.0f);
                }

                // Apply alpha and beta
                if (alpha != 1.0f || (C && beta != 0.0f)) {
                    float* result_data = result.typed_data<float>();
                    int64_t M = result.shape().dims[0];
                    int64_t N = result.shape().dims[1];

                    for (int64_t i = 0; i < M * N; ++i) {
                        result_data[i] *= alpha;
                    }

                    if (C && beta != 0.0f) {
                        const float* C_data = C->typed_data<float>();
                        bool C_is_1d = C->shape().dims.size() == 1;

                        if (C_is_1d) {
                            for (int64_t i = 0; i < M; ++i) {
                                for (int64_t j = 0; j < N; ++j) {
                                    result_data[i * N + j] += beta * C_data[j];
                                }
                            }
                        } else {
                            for (int64_t i = 0; i < M * N; ++i) {
                                result_data[i] += beta * C_data[i];
                            }
                        }
                    }
                }
            } else {
                // Use standard float GEMM
                result = gemm(A, B, C, alpha, beta, transA, transB);
            }

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
                TBN_LOG_DEBUG("MatMul: A=" + shape_to_string(A.shape()) + " B=" + shape_to_string(B.shape()));
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

            // 4D + 3D broadcasting (Conv bias case: [N, C, H, W] + [C, 1, 1])
            if (A.shape().dims.size() == 4 && B.shape().dims.size() == 3 &&
                B.shape().dims[1] == 1 && B.shape().dims[2] == 1) {
                int64_t N = A.shape().dims[0];
                int64_t C = A.shape().dims[1];
                int64_t H = A.shape().dims[2];
                int64_t W = A.shape().dims[3];
                TBN_CHECK(B.shape().dims[0] == C, InvalidShapeError,
                          "Broadcasting: B channels must match A");

                Tensor output(A.shape(), A.dtype());
                const float* A_data = A.typed_data<float>();
                const float* B_data = B.typed_data<float>();
                float* output_data = output.typed_data<float>();

                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        float bias = B_data[c];
                        for (int64_t h = 0; h < H; ++h) {
                            for (int64_t w = 0; w < W; ++w) {
                                int64_t idx = ((n * C + c) * H + h) * W + w;
                                output_data[idx] = A_data[idx] + bias;
                            }
                        }
                    }
                }
                intermediate_tensors_[node.outputs[0]] = output;
                return;
            }

            // 4D + 1D broadcasting (Conv bias case: [N, C, H, W] + [C])
            if (A.shape().dims.size() == 4 && B.shape().dims.size() == 1) {
                int64_t N = A.shape().dims[0];
                int64_t C = A.shape().dims[1];
                int64_t H = A.shape().dims[2];
                int64_t W = A.shape().dims[3];
                TBN_CHECK(B.shape().dims[0] == C, InvalidShapeError,
                          "Broadcasting: B size must match A channels");

                Tensor output(A.shape(), A.dtype());
                const float* A_data = A.typed_data<float>();
                const float* B_data = B.typed_data<float>();
                float* output_data = output.typed_data<float>();

                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        float bias = B_data[c];
                        for (int64_t h = 0; h < H; ++h) {
                            for (int64_t w = 0; w < W; ++w) {
                                int64_t idx = ((n * C + c) * H + h) * W + w;
                                output_data[idx] = A_data[idx] + bias;
                            }
                        }
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
            size_t element_size = 0;
            switch (data.dtype()) {
                case DataType::FLOAT32: element_size = sizeof(float); break;
                case DataType::INT32: element_size = sizeof(int32_t); break;
                case DataType::INT64: element_size = sizeof(int64_t); break;
                case DataType::INT8: element_size = sizeof(int8_t); break;
                case DataType::UINT8: element_size = sizeof(uint8_t); break;
                default: element_size = sizeof(float); // fallback
            }
            std::memcpy(output.data(), data.data(), data.num_elements() * element_size);
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

            // Check if this is the second ReLU (after conv2) by looking at input name
            // In TBN, we apply ternary activation only after conv2, not conv1
            bool apply_ternary = use_quantization_ &&
                                 (input_name.find("conv2d_1") != std::string::npos ||
                                  input_name.find("conv2_out") != std::string::npos);

            if (apply_ternary) {
                // Ternary activation quantization for TBN
                // Quantize raw conv output to {-1, 0, +1}
                const float threshold = 0.5f;
                for (int64_t i = 0; i < input.num_elements(); ++i) {
                    float val = input_data[i];
                    // Ternary quantization: {-1, 0, +1}
                    if (val > threshold) {
                        output_data[i] = 1.0f;
                    } else if (val < -threshold) {
                        output_data[i] = -1.0f;
                    } else {
                        output_data[i] = 0.0f;
                    }
                }
                TBN_LOG_INFO("ReLU replaced with ternary activation for input: " + input_name);
            } else {
                // Standard ReLU
                for (int64_t i = 0; i < input.num_elements(); ++i) {
                    output_data[i] = std::max(0.0f, input_data[i]);
                }
            }

            intermediate_tensors_[output_name] = output;
        }

        void execute_greater(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() == 2, InvalidArgumentError, "Greater expects 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Greater expects 1 output");

            const auto& lhs_name = node.inputs[0];
            const auto& rhs_name = node.inputs[1];
            const auto& output_name = node.outputs[0];

            auto it_lhs = intermediate_tensors_.find(lhs_name);
            auto it_rhs = intermediate_tensors_.find(rhs_name);

            // RHS might be a constant (scalar threshold)
            float rhs_scalar = 0.0f;
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;

            if (it_lhs != intermediate_tensors_.end()) {
                lhs = &it_lhs->second;
            }
            if (it_rhs != intermediate_tensors_.end()) {
                rhs = &it_rhs->second;
            } else {
                // Check if it's an attribute or initializer
                auto attr_it = node.attributes.find("value");
                if (attr_it != node.attributes.end()) {
                    rhs_scalar = attr_it->second.as_float();
                }
            }

            TBN_CHECK(lhs != nullptr, RuntimeError, "Greater input not found");

            Tensor output(lhs->shape(), DataType::FLOAT32);
            const float* lhs_data = lhs->typed_data<float>();
            float* output_data = output.typed_data<float>();

            if (rhs) {
                const float* rhs_data = rhs->typed_data<float>();
                for (int64_t i = 0; i < lhs->num_elements(); ++i) {
                    output_data[i] = (lhs_data[i] > rhs_data[i % rhs->num_elements()]) ? 1.0f : 0.0f;
                }
            } else {
                for (int64_t i = 0; i < lhs->num_elements(); ++i) {
                    output_data[i] = (lhs_data[i] > rhs_scalar) ? 1.0f : 0.0f;
                }
            }

            intermediate_tensors_[output_name] = output;
        }

        void execute_less(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() == 2, InvalidArgumentError, "Less expects 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Less expects 1 output");

            const auto& lhs_name = node.inputs[0];
            const auto& rhs_name = node.inputs[1];
            const auto& output_name = node.outputs[0];

            auto it_lhs = intermediate_tensors_.find(lhs_name);
            auto it_rhs = intermediate_tensors_.find(rhs_name);

            float rhs_scalar = 0.0f;
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;

            if (it_lhs != intermediate_tensors_.end()) {
                lhs = &it_lhs->second;
            }
            if (it_rhs != intermediate_tensors_.end()) {
                rhs = &it_rhs->second;
            } else {
                auto attr_it = node.attributes.find("value");
                if (attr_it != node.attributes.end()) {
                    rhs_scalar = attr_it->second.as_float();
                }
            }

            TBN_CHECK(lhs != nullptr, RuntimeError, "Less input not found");

            Tensor output(lhs->shape(), DataType::FLOAT32);
            const float* lhs_data = lhs->typed_data<float>();
            float* output_data = output.typed_data<float>();

            if (rhs) {
                const float* rhs_data = rhs->typed_data<float>();
                for (int64_t i = 0; i < lhs->num_elements(); ++i) {
                    output_data[i] = (lhs_data[i] < rhs_data[i % rhs->num_elements()]) ? 1.0f : 0.0f;
                }
            } else {
                for (int64_t i = 0; i < lhs->num_elements(); ++i) {
                    output_data[i] = (lhs_data[i] < rhs_scalar) ? 1.0f : 0.0f;
                }
            }

            intermediate_tensors_[output_name] = output;
        }

        void execute_cast(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() == 1, InvalidArgumentError, "Cast expects 1 input");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Cast expects 1 output");

            const auto& input_name = node.inputs[0];
            const auto& output_name = node.outputs[0];

            auto it = intermediate_tensors_.find(input_name);
            TBN_CHECK(it != intermediate_tensors_.end(), RuntimeError, "Cast input not found");

            const Tensor& input = it->second;

            // For TBN, we just pass through - we treat everything as float32
            // The cast is typically from bool/int to float for ternary activation
            intermediate_tensors_[output_name] = input;
        }

        void execute_sub(const ModelNode& node) {
            TBN_CHECK(node.inputs.size() == 2, InvalidArgumentError, "Sub expects 2 inputs");
            TBN_CHECK(node.outputs.size() == 1, InvalidArgumentError, "Sub expects 1 output");

            const auto& lhs_name = node.inputs[0];
            const auto& rhs_name = node.inputs[1];
            const auto& output_name = node.outputs[0];

            auto it_lhs = intermediate_tensors_.find(lhs_name);
            auto it_rhs = intermediate_tensors_.find(rhs_name);
            TBN_CHECK(it_lhs != intermediate_tensors_.end(), RuntimeError, "Sub lhs not found");
            TBN_CHECK(it_rhs != intermediate_tensors_.end(), RuntimeError, "Sub rhs not found");

            const Tensor& lhs = it_lhs->second;
            const Tensor& rhs = it_rhs->second;
            Tensor output(lhs.shape(), DataType::FLOAT32);

            const float* lhs_data = lhs.typed_data<float>();
            const float* rhs_data = rhs.typed_data<float>();
            float* output_data = output.typed_data<float>();

            for (int64_t i = 0; i < lhs.num_elements(); ++i) {
                output_data[i] = lhs_data[i] - rhs_data[i % rhs.num_elements()];
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