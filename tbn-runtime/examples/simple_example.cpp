#include <tbn/tbn.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Set up logging
        tbn::Logger::set_global_level(tbn::LogLevel::INFO);

        std::cout << "TBN Runtime Version: " << tbn::get_version() << std::endl;

        // Example 1: Create tensors
        std::cout << "\n=== Creating Tensors ===" << std::endl;

        // Create a float tensor
        std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto float_tensor = tbn::make_tensor(tbn::Shape{2, 3}, float_data);

        std::cout << "Float tensor shape: [";
        for (size_t i = 0; i < float_tensor.shape().dims.size(); ++i) {
            std::cout << float_tensor.shape().dims[i];
            if (i < float_tensor.shape().dims.size() - 1) std::cout << "x";
        }
        std::cout << "]" << std::endl;

        // Create ternary weights
        std::vector<tbn::TernaryWeight> ternary_data = {
            tbn::TERNARY_MINUS_ONE, tbn::TERNARY_ZERO, tbn::TERNARY_PLUS_ONE,
            tbn::TERNARY_PLUS_ONE, tbn::TERNARY_MINUS_ONE, tbn::TERNARY_ZERO
        };
        auto ternary_tensor = tbn::make_tensor(tbn::Shape{2, 3}, ternary_data);

        std::cout << "Ternary tensor created with " << ternary_tensor.num_elements() << " elements" << std::endl;

        // Example 2: Model loading (placeholder)
        std::cout << "\n=== Model Loading ===" << std::endl;

        try {
            auto model = tbn::load_model("example_model.onnx");
            std::cout << "Model loaded successfully" << std::endl;
        } catch (const tbn::NotImplementedError& e) {
            std::cout << "Model loading not yet implemented: " << e.what() << std::endl;
        }

        // Example 3: Create a simple model manually
        std::cout << "\n=== Creating Simple Model ===" << std::endl;

        auto model_graph = std::make_shared<tbn::ModelGraph>();

        // Define model inputs
        model_graph->inputs = {"input"};
        model_graph->value_info["input"] = tbn::Shape{1, 3, 224, 224};

        // Define model outputs
        model_graph->outputs = {"output"};
        model_graph->value_info["output"] = tbn::Shape{1, 1000};

        // Add a simple node
        tbn::ModelNode relu_node;
        relu_node.name = "relu1";
        relu_node.op_type = "Relu";
        relu_node.inputs = {"input"};
        relu_node.outputs = {"relu_out"};
        model_graph->nodes.push_back(relu_node);
        model_graph->value_info["relu_out"] = tbn::Shape{1, 3, 224, 224};

        auto simple_model = tbn::TBNModel(model_graph);
        simple_model.set_producer("example", "1.0");

        std::cout << "Simple model created with " << simple_model.graph().nodes.size() << " nodes" << std::endl;
        std::cout << "Model inputs: ";
        for (const auto& input : simple_model.inputs()) {
            std::cout << input << " ";
        }
        std::cout << std::endl;

        // Example 4: Inference session
        std::cout << "\n=== Inference Session ===" << std::endl;

        auto session = simple_model.create_session();

        // Create input tensor
        std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);
        auto input_tensor = tbn::Tensor(tbn::Shape{1, 3, 224, 224}, tbn::DataType::FLOAT32);
        std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

        // Set input and run
        session.set_input("input", input_tensor);
        session.run();

        // Get output
        auto output = session.get_output("output");
        std::cout << "Inference completed. Output shape: [";
        for (size_t i = 0; i < output.shape().dims.size(); ++i) {
            std::cout << output.shape().dims[i];
            if (i < output.shape().dims.size() - 1) std::cout << "x";
        }
        std::cout << "]" << std::endl;

        // Example 5: Quantization
        std::cout << "\n=== Quantization ===" << std::endl;

        // Create a ternary quantizer
        auto ternary_quantizer = std::make_unique<tbn::TernaryQuantizer>(-0.5f, 0.5f);

        std::cout << "Ternary quantizer created with thresholds: ["
                  << -0.5f << ", " << 0.5f << "]" << std::endl;

        // Quantize some weights (would need actual implementation)
        std::cout << "Quantization implementation pending..." << std::endl;

        // Example 6: Memory packing
        std::cout << "\n=== Memory Packing ===" << std::endl;

        // Create packed ternary weights
        tbn::TernaryPackedWeights packed_weights(tbn::Shape{64, 64});
        std::cout << "Packed ternary weights: " << packed_weights.size() << " elements in "
                  << packed_weights.packed_size() << " bytes" << std::endl;
        std::cout << "Compression ratio: " << (float)packed_weights.size() / (packed_weights.packed_size() * 4) << ":1" << std::endl;

        // Example 7: Error handling
        std::cout << "\n=== Error Handling ===" << std::endl;

        try {
            // This will throw an exception
            auto bad_tensor = tbn::Tensor(tbn::Shape{100}, tbn::DataType::FLOAT32);
            auto bad_output = bad_tensor.typed_data<int>(); // Wrong type
        } catch (const tbn::TBNError& e) {
            std::cout << "Caught expected error: " << e.what() << std::endl;
        }

        std::cout << "\n=== Example Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/**
 * @brief Example of using the TBN Runtime with OpenInference API compatibility
 *
 * This example demonstrates how to use the TBN Runtime API which is designed
 * to be compatible with the OpenInference API for easy integration.
 */

// OpenInference-compatible API wrapper
namespace openinference {

    using Model = tbn::TBNModel;
    using Tensor = tbn::Tensor;
    using Session = tbn::InferenceSession;

    inline std::shared_ptr<Model> load_model(const std::string& path) {
        return std::make_shared<Model>(tbn::load_model(path));
    }

    inline std::unique_ptr<Session> create_session(std::shared_ptr<Model> model) {
        return tbn::create_session(model);
    }

} // namespace openinference

/**
 * Example usage with OpenInference API:
 *
 * auto model = openinference::load_model("model.onnx");
 * auto session = openinference::create_session(model);
 *
 * session->set_input("input", input_tensor);
 * session->run();
 * auto output = session->get_output("output");
 */