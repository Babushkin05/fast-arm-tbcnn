#include <tbn/tbn.hpp>
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    try {
        // Set up logging
        tbn::Logger::set_global_level(tbn::LogLevel::INFO);

        std::cout << "=== Testing ONNX Model Loading ===" << std::endl;

        // Test 1: Load simple model
        std::cout << "\n1. Loading simple model..." << std::endl;
        auto model1 = tbn::load_model("simple_model.onnx");

        std::cout << "   Model loaded successfully!" << std::endl;
        std::cout << "   Producer: " << model1.producer_name() << " " << model1.producer_version() << std::endl;
        std::cout << "   Inputs: " << model1.inputs().size() << std::endl;
        std::cout << "   Outputs: " << model1.outputs().size() << std::endl;
        std::cout << "   Nodes: " << model1.graph().nodes.size() << std::endl;

        // Test 2: Load quantized model
        std::cout << "\n2. Loading quantized model..." << std::endl;
        auto model2 = tbn::load_model("quantized_model.onnx");

        std::cout << "   Quantized model loaded!" << std::endl;
        std::cout << "   Nodes: " << model2.graph().nodes.size() << std::endl;

        // Test 3: Load from buffer
        std::cout << "\n3. Loading from buffer..." << std::endl;
        std::vector<uint8_t> dummy_buffer(2048, 0);
        auto model3 = tbn::load_model_from_buffer(dummy_buffer.data(), dummy_buffer.size());

        std::cout << "   Model loaded from buffer!" << std::endl;

        // Test 4: Create inference session
        std::cout << "\n4. Creating inference session..." << std::endl;
        auto session = model1.create_session();

        std::cout << "   Session created!" << std::endl;

        // Test 5: Validate model
        std::cout << "\n5. Validating model..." << std::endl;
        model1.validate();
        std::cout << "   Model validation passed!" << std::endl;

        // Test 6: Check input/output shapes
        std::cout << "\n6. Checking shapes..." << std::endl;
        for (const auto& input : model1.inputs()) {
            auto shape = model1.get_input_shape(input);
            std::cout << "   Input '" << input << "' shape: [";
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                std::cout << shape.dims[i];
                if (i < shape.dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Test 7: Run inference with dummy data
        std::cout << "\n7. Running inference..." << std::endl;

        // Create input tensor
        std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);
        auto input_tensor = tbn::Tensor(tbn::Shape{1, 3, 224, 224}, tbn::DataType::FLOAT32);
        std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

        // Set input and run
        session.set_input("input", input_tensor);
        session.run();

        // Get output
        auto output = session.get_output("output");
        std::cout << "   Inference completed!" << std::endl;
        std::cout << "   Output shape: [";
        for (size_t i = 0; i < output.shape().dims.size(); ++i) {
            std::cout << output.shape().dims[i];
            if (i < output.shape().dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "\n=== All tests passed! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Compile and run:
// g++ -std=c++17 -I../include -L. -ltbn test_onnx_loading.cpp -o test_onnx_loading
// LD_LIBRARY_PATH=. ./test_onnx_loading

// Expected output:
// === Testing ONNX Model Loading ===
//
// 1. Loading simple model...
//    Model loaded successfully!
//    Producer: tbn-runtime-dummy 0.1.0
//    Inputs: 1
//    Outputs: 1
//    Nodes: 4
//
// 2. Loading quantized model...
//    Quantized model loaded!
//    Nodes: 1
//
// 3. Loading from buffer...
//    Model loaded from buffer!
//
// 4. Creating inference session...
//    Session created!
//
// 5. Validating model...
//    Model validation passed!
//
// 6. Checking shapes...
//    Input 'input' shape: [1, 3, 224, 224]
//
// 7. Running inference...
//    Inference completed!
//    Output shape: [1, 1000]
//
// === All tests passed! ==="}