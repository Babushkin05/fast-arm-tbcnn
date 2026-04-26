#include <tbn/tbn.hpp>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    std::string model_path = "tests/models/simple/onnx/simple_mlp.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "=== ONNX Inference Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        // Load model
        auto model = tbn::load_model(model_path);
        std::cout << "Model loaded successfully!" << std::endl;

        // Create session
        auto session = model.create_session();

        // Get input shape
        const auto& input_name = model.inputs()[0];
        auto input_shape = model.get_input_shape(input_name);
        std::cout << "Input: " << input_name << " shape: [";
        for (size_t i = 0; i < input_shape.dims.size(); ++i) {
            std::cout << input_shape.dims[i];
            if (i < input_shape.dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Create input tensor with test data
        tbn::Tensor input_tensor(input_shape, tbn::DataType::FLOAT32);
        float* input_data = input_tensor.typed_data<float>();

        // Fill with test values
        for (int64_t i = 0; i < input_tensor.num_elements(); ++i) {
            input_data[i] = static_cast<float>(i) * 0.1f;
        }

        std::cout << "Input values: [";
        for (int64_t i = 0; i < std::min(input_tensor.num_elements(), int64_t(8)); ++i) {
            std::cout << input_data[i];
            if (i < input_tensor.num_elements() - 1) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;

        // Set input
        session.set_input(input_name, input_tensor);

        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        session.run();
        std::cout << "Inference completed!" << std::endl;

        // Get output
        const auto& output_name = model.outputs()[0];
        tbn::Tensor output_tensor = session.get_output(output_name);

        std::cout << "\nOutput: " << output_name << " shape: [";
        auto output_shape = output_tensor.shape();
        for (size_t i = 0; i < output_shape.dims.size(); ++i) {
            std::cout << output_shape.dims[i];
            if (i < output_shape.dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        const float* output_data = output_tensor.typed_data<float>();
        std::cout << "Output values: [";
        for (int64_t i = 0; i < std::min(output_tensor.num_elements(), int64_t(10)); ++i) {
            std::cout << output_data[i];
            if (i < output_tensor.num_elements() - 1) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;

        // Check for NaN/Inf
        bool has_nan = false;
        bool has_inf = false;
        for (int64_t i = 0; i < output_tensor.num_elements(); ++i) {
            if (std::isnan(output_data[i])) has_nan = true;
            if (std::isinf(output_data[i])) has_inf = true;
        }

        if (has_nan) {
            std::cout << "\nWARNING: Output contains NaN values!" << std::endl;
        }
        if (has_inf) {
            std::cout << "\nWARNING: Output contains Inf values!" << std::endl;
        }

        std::cout << "\n=== Test Complete ===" << std::endl;
        return (has_nan || has_inf) ? 1 : 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
