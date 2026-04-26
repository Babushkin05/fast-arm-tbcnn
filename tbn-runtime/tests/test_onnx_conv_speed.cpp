#include <tbn/tbn.hpp>
#include <iostream>
#include <chrono>

using namespace tbn;

int main(int argc, char* argv[]) {
    std::string model_path = "tests/models/simple/onnx/simple_conv.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "=== ONNX Conv Speed Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        auto model = tbn::load_model(model_path);
        auto session = model.create_session();

        const auto& input_name = model.inputs()[0];
        auto input_shape = model.get_input_shape(input_name);

        // Create input tensor
        tbn::Tensor input_tensor(input_shape, tbn::DataType::FLOAT32);
        float* input_data = input_tensor.typed_data<float>();
        for (int64_t i = 0; i < input_tensor.num_elements(); ++i) {
            input_data[i] = static_cast<float>(i % 100) / 100.0f - 0.5f;
        }

        session.set_input(input_name, input_tensor);

        // Warm up
        session.run();

        // Benchmark
        int runs = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) {
            session.run();
        }
        auto end = std::chrono::high_resolution_clock::now();

        int64_t total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        int64_t avg_us = total_us / runs;

        std::cout << "Runs: " << runs << std::endl;
        std::cout << "Total time: " << total_us << " μs" << std::endl;
        std::cout << "Average per run: " << avg_us << " μs" << std::endl;

        // Get output
        const auto& output_name = model.outputs()[0];
        tbn::Tensor output_tensor = session.get_output(output_name);
        std::cout << "Output shape: [" << output_tensor.shape().dims[0];
        for (size_t i = 1; i < output_tensor.shape().dims.size(); ++i) {
            std::cout << ", " << output_tensor.shape().dims[i];
        }
        std::cout << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
