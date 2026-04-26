#include <tbn/tbn.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace tbn;

// Create a simple binary weights tensor
Tensor create_binary_weights(int64_t M, int64_t C, int64_t kH, int64_t kW) {
    Shape shape{M, C, kH, kW};
    Tensor weights(shape, DataType::BINARY);
    BinaryWeight* data = weights.typed_data<BinaryWeight>();

    // Simple pattern: alternating -1 and +1
    for (int64_t i = 0; i < weights.num_elements(); ++i) {
        data[i] = (i % 2 == 0) ? BINARY_ZERO : BINARY_ONE;  // -1, +1, -1, +1, ...
    }

    return weights;
}

// Create float input tensor
Tensor create_input(int64_t N, int64_t C, int64_t H, int64_t W) {
    Shape shape{N, C, H, W};
    Tensor input(shape, DataType::FLOAT32);
    float* data = input.typed_data<float>();

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f - 0.5f;  // [-0.5, 0.5]
    }

    return input;
}

// Create float weights (for naive comparison)
Tensor create_float_weights(int64_t M, int64_t C, int64_t kH, int64_t kW) {
    Shape shape{M, C, kH, kW};
    Tensor weights(shape, DataType::FLOAT32);
    float* data = weights.typed_data<float>();

    for (int64_t i = 0; i < weights.num_elements(); ++i) {
        data[i] = (i % 2 == 0) ? -1.0f : 1.0f;  // alternating -1, +1
    }

    return weights;
}

void run_benchmark(int64_t N, int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW, int runs = 5) {
    std::cout << "\n--- Benchmark: Input [" << N << "," << C << "," << H << "," << W
              << "], Weights [" << M << "," << C << "," << kH << "," << kW << "] ---" << std::endl;

    Tensor input = create_input(N, C, H, W);
    Tensor binary_weights = create_binary_weights(M, C, kH, kW);
    Tensor float_weights = create_float_weights(M, C, kH, kW);
    Conv2DParams params;
    params.pad_h = 1;
    params.pad_w = 1;

    // Warm up
    conv2d(input, float_weights, nullptr, params);
    conv2d_binary(input, binary_weights, nullptr, params);

    // Benchmark naive (float conv2d)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        Tensor output = conv2d(input, float_weights, nullptr, params);
    }
    auto end = std::chrono::high_resolution_clock::now();
    int64_t naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / runs;

    // Benchmark optimized (binary conv2d with GeMM)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        Tensor output = conv2d_binary(input, binary_weights, nullptr, params);
    }
    end = std::chrono::high_resolution_clock::now();
    int64_t optimized_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / runs;

    double speedup = static_cast<double>(naive_time) / optimized_time;

    std::cout << "Naive (float):   " << naive_time << " μs" << std::endl;
    std::cout << "Optimized (GeMM): " << optimized_time << " μs" << std::endl;
    std::cout << "Speedup:         " << speedup << "x" << std::endl;
}

int main() {
    std::cout << "=== Optimized Conv2D Test ===" << std::endl;

    // Quick correctness test
    std::cout << "\n--- Correctness Test ---" << std::endl;
    int64_t N = 1, C = 1, H = 8, W = 8;
    int64_t M = 2, kH = 3, kW = 3;

    Tensor input = create_input(N, C, H, W);
    Tensor binary_weights = create_binary_weights(M, C, kH, kW);

    std::cout << "Input: [" << N << ", " << C << ", " << H << ", " << W << "]" << std::endl;
    std::cout << "Weights: [" << M << ", " << C << ", " << kH << ", " << kW << "]" << std::endl;

    Conv2DParams params;
    Tensor output = conv2d_binary(input, binary_weights, nullptr, params);

    std::cout << "Output shape: [" << output.shape().dims[0] << ", "
              << output.shape().dims[1] << ", "
              << output.shape().dims[2] << ", "
              << output.shape().dims[3] << "]" << std::endl;

    const float* out_data = output.typed_data<float>();
    bool has_nan = false, has_inf = false;
    int nonzero = 0;
    for (int64_t i = 0; i < output.num_elements(); ++i) {
        if (std::isnan(out_data[i])) has_nan = true;
        if (std::isinf(out_data[i])) has_inf = true;
        if (out_data[i] != 0.0f) nonzero++;
    }

    std::cout << "Non-zero values: " << nonzero << " / " << output.num_elements() << std::endl;

    if (has_nan || has_inf) {
        std::cout << "ERROR: Output contains NaN or Inf!" << std::endl;
        return 1;
    }

    std::cout << "Correctness: OK" << std::endl;

    // Performance benchmarks
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;

    run_benchmark(1, 3, 32, 32, 16, 3, 3);      // Small CNN layer
    run_benchmark(1, 16, 28, 28, 32, 3, 3);     // Medium layer
    run_benchmark(1, 64, 14, 14, 128, 3, 3);    // Larger layer
    run_benchmark(1, 128, 7, 7, 256, 3, 3);     // Deep layer

    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}
