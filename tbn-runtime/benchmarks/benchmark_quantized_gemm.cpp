#include <tbn/tbn.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <functional>

using namespace tbn;
using namespace std::chrono;

// Helper to create random float tensor
Tensor create_random_tensor(const Shape& shape, DataType dtype) {
    Tensor tensor(shape, dtype);

    if (dtype == DataType::FLOAT32) {
        float* data = tensor.typed_data<float>();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int64_t i = 0; i < tensor.num_elements(); ++i) {
            data[i] = dis(gen);
        }
    } else if (dtype == DataType::BINARY) {
        BinaryWeight* data = tensor.typed_data<BinaryWeight>();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);

        for (int64_t i = 0; i < tensor.num_elements(); ++i) {
            data[i] = static_cast<BinaryWeight>(dis(gen)); // 0 or 1
        }
    }

    return tensor;
}

// Benchmark function
double benchmark_gemm(
    const std::string& name,
    std::function<Tensor()> operation,
    int warmup_iterations = 5,
    int benchmark_iterations = 50
) {
    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        operation();
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < benchmark_iterations; ++i) {
        operation();
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    double avg_time = duration / static_cast<double>(benchmark_iterations);

    std::cout << std::setw(30) << std::left << name << ": "
              << std::fixed << std::setprecision(2) << avg_time << " μs" << std::endl;

    return avg_time;
}

int main() {
    std::cout << "=== Float x Binary GEMM Benchmark (using GeMM/05-final) ===" << std::endl;
    std::cout << "Quantizing float activations to ternary on-the-fly" << std::endl;
    std::cout << std::endl;

    // Test different matrix sizes (multiples of 128 for optimal GeMM)
    std::vector<std::pair<int, int>> sizes = {
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
    };

    for (const auto& [m, n] : sizes) {
        int k = n;  // K = N for square weight matrix
        std::cout << "Matrix size: A(" << m << "x" << k << ") @ B(" << k << "x" << n << ")" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        // Create tensors
        Tensor A = create_random_tensor({m, k}, DataType::FLOAT32);
        Tensor B_binary = create_random_tensor({k, n}, DataType::BINARY);

        // Benchmark binary matrix multiplication
        benchmark_gemm("Float x Binary MatMul", [&]() {
            return qlinear_matmul_binary(A, B_binary, 1.0f, TilingParams::default_128x128());
        });

        // Benchmark regular GEMM for comparison
        Tensor B_float = create_random_tensor({k, n}, DataType::FLOAT32);
        benchmark_gemm("Regular GEMM (float)", [&]() {
            return gemm(A, B_float);
        });

        // Calculate speedup
        std::cout << std::endl;
    }

    // Test non-optimal sizes (will be padded)
    std::cout << "Non-optimal sizes (will be padded to multiples of 128)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<std::pair<int, int>> nonoptimal_sizes = {
        {100, 100},
        {200, 200},
        {300, 300},
    };

    for (const auto& [m, n] : nonoptimal_sizes) {
        Tensor A = create_random_tensor({m, n}, DataType::FLOAT32);
        Tensor B_binary = create_random_tensor({n, n}, DataType::BINARY);

        benchmark_gemm("Binary MatMul (" + std::to_string(m) + "x" + std::to_string(n) + ")", [&]() {
            return qlinear_matmul_binary(A, B_binary, 1.0f, TilingParams::default_128x128());
        });
    }

    return 0;
}
