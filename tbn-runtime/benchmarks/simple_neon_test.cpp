#include <tbn/tbn.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace tbn;
using namespace std::chrono;

int main() {
    std::cout << "=== Float x Binary GEMM Test (using GeMM/05-final) ===" << std::endl;

    // Test with GeMM-optimal size (multiple of 128)
    int m = 128, n = 128;

    // Create test tensors
    Tensor A = Tensor({m, n}, DataType::FLOAT32);
    Tensor B_binary = Tensor({n, n}, DataType::BINARY);

    // Fill with test data
    float* A_data = A.typed_data<float>();
    BinaryWeight* binary_data = B_binary.typed_data<BinaryWeight>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> float_dis(-1.0f, 1.0f);

    for (int i = 0; i < A.num_elements(); ++i) {
        A_data[i] = float_dis(gen);
    }

    for (int i = 0; i < B_binary.num_elements(); ++i) {
        binary_data[i] = static_cast<BinaryWeight>(i % 2); // alternating 0, 1
    }

    // Warmup
    for (int i = 0; i < 3; ++i) {
        qlinear_matmul_binary(A, B_binary, 1.0f, TilingParams::default_128x128());
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        qlinear_matmul_binary(A, B_binary, 1.0f, TilingParams::default_128x128());
    }
    auto end = high_resolution_clock::now();
    auto time_128 = duration_cast<microseconds>(end - start).count() / 10.0;

    std::cout << "Matrix size: " << m << "x" << n << std::endl;
    std::cout << "Float x Binary MatMul: " << time_128 << " μs" << std::endl;

    // Test larger matrix
    m = 256; n = 256;
    Tensor A2 = Tensor({m, n}, DataType::FLOAT32);
    Tensor B2 = Tensor({n, n}, DataType::BINARY);

    float* A2_data = A2.typed_data<float>();
    BinaryWeight* B2_data = B2.typed_data<BinaryWeight>();

    for (int i = 0; i < A2.num_elements(); ++i) {
        A2_data[i] = float_dis(gen);
    }
    for (int i = 0; i < B2.num_elements(); ++i) {
        B2_data[i] = static_cast<BinaryWeight>(i % 2);
    }

    start = high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        qlinear_matmul_binary(A2, B2, 1.0f, TilingParams::default_128x128());
    }
    end = high_resolution_clock::now();
    auto time_256 = duration_cast<microseconds>(end - start).count() / 10.0;

    std::cout << "\nMatrix size: " << m << "x" << n << std::endl;
    std::cout << "Float x Binary MatMul: " << time_256 << " μs" << std::endl;

    // Test 512x512
    m = 512; n = 512;
    Tensor A3 = Tensor({m, n}, DataType::FLOAT32);
    Tensor B3 = Tensor({n, n}, DataType::BINARY);

    float* A3_data = A3.typed_data<float>();
    BinaryWeight* B3_data = B3.typed_data<BinaryWeight>();

    for (int i = 0; i < A3.num_elements(); ++i) {
        A3_data[i] = float_dis(gen);
    }
    for (int i = 0; i < B3.num_elements(); ++i) {
        B3_data[i] = static_cast<BinaryWeight>(i % 2);
    }

    start = high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) {
        qlinear_matmul_binary(A3, B3, 1.0f, TilingParams::default_128x128());
    }
    end = high_resolution_clock::now();
    auto time_512 = duration_cast<microseconds>(end - start).count() / 5.0;

    std::cout << "\nMatrix size: " << m << "x" << n << std::endl;
    std::cout << "Float x Binary MatMul: " << time_512 << " μs" << std::endl;

    // Calculate GFLOPS
    auto calc_gflops = [](int64_t m, int64_t n, int64_t k, double time_us) {
        double ops = 2.0 * m * n * k;  // multiply-add = 2 ops
        double time_s = time_us / 1e6;
        return ops / time_s / 1e9;
    };

    std::cout << "\n--- Performance ---" << std::endl;
    std::cout << "128x128: " << calc_gflops(128, 128, 128, time_128) << " GFLOPS" << std::endl;
    std::cout << "256x256: " << calc_gflops(256, 256, 256, time_256) << " GFLOPS" << std::endl;
    std::cout << "512x512: " << calc_gflops(512, 512, 512, time_512) << " GFLOPS" << std::endl;

    return 0;
}
