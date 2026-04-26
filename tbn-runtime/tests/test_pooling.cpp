#include <tbn/tbn.hpp>
#include <iostream>
#include <cmath>

using namespace tbn;

Tensor create_input(int64_t N, int64_t C, int64_t H, int64_t W) {
    Shape shape{N, C, H, W};
    Tensor input(shape, DataType::FLOAT32);
    float* data = input.typed_data<float>();

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        data[i] = static_cast<float>(i % 10);
    }

    return input;
}

int main() {
    std::cout << "=== Pooling Test ===" << std::endl;

    // Create input [1, 2, 4, 4]
    int64_t N = 1, C = 2, H = 4, W = 4;
    Tensor input = create_input(N, C, H, W);

    std::cout << "Input shape: [" << N << ", " << C << ", " << H << ", " << W << "]" << std::endl;

    // Test MaxPool2D
    std::cout << "\n--- MaxPool2D (2x2, stride 2) ---" << std::endl;
    Pool2DParams maxpool_params(2, 2, 2, 2);
    Tensor maxpool_out = maxpool2d(input, maxpool_params);

    std::cout << "Output shape: [" << maxpool_out.shape().dims[0];
    for (size_t i = 1; i < maxpool_out.shape().dims.size(); ++i) {
        std::cout << ", " << maxpool_out.shape().dims[i];
    }
    std::cout << "]" << std::endl;

    const float* mp_data = maxpool_out.typed_data<float>();
    std::cout << "Values: ";
    for (int i = 0; i < std::min(maxpool_out.num_elements(), int64_t(8)); ++i) {
        std::cout << mp_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Test AvgPool2D
    std::cout << "\n--- AvgPool2D (2x2, stride 2) ---" << std::endl;
    Tensor avgpool_out = avgpool2d(input, maxpool_params);

    std::cout << "Output shape: [" << avgpool_out.shape().dims[0];
    for (size_t i = 1; i < avgpool_out.shape().dims.size(); ++i) {
        std::cout << ", " << avgpool_out.shape().dims[i];
    }
    std::cout << "]" << std::endl;

    const float* ap_data = avgpool_out.typed_data<float>();
    std::cout << "Values: ";
    for (int i = 0; i < std::min(avgpool_out.num_elements(), int64_t(8)); ++i) {
        std::cout << ap_data[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Test GlobalMaxPool2D
    std::cout << "\n--- GlobalMaxPool2D ---" << std::endl;
    Tensor global_max = global_maxpool2d(input);
    std::cout << "Output shape: [" << global_max.shape().dims[0];
    for (size_t i = 1; i < global_max.shape().dims.size(); ++i) {
        std::cout << ", " << global_max.shape().dims[i];
    }
    std::cout << "]" << std::endl;

    // Test GlobalAvgPool2D
    std::cout << "\n--- GlobalAvgPool2D ---" << std::endl;
    Tensor global_avg = global_avgpool2d(input);
    std::cout << "Output shape: [" << global_avg.shape().dims[0];
    for (size_t i = 1; i < global_avg.shape().dims.size(); ++i) {
        std::cout << ", " << global_avg.shape().dims[i];
    }
    std::cout << "]" << std::endl;

    // Verify values
    bool ok = true;
    // MaxPool: max of [0,1,4,5] = 5, max of [2,3,6,7] = 7, etc.
    if (mp_data[0] != 5.0f || mp_data[1] != 7.0f) {
        std::cout << "ERROR: MaxPool values incorrect!" << std::endl;
        ok = false;
    }

    // AvgPool: avg of [0,1,4,5] = 2.5, avg of [2,3,6,7] = 4.5
    if (std::abs(ap_data[0] - 2.5f) > 0.001f || std::abs(ap_data[1] - 4.5f) > 0.001f) {
        std::cout << "ERROR: AvgPool values incorrect!" << std::endl;
        ok = false;
    }

    std::cout << "\n=== Test " << (ok ? "PASSED" : "FAILED") << " ===" << std::endl;
    return ok ? 0 : 1;
}
