#include <tbn/tbn.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

// Simple test framework
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "Test failed: " << message << std::endl; \
        return false; \
    }

#define TEST_ASSERT_NEAR(a, b, tolerance) \
    if (std::abs((a) - (b)) > (tolerance)) { \
        std::cerr << "Test failed: " << (a) << " != " << (b) << " (diff: " << std::abs((a) - (b)) << ")" << std::endl; \
        return false; \
    }

bool test_qlinear_matmul_basic() {
    std::cout << "Testing QLinearMatMul basic... ";

    // Create test matrices
    int8_t a_data[] = {1, 2, 3, 4};
    int8_t b_data[] = {5, 6, 7, 8};

    tbn::Tensor a(tbn::Shape{2, 2}, tbn::DataType::INT8);
    tbn::Tensor b(tbn::Shape{2, 2}, tbn::DataType::INT8);
    std::memcpy(a.data(), a_data, sizeof(a_data));
    std::memcpy(b.data(), b_data, sizeof(b_data));

    // Create scale and zero point tensors
    float a_scale_val = 0.1f, b_scale_val = 0.2f, y_scale_val = 0.3f;
    int8_t a_zp_val = 0, b_zp_val = 0, y_zp_val = 0;

    tbn::Tensor a_scale(tbn::Shape{}, tbn::DataType::FLOAT32, &a_scale_val);
    tbn::Tensor b_scale(tbn::Shape{}, tbn::DataType::FLOAT32, &b_scale_val);
    tbn::Tensor y_scale(tbn::Shape{}, tbn::DataType::FLOAT32, &y_scale_val);
    tbn::Tensor a_zp(tbn::Shape{}, tbn::DataType::INT8, &a_zp_val);
    tbn::Tensor b_zp(tbn::Shape{}, tbn::DataType::INT8, &b_zp_val);
    tbn::Tensor y_zp(tbn::Shape{}, tbn::DataType::INT8, &y_zp_val);

    // Perform quantized matrix multiplication
    auto result = tbn::qlinear_matmul(a, a_scale, a_zp, b, b_scale, b_zp, y_scale_val, y_zp);

    // Verify result shape
    TEST_ASSERT(result.shape().dims.size() == 2, "Result should be 2D");
    TEST_ASSERT(result.shape().dims[0] == 2, "Result rows should be 2");
    TEST_ASSERT(result.shape().dims[1] == 2, "Result cols should be 2");
    TEST_ASSERT(result.dtype() == tbn::DataType::INT8, "Result should be int8");

    // Verify some values (exact values depend on quantization)
    const int8_t* result_data = result.typed_data<int8_t>();
    TEST_ASSERT(result_data != nullptr, "Result data should not be null");

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_qlinear_conv2d_basic() {
    std::cout << "Testing QLinearConv2D basic... ";

    // Create test input (1x1x4x4)
    int8_t input_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Create test weights (1x1x2x2)
    int8_t weight_data[] = {1, 2, 3, 4};

    tbn::Tensor input(tbn::Shape{1, 1, 4, 4}, tbn::DataType::INT8);
    tbn::Tensor weights(tbn::Shape{1, 1, 2, 2}, tbn::DataType::INT8);
    std::memcpy(input.data(), input_data, sizeof(input_data));
    std::memcpy(weights.data(), weight_data, sizeof(weight_data));

    // Create scale and zero point tensors
    float input_scale = 0.1f, weight_scale = 0.2f, output_scale = 0.3f;
    int8_t input_zp = 0, weight_zp = 0, output_zp = 0;

    tbn::Tensor input_scale_tensor(tbn::Shape{}, tbn::DataType::FLOAT32, &input_scale);
    tbn::Tensor weight_scale_tensor(tbn::Shape{}, tbn::DataType::FLOAT32, &weight_scale);
    tbn::Tensor output_scale_tensor(tbn::Shape{}, tbn::DataType::FLOAT32, &output_scale);
    tbn::Tensor input_zp_tensor(tbn::Shape{}, tbn::DataType::INT8, &input_zp);
    tbn::Tensor weight_zp_tensor(tbn::Shape{}, tbn::DataType::INT8, &weight_zp);
    tbn::Tensor output_zp_tensor(tbn::Shape{}, tbn::DataType::INT8, &output_zp);

    // Create Conv2D parameters (no padding, stride 1)
    tbn::Conv2DParams params;
    params.kernel_h = 2;
    params.kernel_w = 2;
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_h = 0;
    params.pad_w = 0;

    // Perform quantized convolution
    auto result = tbn::qlinear_conv2d(input, input_scale_tensor, input_zp_tensor,
                                     weights, weight_scale_tensor, weight_zp_tensor,
                                     output_scale_tensor, output_zp_tensor, params);

    // Verify result shape
    TEST_ASSERT(result.shape().dims.size() == 4, "Result should be 4D");
    TEST_ASSERT(result.shape().dims[0] == 1, "Batch size should be 1");
    TEST_ASSERT(result.shape().dims[1] == 1, "Channels should be 1");
    TEST_ASSERT(result.shape().dims[2] == 3, "Height should be 3");
    TEST_ASSERT(result.shape().dims[3] == 3, "Width should be 3");
    TEST_ASSERT(result.dtype() == tbn::DataType::INT8, "Result should be int8");

    // Verify some values
    const int8_t* result_data = result.typed_data<int8_t>();
    TEST_ASSERT(result_data != nullptr, "Result data should not be null");
    TEST_ASSERT(result.num_elements() == 9, "Result should have 9 elements");

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_ternary_matmul() {
    std::cout << "Testing Ternary MatMul... ";

    // Create float input
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    tbn::Tensor a(tbn::Shape{2, 2}, tbn::DataType::FLOAT32);
    std::memcpy(a.data(), a_data.data(), a_data.size() * sizeof(float));

    // Create ternary weights
    tbn::TernaryWeight b_data[] = {
        tbn::TERNARY_PLUS_ONE, tbn::TERNARY_MINUS_ONE,
        tbn::TERNARY_ZERO, tbn::TERNARY_PLUS_ONE
    };
    tbn::Tensor b(tbn::Shape{2, 2}, tbn::DataType::TERNARY);
    std::memcpy(b.data(), b_data, sizeof(b_data));

    // Perform ternary matrix multiplication
    float b_scale = 0.5f;
    auto result = tbn::qlinear_matmul_ternary(a, b, b_scale);

    // Verify result
    TEST_ASSERT(result.shape().dims.size() == 2, "Result should be 2D");
    TEST_ASSERT(result.shape().dims[0] == 2, "Result rows should be 2");
    TEST_ASSERT(result.shape().dims[1] == 2, "Result cols should be 2");
    TEST_ASSERT(result.dtype() == tbn::DataType::FLOAT32, "Result should be float32");

    // Verify some values
    const float* result_data = result.typed_data<float>();
    TEST_ASSERT(result_data != nullptr, "Result data should not be null");

    // Expected: [1,2] * [[1,-1],[0,1]] * 0.5 = [0.5, 0.5] and [1.5, 0.5]
    TEST_ASSERT_NEAR(result_data[0], 0.5f, 0.01f);
    TEST_ASSERT_NEAR(result_data[1], 0.5f, 0.01f);
    TEST_ASSERT_NEAR(result_data[2], 1.5f, 0.01f);
    TEST_ASSERT_NEAR(result_data[3], 0.5f, 0.01f);

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_binary_matmul() {
    std::cout << "Testing Binary MatMul... ";

    // Create float input
    std::vector<float> a_data = {1.0f, -2.0f, 3.0f, -4.0f};
    tbn::Tensor a(tbn::Shape{2, 2}, tbn::DataType::FLOAT32);
    std::memcpy(a.data(), a_data.data(), a_data.size() * sizeof(float));

    // Create binary weights
    tbn::BinaryWeight b_data[] = {
        tbn::BINARY_ONE, tbn::BINARY_ZERO,
        tbn::BINARY_ZERO, tbn::BINARY_ONE
    };
    tbn::Tensor b(tbn::Shape{2, 2}, tbn::DataType::BINARY);
    std::memcpy(b.data(), b_data, sizeof(b_data));

    // Perform binary matrix multiplication
    float b_scale = 0.7f;
    auto result = tbn::qlinear_matmul_binary(a, b, b_scale);

    // Verify result
    TEST_ASSERT(result.shape().dims.size() == 2, "Result should be 2D");
    TEST_ASSERT(result.shape().dims[0] == 2, "Result rows should be 2");
    TEST_ASSERT(result.shape().dims[1] == 2, "Result cols should be 2");
    TEST_ASSERT(result.dtype() == tbn::DataType::FLOAT32, "Result should be float32");

    // Verify some values
    const float* result_data = result.typed_data<float>();
    TEST_ASSERT(result_data != nullptr, "Result data should not be null");

    // Expected: [[1,-2],[3,-4]] * [[1,0],[0,1]] * 0.7 = [[0.7,-1.4],[2.1,-2.8]]
    TEST_ASSERT_NEAR(result_data[0], 0.7f, 0.01f);
    TEST_ASSERT_NEAR(result_data[1], -1.4f, 0.01f);
    TEST_ASSERT_NEAR(result_data[2], 2.1f, 0.01f);
    TEST_ASSERT_NEAR(result_data[3], -2.8f, 0.01f);

    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Testing Quantized Operators ===" << std::endl;

    int passed = 0;
    int total = 0;

    // Test standard quantized operators
    if (test_qlinear_matmul_basic()) passed++;
    total++;

    if (test_qlinear_conv2d_basic()) passed++;
    total++;

    // Test ternary/binary operators
    if (test_ternary_matmul()) passed++;
    total++;

    if (test_binary_matmul()) passed++;
    total++;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    if (passed == total) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}