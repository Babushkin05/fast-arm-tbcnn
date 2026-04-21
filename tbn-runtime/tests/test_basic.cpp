#include <tbn/tbn.hpp>
#include <iostream>
#include <vector>
#include <cstring>

bool test_tensor_creation() {
    std::cout << "Testing tensor creation... ";

    // Create a simple tensor
    tbn::Shape shape{2, 3};
    auto tensor = tbn::Tensor(shape, tbn::DataType::FLOAT32);

    if (tensor.shape().dims != shape.dims) {
        std::cerr << "Shape mismatch" << std::endl;
        return false;
    }
    if (tensor.dtype() != tbn::DataType::FLOAT32) {
        std::cerr << "Data type mismatch" << std::endl;
        return false;
    }
    if (tensor.num_elements() != 6) {
        std::cerr << "Element count mismatch" << std::endl;
        return false;
    }
    if (tensor.data() == nullptr) {
        std::cerr << "Data pointer is null" << std::endl;
        return false;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_tensor_with_data() {
    std::cout << "Testing tensor with data... ";

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto tensor = tbn::make_tensor(tbn::Shape{2, 3}, data);

    if (tensor.num_elements() != 6) {
        std::cerr << "Element count mismatch" << std::endl;
        return false;
    }

    const float* tensor_data = tensor.typed_data<float>();
    for (int i = 0; i < 6; ++i) {
        if (tensor_data[i] != data[i]) {
            std::cerr << "Data mismatch at index " << i << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_quantized_tensor() {
    std::cout << "Testing quantized tensor... ";

    // Create ternary tensor
    std::vector<tbn::TernaryWeight> ternary_data = {
        tbn::TERNARY_MINUS_ONE, tbn::TERNARY_ZERO,
        tbn::TERNARY_PLUS_ONE, tbn::TERNARY_MINUS_ONE
    };

    tbn::Tensor tensor(tbn::Shape{2, 2}, tbn::DataType::TERNARY);
    std::memcpy(tensor.data(), ternary_data.data(), ternary_data.size() * sizeof(tbn::TernaryWeight));

    if (tensor.dtype() != tbn::DataType::TERNARY) {
        std::cerr << "Data type mismatch" << std::endl;
        return false;
    }

    const tbn::TernaryWeight* data = tensor.typed_data<tbn::TernaryWeight>();
    for (int i = 0; i < 4; ++i) {
        if (data[i] != ternary_data[i]) {
            std::cerr << "Data mismatch at index " << i << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Running Basic Tests ===" << std::endl;

    int passed = 0;
    int total = 0;

    if (test_tensor_creation()) passed++;
    total++;

    if (test_tensor_with_data()) passed++;
    total++;

    if (test_quantized_tensor()) passed++;
    total++;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    return (passed == total) ? 0 : 1;
}