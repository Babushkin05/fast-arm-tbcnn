#include <tbn/tbn.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // Default model path
    std::string model_path = "tests/models/simple/onnx/simple_matmul.onnx";

    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "=== ONNX Model Loading Test ===" << std::endl;
    std::cout << "Loading: " << model_path << std::endl;

    try {
        // Load model
        auto model = tbn::load_model(model_path);

        std::cout << "\nModel loaded successfully!" << std::endl;
        std::cout << "Producer: " << model.producer_name() << " " << model.producer_version() << std::endl;

        // Print inputs
        std::cout << "\nInputs:" << std::endl;
        for (const auto& input : model.inputs()) {
            auto shape = model.get_input_shape(input);
            std::cout << "  " << input << ": [";
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                std::cout << shape.dims[i];
                if (i < shape.dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Print outputs
        std::cout << "\nOutputs:" << std::endl;
        for (const auto& output : model.outputs()) {
            auto shape = model.get_output_shape(output);
            std::cout << "  " << output << ": [";
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                std::cout << shape.dims[i];
                if (i < shape.dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Print nodes
        std::cout << "\nNodes (" << model.graph().nodes.size() << "):" << std::endl;
        for (const auto& node : model.graph().nodes) {
            std::cout << "  " << node.name << " [" << node.op_type << "]" << std::endl;
            std::cout << "    inputs: ";
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                std::cout << node.inputs[i];
                if (i < node.inputs.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << "    outputs: ";
            for (size_t i = 0; i < node.outputs.size(); ++i) {
                std::cout << node.outputs[i];
                if (i < node.outputs.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;

            // Print attributes
            if (!node.attributes.empty()) {
                std::cout << "    attributes:" << std::endl;
                for (const auto& [name, attr] : node.attributes) {
                    std::cout << "      " << name << ": ";
                    if (attr.is_int()) {
                        std::cout << attr.as_int();
                    } else if (attr.is_float()) {
                        std::cout << attr.as_float();
                    } else if (attr.is_string()) {
                        std::cout << "\"" << attr.as_string() << "\"";
                    } else if (attr.is_ints()) {
                        const auto& ints = attr.as_ints();
                        std::cout << "[";
                        for (size_t i = 0; i < ints.size(); ++i) {
                            std::cout << ints[i];
                            if (i < ints.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                    } else if (attr.is_floats()) {
                        const auto& floats = attr.as_floats();
                        std::cout << "[";
                        for (size_t i = 0; i < floats.size(); ++i) {
                            std::cout << floats[i];
                            if (i < floats.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                    } else if (attr.is_tensor()) {
                        std::cout << "<tensor>";
                    } else {
                        std::cout << "<unknown>";
                    }
                    std::cout << std::endl;
                }
            }
        }

        // Print initializers (weights)
        std::cout << "\nInitializers (" << model.graph().initializers.size() << "):" << std::endl;
        for (const auto& [name, tensor] : model.graph().initializers) {
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < tensor.shape().dims.size(); ++i) {
                std::cout << tensor.shape().dims[i];
                if (i < tensor.shape().dims.size() - 1) std::cout << ", ";
            }
            std::cout << "] dtype=" << static_cast<int>(tensor.dtype()) << std::endl;
        }

        std::cout << "\n=== Test Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
