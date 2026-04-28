#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>

#include "tbn/runtime/model.hpp"
#include "tbn/runtime/tensor.hpp"
#include "tbn/onnx_integration/onnx_parser.hpp"

namespace py = pybind11;
using namespace tbn;

// Convert numpy array to Tensor
Tensor numpy_to_tensor(py::array_t<float> array) {
    py::buffer_info buf = array.request();

    Shape shape;
    for (ssize_t i = 0; i < buf.ndim; ++i) {
        shape.dims.push_back(buf.shape[i]);
    }

    Tensor tensor(shape, DataType::FLOAT32);
    std::memcpy(tensor.data(), buf.ptr, tensor.data_size());

    return tensor;
}

// Convert Tensor to numpy array
py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    std::vector<ssize_t> shape;
    for (auto dim : tensor.shape().dims) {
        shape.push_back(dim);
    }

    std::vector<ssize_t> strides;
    if (!shape.empty()) {
        strides.resize(shape.size());
        strides.back() = sizeof(float);
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // Make a copy of the data
    py::array_t<float> result(shape, strides);
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, tensor.data(), tensor.data_size());

    return result;
}

// Python wrapper for Model
class PyModel {
public:
    PyModel(const std::string& path,
            uint32_t mblk = 64, uint32_t nblk = 64, uint32_t kblk = 128,
            uint32_t mmk = 32, uint32_t nmk = 32) {
        TilingParams tiling{mblk, nblk, kblk, mmk, nmk};
        TBN_LOG_INFO("TBN: tiling mblk=" + std::to_string(mblk) +
                     " nblk=" + std::to_string(nblk) +
                     " kblk=" + std::to_string(kblk) +
                     " mmk=" + std::to_string(mmk) +
                     " nmk=" + std::to_string(nmk));
        model_ = load_onnx_model(path);
        session_ = std::make_unique<TBNModel::Session>(model_.create_session(tiling));
    }

    py::array_t<float> run(py::array_t<float> input, bool use_quantization = false) {
        // Convert input
        Tensor input_tensor = numpy_to_tensor(input);

        // Get input name from model
        const auto& input_names = model_.graph().inputs;
        if (input_names.empty()) {
            throw std::runtime_error("Model has no inputs");
        }

        // Reuse existing session
        session_->set_quantization(use_quantization);
        session_->set_input(input_names[0], input_tensor);
        session_->run();

        // Get output
        const auto& output_names = model_.graph().outputs;
        if (output_names.empty()) {
            throw std::runtime_error("Model has no outputs");
        }

        Tensor output = session_->get_output(output_names[0]);
        return tensor_to_numpy(output);
    }

    // Run with quantization enabled (convenience method)
    py::array_t<float> run_quantized(py::array_t<float> input) {
        return run(input, true);
    }

    std::vector<std::string> get_input_names() const {
        return model_.graph().inputs;
    }

    std::vector<std::string> get_output_names() const {
        return model_.graph().outputs;
    }

    Shape get_input_shape(const std::string& name) const {
        auto it = model_.graph().value_info.find(name);
        if (it == model_.graph().value_info.end()) {
            throw std::runtime_error("Input not found: " + name);
        }
        return it->second;
    }

private:
    TBNModel model_;
    std::unique_ptr<TBNModel::Session> session_;
};

PYBIND11_MODULE(_tbn, m) {
    m.doc() = "TBN Runtime - Optimized inference for ternary-binary neural networks";

    // Shape class
    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<std::vector<int64_t>>())
        .def_readwrite("dims", &Shape::dims)
        .def("__repr__", [](const Shape& s) {
            std::string str = "Shape([";
            for (size_t i = 0; i < s.dims.size(); ++i) {
                if (i > 0) str += ", ";
                str += std::to_string(s.dims[i]);
            }
            str += "])";
            return str;
        });

    // Model class
    py::class_<PyModel>(m, "Model")
        .def(py::init<const std::string&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("path"),
             py::arg("mblk") = 64,
             py::arg("nblk") = 64,
             py::arg("kblk") = 128,
             py::arg("mmk") = 32,
             py::arg("nmk") = 32,
             "Load an ONNX model with explicit tiling parameters.\n"
             "mblk/nblk/kblk: outer block sizes (L2 cache).\n"
             "mmk/nmk: microkernel sizes (L1 cache).\n"
             "Constraints: n %% 128 == 0, kblk %% 128 == 0, m %% mmk == 0, k %% nmk == 0.")
        .def("run", &PyModel::run, py::arg("input"), py::arg("use_quantization") = false)
        .def("run_quantized", &PyModel::run_quantized, py::arg("input"))
        .def("input_names", &PyModel::get_input_names)
        .def("output_names", &PyModel::get_output_names)
        .def("input_shape", &PyModel::get_input_shape, py::arg("name"));

    // Convenience function
    m.def("load_model", [](const std::string& path,
                            uint32_t mblk, uint32_t nblk, uint32_t kblk,
                            uint32_t mmk, uint32_t nmk) {
        return PyModel(path, mblk, nblk, kblk, mmk, nmk);
    }, py::arg("path"),
       py::arg("mblk") = 64,
       py::arg("nblk") = 64,
       py::arg("kblk") = 128,
       py::arg("mmk") = 32,
       py::arg("nmk") = 32,
       "Load an ONNX model with explicit tiling parameters.");
}
