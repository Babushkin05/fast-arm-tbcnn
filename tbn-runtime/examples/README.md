# TBN Runtime Examples

This directory contains example applications demonstrating how to use the TBN Runtime.

## Examples

### simple_example.cpp
A comprehensive example showing:
- Tensor creation and manipulation
- Model loading (placeholder implementation)
- Manual model construction
- Inference session usage
- Quantization examples
- Memory packing for ternary/binary weights
- Error handling

## Building Examples

From the build directory:
```bash
make examples
```

Or to build a specific example:
```bash
make simple_example
```

## Running Examples

```bash
./bin/simple_example
```

## OpenInference API Compatibility

The TBN Runtime is designed to be compatible with the OpenInference API. The example demonstrates both the native TBN API and a wrapper for OpenInference compatibility:

```cpp
// Native TBN API
auto model = tbn::load_model("model.onnx");
auto session = model.create_session();

// OpenInference-compatible API
auto model = openinference::load_model("model.onnx");
auto session = openinference::create_session(model);
```

## Future Examples

- **advanced_example.cpp**: Advanced usage patterns, custom operators, optimization
- **quantization_example.cpp**: Detailed quantization workflows
- **performance_example.cpp**: Performance tuning and benchmarking
- **mobile_example.cpp**: Mobile-specific optimizations
- **onnx_example.cpp**: Full ONNX model loading and inference

## API Documentation

For detailed API documentation, see the main header files in `include/tbn/`.