#!/usr/bin/env python3
"""
Create test ONNX models for TBN Runtime testing.
Generates simple models with various operators and quantization.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def create_simple_model():
    """Create a simple Conv->Relu->Gemm model"""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

    # Conv weights and bias
    conv_weight = helper.make_tensor(
        'conv1.weight',
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten()
    )

    conv_bias = helper.make_tensor(
        'conv1.bias',
        TensorProto.FLOAT,
        [64],
        np.zeros(64, dtype=np.float32)
    )

    # FC weights and bias
    fc_weight = helper.make_tensor(
        'fc1.weight',
        TensorProto.FLOAT,
        [1000, 6272],  # 64*14*14 after pooling
        np.random.randn(1000, 6272).astype(np.float32).flatten() * 0.01
    )

    fc_bias = helper.make_tensor(
        'fc1.bias',
        TensorProto.FLOAT,
        [1000],
        np.zeros(1000, dtype=np.float32)
    )

    # Nodes
    nodes = [
        # Conv
        helper.make_node(
            'Conv',
            inputs=['input', 'conv1.weight', 'conv1.bias'],
            outputs=['conv_output'],
            name='conv1',
            kernel_shape=[7, 7],
            strides=[2, 2],
            pads=[3, 3, 3, 3]
        ),

        # Relu
        helper.make_node(
            'Relu',
            inputs=['conv_output'],
            outputs=['relu_output'],
            name='relu1'
        ),

        # GlobalAveragePool
        helper.make_node(
            'GlobalAveragePool',
            inputs=['relu_output'],
            outputs=['pool_output'],
            name='global_pool'
        ),

        # Flatten (implicit in Gemm)

        # Gemm (FC)
        helper.make_node(
            'Gemm',
            inputs=['pool_output', 'fc1.weight', 'fc1.bias'],
            outputs=['output'],
            name='fc1',
            transB=1  # Weight matrix is transposed
        )
    ]

    # Graph
    graph = helper.make_graph(
        nodes,
        'simple_model',
        [input_tensor],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])],
        [conv_weight, conv_bias, fc_weight, fc_bias]
    )

    # Model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    return model

def create_quantized_model():
    """Create a model with quantized operators"""

    # Input (already quantized)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.INT8, [1, 3, 224, 224])

    # Quantization parameters
    input_scale = helper.make_tensor('input_scale', TensorProto.FLOAT, [], [0.1])
    input_zero_point = helper.make_tensor('input_zero_point', TensorProto.INT8, [], [0])

    # Conv weights (int8 quantized)
    conv_weight = helper.make_tensor(
        'conv1.weight_quantized',
        TensorProto.INT8,
        [64, 3, 7, 7],
        np.random.randint(-128, 127, size=(64, 3, 7, 7), dtype=np.int8).flatten()
    )

    conv_weight_scale = helper.make_tensor('conv1.weight_scale', TensorProto.FLOAT, [], [0.05])
    conv_weight_zero_point = helper.make_tensor('conv1.weight_zero_point', TensorProto.INT8, [], [0])

    # Conv bias (float32)
    conv_bias = helper.make_tensor(
        'conv1.bias',
        TensorProto.FLOAT,
        [64],
        np.zeros(64, dtype=np.float32)
    )

    # Output quantization params
    output_scale = helper.make_tensor('output_scale', TensorProto.FLOAT, [], [0.2])
    output_zero_point = helper.make_tensor('output_zero_point', TensorProto.INT8, [], [0])

    # Nodes
    nodes = [
        # Dequantize input
        helper.make_node(
            'DequantizeLinear',
            inputs=['input', 'input_scale', 'input_zero_point'],
            outputs=['input_float'],
            name='dequant_input'
        ),

        # Dequantize weights
        helper.make_node(
            'DequantizeLinear',
            inputs=['conv1.weight_quantized', 'conv1.weight_scale', 'conv1.weight_zero_point'],
            outputs=['conv1.weight_float'],
            name='dequant_weight'
        ),

        # Conv with float weights
        helper.make_node(
            'Conv',
            inputs=['input_float', 'conv1.weight_float', 'conv1.bias'],
            outputs=['conv_float'],
            name='conv1',
            kernel_shape=[7, 7],
            strides=[2, 2],
            pads=[3, 3, 3, 3]
        ),

        # Quantize output
        helper.make_node(
            'QuantizeLinear',
            inputs=['conv_float', 'output_scale', 'output_zero_point'],
            outputs=['output'],
            name='quant_output'
        )
    ]

    # Graph
    graph = helper.make_graph(
        nodes,
        'quantized_model',
        [input_tensor],
        [helper.make_tensor_value_info('output', TensorProto.INT8, [1, 64, 112, 112])],
        [
            input_scale, input_zero_point,
            conv_weight, conv_weight_scale, conv_weight_zero_point,
            conv_bias,
            output_scale, output_zero_point
        ]
    )

    # Model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    return model

def create_tiny_model():
    """Create a very small model for quick testing"""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # FC weights and bias
    fc_weight = helper.make_tensor(
        'fc.weight',
        TensorProto.FLOAT,
        [2, 4],
        np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32).flatten()
    )

    fc_bias = helper.make_tensor(
        'fc.bias',
        TensorProto.FLOAT,
        [2],
        np.array([0.1, 0.2], dtype=np.float32)
    )

    # Nodes
    nodes = [
        helper.make_node(
            'Gemm',
            inputs=['input', 'fc.weight', 'fc.bias'],
            outputs=['output'],
            name='fc'
        )
    ]

    # Graph
    graph = helper.make_graph(
        nodes,
        'tiny_model',
        [input_tensor],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
        [fc_weight, fc_bias]
    )

    # Model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    return model

def main():
    """Generate test ONNX models"""

    print("Creating test ONNX models...")

    # Create models
    models = {
        'simple_model.onnx': create_simple_model(),
        'quantized_model.onnx': create_quantized_model(),
        'tiny_model.onnx': create_tiny_model()
    }

    # Save models
    for filename, model in models.items():
        onnx.save(model, filename)
        print(f"Saved {filename}")

        # Print model info
        print(f"  - Inputs: {[i.name for i in model.graph.input]}")
        print(f"  - Outputs: {[o.name for o in model.graph.output]}")
        print(f"  - Nodes: {len(model.graph.node)}")
        print(f"  - Initializers: {len(model.graph.initializer)}")

        # Check model
        try:
            onnx.checker.check_model(model)
            print(f"  - Model is valid ✓")
        except Exception as e:
            print(f"  - Model validation failed: {e}")

        print()

    print("Test models created successfully!")
    print("\nUsage:")
    print("  - simple_model.onnx: Standard Conv->Relu->Gemm model")
    print("  - quantized_model.onnx: Model with quantized operators")
    print("  - tiny_model.onnx: Small model for quick testing")

if __name__ == "__main__":
    main()