#!/usr/bin/env python3
"""
Generate simple ONNX models for testing TBN runtime.

Usage:
    python generate_models.py

Creates:
    - simple_matmul.onnx     : Single MatMul layer
    - simple_conv.onnx       : Single Conv layer
    - simple_mlp.onnx        : 2-layer MLP (MatMul + ReLU + MatMul)
    - binary_weights.onnx    : Model with binary-like weights
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import os


def create_simple_matmul():
    """Create a simple MatMul model: Y = X @ W + B"""
    # Input: (1, 4)
    # Weight: (4, 8)
    # Output: (1, 8)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8])

    # Weight matrix
    W = np.random.randn(4, 8).astype(np.float32)
    W_tensor = numpy_helper.from_array(W, name='W')

    # Bias
    B = np.zeros(8, dtype=np.float32)
    B_tensor = numpy_helper.from_array(B, name='B')

    # MatMul node
    matmul_node = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['Y_temp'])

    # Add node for bias
    add_node = helper.make_node('Add', inputs=['Y_temp', 'B'], outputs=['Y'])

    # Graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        'simple_matmul',
        [X],
        [Y],
        [W_tensor, B_tensor]
    )

    model = helper.make_model(graph)
    model.ir_version = 8

    # Set opset version
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13

    return model


def create_simple_conv():
    """Create a simple Conv2D model."""
    # Input: (1, 3, 32, 32)
    # Conv: 16 filters, 3x3 kernel
    # Output: (1, 16, 30, 30)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16, 30, 30])

    # Conv weights (16, 3, 3, 3)
    W = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
    W_tensor = numpy_helper.from_array(W, name='conv_weight')

    # Bias (16,)
    B = np.zeros(16, dtype=np.float32)
    B_tensor = numpy_helper.from_array(B, name='conv_bias')

    # Conv node
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'conv_weight', 'conv_bias'],
        outputs=['Y'],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )

    graph = helper.make_graph(
        [conv_node],
        'simple_conv',
        [X],
        [Y],
        [W_tensor, B_tensor]
    )

    model = helper.make_model(graph)
    model.ir_version = 8

    # Set opset version
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13

    return model


def create_simple_mlp():
    """Create a 2-layer MLP: Linear -> ReLU -> Linear."""
    # Input: (1, 16)
    # Hidden: (1, 32)
    # Output: (1, 10)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])

    # Layer 1 weights
    W1 = np.random.randn(16, 32).astype(np.float32) * 0.1
    W1_tensor = numpy_helper.from_array(W1, name='W1')
    B1 = np.zeros(32, dtype=np.float32)
    B1_tensor = numpy_helper.from_array(B1, name='B1')

    # Layer 2 weights
    W2 = np.random.randn(32, 10).astype(np.float32) * 0.1
    W2_tensor = numpy_helper.from_array(W2, name='W2')
    B2 = np.zeros(10, dtype=np.float32)
    B2_tensor = numpy_helper.from_array(B2, name='B2')

    # Nodes
    matmul1 = helper.make_node('MatMul', inputs=['X', 'W1'], outputs=['temp1'])
    add1 = helper.make_node('Add', inputs=['temp1', 'B1'], outputs=['temp2'])
    relu = helper.make_node('Relu', inputs=['temp2'], outputs=['temp3'])
    matmul2 = helper.make_node('MatMul', inputs=['temp3', 'W2'], outputs=['temp4'])
    add2 = helper.make_node('Add', inputs=['temp4', 'B2'], outputs=['Y'])

    graph = helper.make_graph(
        [matmul1, add1, relu, matmul2, add2],
        'simple_mlp',
        [X],
        [Y],
        [W1_tensor, B1_tensor, W2_tensor, B2_tensor]
    )

    model = helper.make_model(graph)
    model.ir_version = 8

    # Set opset version
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13

    return model


def create_binary_weights():
    """Create a model with binary-like weights (-1, +1)."""
    # Input: (1, 8)
    # Output: (1, 4)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    # Binary weights: -1 or +1 only
    W = np.sign(np.random.randn(8, 4)).astype(np.float32)
    W_tensor = numpy_helper.from_array(W, name='W')

    B = np.zeros(4, dtype=np.float32)
    B_tensor = numpy_helper.from_array(B, name='B')

    matmul = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['temp'])
    add = helper.make_node('Add', inputs=['temp', 'B'], outputs=['Y'])

    graph = helper.make_graph(
        [matmul, add],
        'binary_weights',
        [X],
        [Y],
        [W_tensor, B_tensor]
    )

    model = helper.make_model(graph)
    model.ir_version = 8

    # Set opset version
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13

    return model


def create_ternary_weights():
    """Create a model with ternary weights (-1, 0, +1)."""
    # Input: (1, 8)
    # Output: (1, 4)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    # Ternary weights: -1, 0, or +1
    W = np.random.choice([-1, 0, 1], size=(8, 4)).astype(np.float32)
    W_tensor = numpy_helper.from_array(W, name='W')

    B = np.zeros(4, dtype=np.float32)
    B_tensor = numpy_helper.from_array(B, name='B')

    matmul = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['temp'])
    add = helper.make_node('Add', inputs=['temp', 'B'], outputs=['Y'])

    graph = helper.make_graph(
        [matmul, add],
        'ternary_weights',
        [X],
        [Y],
        [W_tensor, B_tensor]
    )

    model = helper.make_model(graph)
    model.ir_version = 8

    # Set opset version
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13

    return model


def main():
    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'onnx')
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ('simple_matmul.onnx', create_simple_matmul()),
        ('simple_conv.onnx', create_simple_conv()),
        ('simple_mlp.onnx', create_simple_mlp()),
        ('binary_weights.onnx', create_binary_weights()),
        ('ternary_weights.onnx', create_ternary_weights()),
    ]

    for filename, model in models:
        filepath = os.path.join(script_dir, filename)
        onnx.save(model, filepath)

        # Validate
        onnx.checker.check_model(model)
        print(f"Created: {filepath}")

    print(f"\nGenerated {len(models)} test models")


if __name__ == '__main__':
    main()
