# TBN Python Package - Quantization utilities
"""
Post-training quantization utilities for TBN models.

Converts float weights to binary/ternary representations for
efficient inference on embedded devices.
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from typing import Optional, Tuple, Dict, List


def quantize_to_binary(weights: np.ndarray) -> np.ndarray:
    """
    Quantize float weights to binary {-1, +1}.

    Args:
        weights: Float weight array

    Returns:
        Binary weight array with values -1 or +1
    """
    binary = np.sign(weights)
    # Replace zeros with +1
    binary[binary == 0] = 1
    return binary.astype(np.float32)


def quantize_to_ternary(
    weights: np.ndarray,
    threshold_low: float = -0.1,
    threshold_high: float = 0.1
) -> np.ndarray:
    """
    Quantize float weights to ternary {-1, 0, +1}.

    Args:
        weights: Float weight array
        threshold_low: Values below this become -1
        threshold_high: Values above this become +1

    Returns:
        Ternary weight array with values -1, 0, or +1
    """
    ternary = np.zeros_like(weights, dtype=np.float32)
    ternary[weights < threshold_low] = -1
    ternary[weights > threshold_high] = +1
    return ternary


def analyze_weight_statistics(weights: np.ndarray, name: str = "weights") -> Dict:
    """
    Analyze weight distribution statistics.

    Args:
        weights: Weight array to analyze
        name: Name for display

    Returns:
        Dictionary with statistics
    """
    return {
        'name': name,
        'shape': weights.shape,
        'mean': float(weights.mean()),
        'std': float(weights.std()),
        'min': float(weights.min()),
        'max': float(weights.max()),
        'near_zero_pct': float(100 * np.sum(np.abs(weights) < 0.01) / weights.size),
    }


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    weight_type: str = 'binary',
    threshold_low: float = -0.1,
    threshold_high: float = 0.1,
    verbose: bool = True
) -> Tuple[str, int]:
    """
    Quantize all Conv and Linear weights in an ONNX model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        weight_type: 'binary' or 'ternary'
        threshold_low: Low threshold for ternary quantization
        threshold_high: High threshold for ternary quantization
        verbose: Print progress information

    Returns:
        Tuple of (output_path, number_of_quantized_tensors)
    """
    if verbose:
        print(f"Loading model: {input_path}")

    model = onnx.load(input_path)

    # Map of initializer names to their values
    initializers = {init.name: init for init in model.graph.initializer}

    quantized_count = 0
    quantize_fn = quantize_to_binary if weight_type == 'binary' else quantize_to_ternary

    for node in model.graph.node:
        weight_name = None

        if node.op_type == 'Conv':
            # Conv weights are typically the second input
            if len(node.input) >= 2:
                weight_name = node.input[1]

        elif node.op_type in ('MatMul', 'Gemm'):
            # MatMul/Gemm weights are typically the second input
            if len(node.input) >= 2:
                weight_name = node.input[1]

        if weight_name and weight_name in initializers:
            init = initializers[weight_name]
            weights = numpy_helper.to_array(init)

            if verbose:
                stats = analyze_weight_statistics(weights, weight_name)
                print(f"\nQuantizing {node.op_type} weight: {weight_name}")
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

            # Quantize
            quantized = quantize_fn(weights, threshold_low, threshold_high) \
                if weight_type == 'ternary' else quantize_fn(weights)

            # Create new initializer
            new_init = TensorProto()
            new_init.name = weight_name
            new_init.dims.extend(weights.shape)
            new_init.data_type = TensorProto.FLOAT
            new_init.float_data.extend(quantized.flatten().tolist())

            # Replace in graph
            for i, init in enumerate(model.graph.initializer):
                if init.name == weight_name:
                    model.graph.initializer[i].CopyFrom(new_init)
                    break

            quantized_count += 1

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Quantized {quantized_count} weight tensors to {weight_type}")
        print(f"Saved to: {output_path}")

    return output_path, quantized_count


def compare_model_accuracy(
    original_path: str,
    quantized_path: str,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_samples: int = 100
) -> Tuple[float, float]:
    """
    Compare accuracy of original and quantized models.

    Args:
        original_path: Path to original ONNX model
        quantized_path: Path to quantized ONNX model
        test_images: Test images array (N, C, H, W)
        test_labels: Test labels array (N,)
        num_samples: Number of samples to test

    Returns:
        Tuple of (original_accuracy, quantized_accuracy)
    """
    import onnxruntime as ort

    def evaluate(path: str) -> float:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

        correct = 0
        n = min(num_samples, len(test_images))

        for i in range(n):
            img = np.expand_dims(test_images[i], axis=0)
            output = sess.run(None, {input_name: img})[0]
            if np.argmax(output) == test_labels[i]:
                correct += 1

        return 100.0 * correct / n

    print(f"\nEvaluating on {num_samples} samples...")
    orig_acc = evaluate(original_path)
    print(f"  Original model: {orig_acc:.1f}%")

    quant_acc = evaluate(quantized_path)
    print(f"  Quantized model: {quant_acc:.1f}%")

    print(f"  Accuracy drop: {orig_acc - quant_acc:.1f}%")

    return orig_acc, quant_acc


# Convenience class for model quantization
class ModelQuantizer:
    """
    High-level interface for model quantization.
    """

    def __init__(self, model_path: str):
        """
        Initialize quantizer with a model path.

        Args:
            model_path: Path to ONNX model
        """
        self.model_path = model_path
        self.model = onnx.load(model_path)

    def get_weight_info(self) -> List[Dict]:
        """
        Get information about all weights in the model.

        Returns:
            List of weight info dictionaries
        """
        initializers = {init.name: init for init in self.model.graph.initializer}
        weights_info = []

        for node in self.model.graph.node:
            weight_name = None
            op_type = node.op_type

            if op_type == 'Conv' and len(node.input) >= 2:
                weight_name = node.input[1]
            elif op_type in ('MatMul', 'Gemm') and len(node.input) >= 2:
                weight_name = node.input[1]

            if weight_name and weight_name in initializers:
                init = initializers[weight_name]
                weights = numpy_helper.to_array(init)
                stats = analyze_weight_statistics(weights, weight_name)
                stats['op_type'] = op_type
                weights_info.append(stats)

        return weights_info

    def quantize(
        self,
        output_path: str,
        weight_type: str = 'binary',
        **kwargs
    ) -> str:
        """
        Quantize and save the model.

        Args:
            output_path: Where to save quantized model
            weight_type: 'binary' or 'ternary'
            **kwargs: Additional arguments for quantize_onnx_model

        Returns:
            Path to quantized model
        """
        path, _ = quantize_onnx_model(
            self.model_path,
            output_path,
            weight_type=weight_type,
            **kwargs
        )
        return path
