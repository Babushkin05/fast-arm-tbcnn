# TBN Python Package
"""
TBN Runtime - Optimized inference for ternary-binary neural networks.

This package provides:
- Model loading and inference (from C++ bindings)
- Post-training quantization utilities
"""

# Import from C++ bindings (built by pybind11)
try:
    from ._tbn import load_model, Model, Shape
except ImportError:
    # Fall back for development
    from tbn import load_model, Model, Shape

# Import quantization utilities
from .quantization import (
    quantize_to_binary,
    quantize_to_ternary,
    quantize_onnx_model,
    compare_model_accuracy,
    ModelQuantizer,
    analyze_weight_statistics,
)

__all__ = [
    # Model loading (C++)
    'load_model',
    'Model',
    'Shape',

    # Quantization
    'quantize_to_binary',
    'quantize_to_ternary',
    'quantize_onnx_model',
    'compare_model_accuracy',
    'ModelQuantizer',
    'analyze_weight_statistics',
]

__version__ = '0.1.0'
