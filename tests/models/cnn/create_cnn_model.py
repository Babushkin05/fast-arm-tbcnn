#!/usr/bin/env python3
"""
Create a simple LeNet-like CNN model for testing TBN runtime.
Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> FC
"""

import torch
import torch.nn as nn
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  # 28x28 -> 24x24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 12x12 -> 8x8

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # After 2x2 pooling: 8x8 -> 4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Conv1 + ReLU + Pool: 28x28 -> 24x24 -> 12x12
        x = self.pool(torch.relu(self.conv1(x)))

        # Conv2 + ReLU + Pool: 12x12 -> 8x8 -> 4x4
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, 16 * 4 * 4)

        # FC layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    # Create model
    model = SimpleCNN()
    model.eval()

    # Create dummy input (batch=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Ensure output directory exists
    output_dir = "onnx"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simple_cnn.onnx")

    # Export to ONNX - simpler version
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,  # Use a more compatible version
        do_constant_folding=True,
    )

    print(f"Model exported to {output_path}")

    # Test inference
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output[0][:5].tolist()}...")

if __name__ == "__main__":
    main()
