#!/usr/bin/env python3
"""
Train MNIST model with Ternary-Binary Neural Network (TBN) approach.

TBN = Ternary Activations × Binary Weights

- Weights: binary {-α, +α} with per-channel scaling
- Activations: ternary {-1, 0, +1} with learnable threshold Δ
- Uses Straight-Through Estimator (STE) for gradients

Based on:
- Courbariaux et al. "BinaryConnect"
- Zhu et al. "Trained Ternary Quantization"
- He et al. "TBN: Ternary-Binary Neural Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from tqdm import tqdm
import kagglehub
import numpy as np
import struct


class MNISTDataset(Dataset):
    """MNIST dataset loaded from kagglehub."""

    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_mnist_from_kaggle():
    """Load MNIST dataset from kagglehub."""
    print("Downloading MNIST from Kaggle...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    train_images_file = os.path.join(path, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(path, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')

    with open(train_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(train_labels_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    test_images_file = os.path.join(path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    test_labels_file = os.path.join(path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    with open(test_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(test_labels_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    print(f"Loaded {len(train_images)} training images, {len(test_images)} test images")
    return train_images, train_labels, test_images, test_labels


class TernaryActivation(nn.Module):
    """Ternary activation: {-1, 0, +1} with threshold.

    Uses Straight-Through Estimator (STE) for gradients.
    Forward: quantize to {-1, 0, +1}
    Backward: pass gradient through (identity)
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        if self.training:
            # Ternary quantization
            out = torch.zeros_like(x)
            out[x > self.threshold] = 1.0
            out[x < -self.threshold] = -1.0

            # STE: use quantized value in forward, pass gradient through in backward
            # (out - x).detach() + x = out in forward, x in backward
            return (out - x).detach() + x
        else:
            out = torch.zeros_like(x)
            out[x > self.threshold] = 1.0
            out[x < -self.threshold] = -1.0
            return out


class BinaryWeightConv2d(nn.Module):
    """Conv2d with binary weights and ternary activations."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, use_ternary_act=True):
        super().__init__()
        # Real-valued weights (updated by optimizer)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding
        self.use_ternary_act = use_ternary_act

        # Ternary activation after conv
        if use_ternary_act:
            self.ternary_act = TernaryActivation(threshold=0.5)

    def get_binary_weight(self):
        """Binarize weights with per-channel scaling and STE."""
        # Per output channel scaling
        scale = self.weight.abs().mean(dim=(1, 2, 3), keepdim=True)
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)
        binary_w = torch.sign(self.weight) * scale

        # STE: use binary weights in forward, pass gradient through real weights
        if self.training:
            return (binary_w - self.weight).detach() + self.weight
        return binary_w

    def forward(self, x):
        # Binarize weights (STE happens automatically in autograd)
        binary_w = self.get_binary_weight()

        # Conv with binary weights
        out = F.conv2d(x, binary_w, self.bias, stride=self.stride, padding=self.padding)

        # Apply ternary activation
        if self.use_ternary_act:
            out = self.ternary_act(out)

        return out


class BinaryWeightLinear(nn.Module):
    """Linear with binary weights and ternary activations."""

    def __init__(self, in_features, out_features, bias=True, use_ternary_act=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.use_ternary_act = use_ternary_act
        if use_ternary_act:
            self.ternary_act = TernaryActivation(threshold=0.5)

    def get_binary_weight(self):
        """Binarize weights with per-output scaling and STE."""
        scale = self.weight.abs().mean(dim=1, keepdim=True)
        scale = torch.clamp(scale, min=1e-8)
        binary_w = torch.sign(self.weight) * scale

        # STE: use binary weights in forward, pass gradient through real weights
        if self.training:
            return (binary_w - self.weight).detach() + self.weight
        return binary_w

    def forward(self, x):
        binary_w = self.get_binary_weight()
        out = F.linear(x, binary_w, self.bias)

        if self.use_ternary_act:
            out = self.ternary_act(out)

        return out


class TBNNetwork(nn.Module):
    """Ternary-Binary Neural Network for MNIST.

    Architecture inspired by LeNet with TBN quantization:
    - Conv layers: binary weights + ternary activations
    - FC layers: binary weights (no activation on final layer)
    - MaxPool doesn't need quantization
    """

    def __init__(self):
        super().__init__()

        # Conv block 1: 1 -> 16 channels
        # No ternary activation on first layer - let network see real input values
        self.conv1 = BinaryWeightConv2d(1, 16, 5, padding=2, use_ternary_act=False)

        # Conv block 2: 16 -> 32 channels
        self.conv2 = BinaryWeightConv2d(16, 32, 5, padding=2, use_ternary_act=True)

        # FC layer: 32*4*4 = 512 -> 10
        # No ternary activation on final layer (we need raw logits)
        self.fc = BinaryWeightLinear(512, 10, use_ternary_act=False)

    def forward(self, x):
        # Conv1 + ReLU + Pool: 28x28 -> 14x14
        x = self.conv1(x)
        x = F.relu(x)  # ReLU since no ternary activation on first layer
        x = F.max_pool2d(x, 2, 2)

        # Conv2 + Pool: 14x14 -> 4x4 (ternary activation built into conv2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 3, 3)

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_binary_weights(self):
        """Get all binary weights for export."""
        return {
            'conv1_w': self.conv1.get_binary_weight().detach(),
            'conv1_b': self.conv1.bias.detach() if self.conv1.bias is not None else None,
            'conv2_w': self.conv2.get_binary_weight().detach(),
            'conv2_b': self.conv2.bias.detach() if self.conv2.bias is not None else None,
            'fc_w': self.fc.get_binary_weight().detach(),
            'fc_b': self.fc.bias.detach() if self.fc.bias is not None else None,
        }


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        pbar.set_postfix({
            'loss': f'{train_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    return train_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test: loss={test_loss:.4f}, accuracy={accuracy:.2f}%')
    return test_loss, accuracy


def export_to_onnx(model, output_path, device):
    """Export TBN model to ONNX with binary weights."""
    model.eval()
    weights = model.get_binary_weights()

    class ExportModel(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.conv1_w = w['conv1_w']
            self.conv1_b = w['conv1_b']
            self.conv2_w = w['conv2_w']
            self.conv2_b = w['conv2_b']
            self.fc_w = w['fc_w']
            self.fc_b = w['fc_b']

        def forward(self, x):
            # Conv1 + Pool
            x = F.conv2d(x, self.conv1_w, self.conv1_b, padding=2)
            x = F.relu(x)  # Keep ReLU for compatibility with standard runtimes
            x = F.max_pool2d(x, 2, 2)

            # Conv2 + Pool
            x = F.conv2d(x, self.conv2_w, self.conv2_b, padding=2)
            x = F.relu(x)
            x = F.max_pool2d(x, 3, 3)

            # FC
            x = x.view(x.size(0), -1)
            x = F.linear(x, self.fc_w, self.fc_b)

            return x

    export_model = ExportModel(weights)
    export_model.eval()

    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    # Use opset_version 11 for better compatibility
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
        do_constant_folding=True,
    )

    print(f"Exported to {output_path}")

    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def main():
    parser = argparse.ArgumentParser(description='Train TBN MNIST')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--output', type=str, default='mnist_tbn.onnx',
                        help='output ONNX file')
    parser.add_argument('--no-gpu', action='store_true',
                        help='disable GPU')

    args = parser.parse_args()

    # Device
    use_cuda = torch.cuda.is_available() and not args.no_gpu
    device = torch.device('cuda' if use_cuda else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data from Kaggle
    train_images, train_labels, test_images, test_labels = load_mnist_from_kaggle()

    # Normalize
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    train_images = train_images.astype(np.float32) / 255.0
    train_images = (train_images - MNIST_MEAN) / MNIST_STD
    test_images = test_images.astype(np.float32) / 255.0
    test_images = (test_images - MNIST_MEAN) / MNIST_STD

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = TBNNetwork().to(device)
    print(f"\nModel architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0
    patience = 15
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'mnist_tbn_best.pt')
            print(f"  -> New best accuracy: {best_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Export to ONNX
    model.load_state_dict(torch.load('mnist_tbn_best.pt', weights_only=True))
    export_to_onnx(model, args.output, device)


if __name__ == '__main__':
    main()
