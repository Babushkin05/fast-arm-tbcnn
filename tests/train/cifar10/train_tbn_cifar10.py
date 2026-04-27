#!/usr/bin/env python3
"""
Train a TBN-style CNN on CIFAR-10 for benchmarking.

Architecture:
- Conv1: 3 -> 32 channels, 3x3 kernel
- Conv2: 32 -> 64 channels, 3x3 kernel
- Conv3: 64 -> 128 channels, 3x3 kernel
- FC1: 128*4*4 -> 256
- FC2: 256 -> 10

This creates larger GeMM operations for meaningful benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import onnx
from onnx import numpy_helper
import numpy as np
import os
import kagglehub
from PIL import Image

class TBNCifarNet(nn.Module):
    """CNN for CIFAR-10 with binary weights support."""

    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten and FC
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def load_cifar10():
    """Load CIFAR-10 from Kaggle (image folders structure)."""
    print("Downloading CIFAR-10 from Kaggle...")
    path = kagglehub.dataset_download("ayush1220/cifar10")
    print(f"Dataset path: {path}")

    # The structure is: path/cifar10/train/{class_name}/*.png
    # and: path/cifar10/test/{class_name}/*.png
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Normalization params
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

    def load_images(split):
        """Load all images from a split (train or test)."""
        split_path = os.path.join(path, 'cifar10', split)
        images = []
        labels = []

        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            class_idx = class_to_idx[class_name]

            for img_name in os.listdir(class_path):
                if not img_name.endswith('.png'):
                    continue

                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32) / 255.0
                # HWC -> CHW
                img_array = np.transpose(img_array, (2, 0, 1))
                # Normalize
                img_array = (img_array - mean) / std
                images.append(img_array)
                labels.append(class_idx)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

    print("Loading training images...")
    train_data, train_labels = load_images('train')
    print(f"Loaded {len(train_data)} training images")

    print("Loading test images...")
    test_data, test_labels = load_images('test')
    print(f"Loaded {len(test_data)} test images")

    return train_data, train_labels, test_data, test_labels


def load_cifar10_direct():
    """Load CIFAR-10 from direct URL download (fallback for SSL issues)."""
    import urllib.request
    import tarfile
    import pickle

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir = './data/cifar-10-batches-py'

    # Download if not exists
    if not os.path.exists(data_dir):
        os.makedirs('./data', exist_ok=True)
        print(f"Downloading CIFAR-10 from {url}...")
        tar_path = './data/cifar-10-python.tar.gz'
        urllib.request.urlretrieve(url, tar_path)

        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall('./data')
        os.remove(tar_path)

    # Load batches
    def load_batch(filepath):
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        return batch['data'], batch['labels']

    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join('./data', 'cifar-10-batches-py', f'data_batch_{i}')
        data, labels = load_batch(batch_file)
        train_data.append(data)
        train_labels.extend(labels)

    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    train_labels = np.array(train_labels)

    # Load test data
    test_file = os.path.join('./data', 'cifar-10-batches-py', 'test_batch')
    test_data, test_labels = load_batch(test_file)
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = np.array(test_labels)

    # Normalize
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    print(f"Loaded {len(train_data)} training and {len(test_data)} test images")
    return train_data, train_labels, test_data, test_labels


def train_model(epochs=20, batch_size=128, lr=0.001):
    """Train the model on CIFAR-10."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load data
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create data loaders
    trainset = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torch.utils.data.TensorDataset(test_data, test_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Model
    model = TBNCifarNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Test Acc: {100.*correct/total:.2f}%')

    return model, testloader


def export_to_onnx(model, output_path='cifar10_tbn.onnx'):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )

    print(f"Model exported to {output_path}")

    # Convert raw_data to float_data for C++ compatibility
    onnx_model = onnx.load(output_path)
    for init in onnx_model.graph.initializer:
        if len(init.raw_data) > 0 and len(init.float_data) == 0 and init.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            init.ClearField('raw_data')
            init.float_data.extend(arr.flatten().tolist())

    onnx.save(onnx_model, output_path)
    print("Converted raw_data to float_data for C++ compatibility")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(output_dir, 'cifar10_tbn.onnx')

    if args.export_only:
        # Load existing model and export
        model = TBNCifarNet()
        # You would load weights here
        export_to_onnx(model, model_path)
        return

    # Train
    print("Training CIFAR-10 model...")
    model, testloader = train_model(epochs=args.epochs)

    # Export
    export_to_onnx(model, model_path)

    # Final test
    print("\nFinal evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Final Test Accuracy: {100.*correct/total:.2f}%')


if __name__ == '__main__':
    main()
