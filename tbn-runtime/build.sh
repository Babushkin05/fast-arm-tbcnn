#!/bin/bash

# Build script for TBN Runtime

set -e

echo "Building TBN Runtime..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTBN_BUILD_TESTS=ON \
    -DTBN_BUILD_EXAMPLES=ON \
    -DTBN_BUILD_BENCHMARKS=OFF

# Build
echo "Building..."
make -j$(nproc)

echo "Build complete!"
echo ""
echo "To run tests: make test"
echo "To run example: ./bin/simple_example"
echo "To install: sudo make install"