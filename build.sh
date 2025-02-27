#!/bin/bash

# Debugging: Print current directory and contents
echo "=== Current directory ==="
pwd
ls -al

# Install dependencies
echo "=== Installing dependencies ==="
pip install -r requirements.txt

# Create models directory
echo "=== Creating models directory ==="
mkdir -p models

# Download model weights
echo "=== Downloading MiDaS model weights ==="
wget -O models/midas_v21_small_256.pt https://huggingface.co/Rizwandatapro/midas_v21_small_256/resolve/main/midas_v21_small_256.pt

# Clone and verify MiDaS repository
echo "=== Cloning MiDaS repository ==="
git clone https://github.com/intel-isl/MiDaS.git midas_repo

echo "=== Verifying MiDaS folder structure ==="
ls -al midas_repo  # Debug: List repo contents

echo "=== Moving midas folder ==="
mv midas_repo/midas /opt/render/project/src/midas

echo "=== Verifying midas folder contents ==="
ls -al /opt/render/project/src/midas  # Debug: List midas folder contents

# Cleanup
echo "=== Cleaning up ==="
rm -rf midas_repo

echo "✅ Build completed successfully!"
