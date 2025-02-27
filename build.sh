#!/bin/bash

# Install Python dependencies
echo "Installing dependencies..."
pip install numpy>=1.23.5  # Install numpy first to resolve OpenCV conflicts
pip install -r requirements.txt

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Download the MiDaS model weights
echo "Downloading MiDaS model..."
wget -O models/midas_v21_small_256.pt https://huggingface.co/Rizwandatapro/midas_v21_small_256/resolve/main/midas_v21_small_256.pt

# Clone the MiDaS repository
echo "Cloning MiDaS repository..."
git clone https://github.com/intel-isl/MiDaS.git midas_repo

# Move the midas folder to the project directory
echo "Moving MiDaS files..."
mv midas_repo/midas /opt/render/project/src/midas

# Clean up the repository
echo "Cleaning up..."
rm -rf midas_repo

echo "✅ Build completed successfully!"
