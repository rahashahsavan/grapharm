#!/bin/bash
"""
CUDA installation fix script for GraphARM.
This script fixes common CUDA installation issues.
"""

echo "🔧 CUDA Installation Fix Script"
echo "================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check NVIDIA GPU
echo "🔍 Checking NVIDIA GPU..."
if command_exists nvidia-smi; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi
else
    echo "❌ NVIDIA GPU not detected"
    echo "Please install NVIDIA drivers first"
    exit 1
fi

# Remove existing PyTorch
echo ""
echo "🗑️  Removing existing PyTorch..."
conda remove pytorch torchvision torchaudio -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Remove existing CUDA packages
echo ""
echo "🗑️  Removing existing CUDA packages..."
conda remove cudatoolkit cuda-toolkit cuda-runtime cuda-libraries -y 2>/dev/null || true

# Install CUDA toolkit
echo ""
echo "📦 Installing CUDA toolkit..."
conda install cuda-toolkit=11.8 -c nvidia -y

# Install PyTorch with CUDA
echo ""
echo "📦 Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Test installation
echo ""
echo "🧪 Testing installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation successful!"
    echo "You can now run GraphARM tests:"
    echo "python quick_test.py"
else
    echo ""
    echo "❌ Installation failed. Trying alternative method..."
    
    # Try pip installation
    echo "📦 Trying pip installation..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Test again
    echo "🧪 Testing pip installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Installation successful with pip!"
    else
        echo ""
        echo "❌ All installation methods failed."
        echo "Please check your CUDA installation manually."
    fi
fi


