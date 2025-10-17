#!/bin/bash
"""
Complete PyTorch installation fix script.
This script fixes channel conflicts and installs PyTorch properly.
"""

echo "🔧 PyTorch Installation Fix Script"
echo "==================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check current environment
echo "🔍 Checking current environment..."
echo "Python version: $(python --version)"
echo "Conda version: $(conda --version)"

# Clean up channels
echo ""
echo "🧹 Cleaning up conda channels..."
conda config --remove channels nvidia 2>/dev/null || true
conda config --remove channels pytorch 2>/dev/null || true
conda config --remove channels conda-forge 2>/dev/null || true

# Show current channels
echo "Current channels:"
conda config --show channels

# Remove existing PyTorch
echo ""
echo "🗑️  Removing existing PyTorch..."
conda remove pytorch torchvision torchaudio -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Try conda installation first
echo ""
echo "📦 Trying conda installation..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Test conda installation
echo ""
echo "🧪 Testing conda installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo "✅ Conda installation successful!"
    INSTALLATION_METHOD="conda"
else
    echo "❌ Conda installation failed. Trying pip..."
    
    # Try pip installation
    echo ""
    echo "📦 Trying pip installation..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Test pip installation
    echo ""
    echo "🧪 Testing pip installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    
    if [ $? -eq 0 ]; then
        echo "✅ Pip installation successful!"
        INSTALLATION_METHOD="pip"
    else
        echo "❌ All installation methods failed."
        exit 1
    fi
fi

# Install PyTorch Geometric
echo ""
echo "📦 Installing PyTorch Geometric..."
if [ "$INSTALLATION_METHOD" = "conda" ]; then
    conda install pyg -c pyg -y
else
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
fi

# Install other requirements
echo ""
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Final test
echo ""
echo "🧪 Final test..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo "You can now run GraphARM tests:"
    echo "python quick_test.py"
else
    echo ""
    echo "❌ Installation failed."
    exit 1
fi


