#!/bin/bash
"""
Complete conda reinstallation script for GraphARM.
This script removes conda completely and reinstalls it fresh.
"""

echo "🗑️  Complete Conda Reinstallation Script"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Remove existing conda
echo "🗑️  Step 1: Removing existing conda..."
if [ -d "$HOME/anaconda3" ]; then
    echo "Removing ~/anaconda3..."
    rm -rf ~/anaconda3
fi

if [ -d "$HOME/miniconda3" ]; then
    echo "Removing ~/miniconda3..."
    rm -rf ~/miniconda3
fi

# Remove conda from PATH
echo "Cleaning up PATH..."
sed -i '/anaconda3/d' ~/.bashrc 2>/dev/null || true
sed -i '/miniconda3/d' ~/.bashrc 2>/dev/null || true
sed -i '/conda activate/d' ~/.bashrc 2>/dev/null || true

# Remove conda cache
echo "Removing conda cache..."
rm -rf ~/.conda
rm -rf ~/.condarc

# Reload bashrc
source ~/.bashrc

echo "✅ Conda removal completed"

# Step 2: Download and install Miniconda
echo ""
echo "📦 Step 2: Installing Miniconda..."

# Download Miniconda
echo "Downloading Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
echo "Installing Miniconda..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Add to PATH
echo "Adding to PATH..."
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
echo 'conda init bash' >> ~/.bashrc

# Initialize conda
$HOME/miniconda3/bin/conda init bash

# Reload bashrc
source ~/.bashrc

echo "✅ Miniconda installation completed"

# Step 3: Create GraphARM environment
echo ""
echo "🧬 Step 3: Creating GraphARM environment..."

# Create environment
conda create -n grapharm python=3.9 -y

# Activate environment
conda activate grapharm

echo "✅ GraphARM environment created"

# Step 4: Install PyTorch
echo ""
echo "🔥 Step 4: Installing PyTorch..."

# Install PyTorch CPU-only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Test PyTorch
echo "Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installation successful"
else
    echo "❌ PyTorch installation failed, trying pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Step 5: Install PyTorch Geometric
echo ""
echo "📊 Step 5: Installing PyTorch Geometric..."

pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Test PyTorch Geometric
echo "Testing PyTorch Geometric installation..."
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch Geometric installation successful"
else
    echo "❌ PyTorch Geometric installation failed"
fi

# Step 6: Install other requirements
echo ""
echo "📦 Step 6: Installing other requirements..."

pip install -r requirements.txt

# Step 7: Final test
echo ""
echo "🧪 Step 7: Final test..."

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

# Clean up
echo ""
echo "🧹 Cleaning up..."
rm -f Miniconda3-latest-Linux-x86_64.sh

echo "✅ Installation script completed!"


