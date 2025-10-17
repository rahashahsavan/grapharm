#!/bin/bash
"""
GraphARM environment setup script.
This script removes the existing environment and creates a fresh one.
"""

echo "🧬 GraphARM Environment Setup"
echo "============================="

# Step 1: Remove existing environment
echo "🗑️  Removing existing grapharm environment..."
conda remove -n grapharm --all -y 2>/dev/null || true

# Step 2: Create new environment
echo "📦 Creating new grapharm environment..."
conda create -n grapharm python=3.9 -y

# Step 3: Activate environment
echo "🔄 Activating environment..."
conda activate grapharm

# Step 4: Install PyTorch
echo "🔥 Installing PyTorch..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Test PyTorch
echo "🧪 Testing PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installation successful"
else
    echo "❌ PyTorch installation failed, trying pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Step 5: Install PyTorch Geometric
echo "📊 Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Test PyTorch Geometric
echo "🧪 Testing PyTorch Geometric..."
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch Geometric installation successful"
else
    echo "❌ PyTorch Geometric installation failed"
fi

# Step 6: Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Step 7: Final test
echo "🧪 Final test..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Environment setup completed successfully!"
    echo "You can now run GraphARM tests:"
    echo "python quick_test.py"
else
    echo ""
    echo "❌ Environment setup failed."
    exit 1
fi


