#!/usr/bin/env python3
"""
CUDA and PyTorch installation checker script.
This script checks if CUDA is properly installed and configured.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_cuda():
    """Check CUDA installation."""
    print("🔍 Checking CUDA installation...")
    
    # Check nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA GPU detected")
        print(f"GPU Info:\n{stdout}")
        return True
    else:
        print("❌ NVIDIA GPU not detected")
        print(f"Error: {stderr}")
        return False

def check_pytorch():
    """Check PyTorch installation."""
    print("\n🔍 Checking PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_conda_cuda():
    """Check CUDA in conda environment."""
    print("\n🔍 Checking conda CUDA packages...")
    
    success, stdout, stderr = run_command("conda list | grep cuda")
    if success and stdout.strip():
        print("✅ CUDA packages found in conda:")
        print(stdout)
        return True
    else:
        print("❌ No CUDA packages found in conda")
        return False

def provide_solution():
    """Provide installation solution."""
    print("\n🔧 SOLUTION:")
    print("=" * 50)
    print("1. Remove current PyTorch:")
    print("   conda remove pytorch torchvision torchaudio -y")
    print("   pip uninstall torch torchvision torchaudio -y")
    print()
    print("2. Install PyTorch with CUDA:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y")
    print()
    print("3. Install CUDA toolkit:")
    print("   conda install cudatoolkit=11.8 -c conda-forge -y")
    print()
    print("4. Verify installation:")
    print("   python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\"")

def main():
    """Main function."""
    print("🚀 CUDA and PyTorch Installation Checker")
    print("=" * 50)
    
    cuda_ok = check_cuda()
    pytorch_ok = check_pytorch()
    conda_cuda_ok = check_conda_cuda()
    
    print("\n📊 SUMMARY:")
    print("=" * 50)
    print(f"NVIDIA GPU: {'✅' if cuda_ok else '❌'}")
    print(f"PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"Conda CUDA: {'✅' if conda_cuda_ok else '❌'}")
    
    if not pytorch_ok:
        provide_solution()
    else:
        print("\n🎉 Everything looks good!")

if __name__ == "__main__":
    main()


