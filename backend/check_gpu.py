#!/usr/bin/env python3
"""
GPU Availability Checker
Quick script to check if GPU is available for training
"""

import torch
import sys
from pathlib import Path

def check_gpu_detailed():
    """Comprehensive GPU check"""
    print("🔍 GPU Availability Check")
    print("=" * 30)
    
    # Basic CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available")
        print("\n💡 Possible reasons:")
        print("   - No NVIDIA GPU")
        print("   - CUDA drivers not installed")
        print("   - PyTorch CPU-only version")
        return False
    
    # Device count
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")
    
    if device_count == 0:
        print("❌ No CUDA devices found")
        return False
    
    # Device details
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        
        print(f"\n📱 GPU {i}:")
        print(f"   Name: {props.name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute: {props.major}.{props.minor}")
        
        if memory_gb < 2.0:
            print(f"   ⚠️ Warning: Low memory ({memory_gb:.1f} GB < 2.0 GB)")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\nCurrent Device: {current_device}")
    
    # Memory test
    try:
        # Try to allocate a small tensor on GPU
        test_tensor = torch.randn(100, 100).cuda()
        print("✅ GPU memory allocation test: PASSED")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU memory allocation test: FAILED ({e})")
        return False
    
    print("\n🎉 GPU is ready for training!")
    return True

def main():
    gpu_ready = check_gpu_detailed()
    
    if gpu_ready:
        print("\n🚀 Ready to train:")
        print("   python train.py --epochs 50")
        sys.exit(0)
    else:
        print("\n🛑 GPU not ready for training")
        print("💡 Install CUDA and GPU drivers, or use CPU training (slower)")
        sys.exit(1)

if __name__ == "__main__":
    main()