#!/usr/bin/env python3
"""
Test script to verify CPU/GPU device selection
"""
import sys
import torch

# Add game directory to path
sys.path.insert(0, '.')

print("=" * 60)
print("DQN Missile Evasion Game - Device Detection Test")
print("=" * 60)

# Test 1: Auto-detect
print("\n[Test 1] Auto-detect mode:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Selected Device: {device}")
print(f"  Device Type: {device.type}")

# Test 2: Force CPU
print("\n[Test 2] Force CPU mode:")
device_cpu = torch.device("cpu")
print(f"  Selected Device: {device_cpu}")
print(f"  Device Type: {device_cpu.type}")

# Test 3: Force CUDA (if available)
print("\n[Test 3] Force CUDA mode:")
if torch.cuda.is_available():
    device_cuda = torch.device("cuda")
    print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Device Properties:")
    print(f"    - Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"  CUDA not available (will fallback to CPU)")

# Test DQN tensor operations on device
print("\n[Test 4] DQN Tensor Operations Test:")
test_tensor = torch.randn(4, 9)  # Batch of 4 states with 9 features
print(f"  Original tensor device: {test_tensor.device}")
test_tensor = test_tensor.to(device)
print(f"  After .to(device): {test_tensor.device}")

print("\n" + "=" * 60)
print("✓ All device tests completed successfully!")
print("=" * 60)
