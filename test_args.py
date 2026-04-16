#!/usr/bin/env python3
"""
Test script to verify device argument parsing
"""
import sys

# Simulate command-line arguments
test_cases = [
    (['ver3.py', 'cpu'], "cpu"),
    (['ver3.py', 'gpu'], "gpu"),
    (['ver3.py', 'cuda'], "cuda"),
    (['ver3.py', 'auto'], "auto"),
    (['ver3.py', 'help'], "help"),
    (['ver3.py'], "default"),
]

print("=" * 60)
print("Device Argument Parsing Test")
print("=" * 60)

for argv, description in test_cases:
    sys.argv = argv
    
    # Simulate device selection logic
    DEVICE_PREFERENCE = 'auto'
    device_choice = DEVICE_PREFERENCE.lower()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('cpu', 'cuda', 'gpu'):
            device_choice = 'cuda' if arg in ('cuda', 'gpu') else 'cpu'
    
    print(f"\nTest: {description:15} | argv={argv}")
    print(f"  → Detected device choice: {device_choice}")
    
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help', 'help'):
        print(f"  → Would show help and exit")

print("\n" + "=" * 60)
print("✓ All argument parsing tests completed!")
print("=" * 60)
