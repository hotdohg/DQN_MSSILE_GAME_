# DQN Missile Evasion Game - CPU/GPU Configuration Guide

## Overview

The game now supports flexible CPU/GPU switching for PyTorch DQN computations. You can choose between:
- **CPU**: More compatible, slower (good for testing on any machine)
- **GPU (CUDA)**: Much faster (requires NVIDIA GPU with CUDA support)
- **Auto-detect**: Automatically uses GPU if available, falls back to CPU

---

## Running the Game

### Method 1: Command-Line Arguments (Recommended)

Run the game with a device argument:

```bash
# Force CPU mode
python ver3.py cpu

# Use GPU (CUDA) mode
python ver3.py gpu
# or
python ver3.py cuda

# Auto-detect (default - CUDA if available, else CPU)
python ver3.py auto

# Show help
python ver3.py help
```

### Method 2: Modify Code Constant

Edit the `DEVICE_PREFERENCE` constant in `ver3.py` (line ~105):

```python
# Device settings (CPU or GPU)
# Options: 'auto' (auto-detect), 'cpu', 'cuda', 'gpu'
DEVICE_PREFERENCE = 'auto'  # <-- Change this to 'cpu' or 'cuda'
```

Then run normally:
```bash
python ver3.py
```

---

## Device Priority

When starting the game, device selection follows this priority:

1. **Command-line argument** (if provided)
   ```bash
   python ver3.py cpu  # This overrides everything
   ```

2. **DEVICE_PREFERENCE constant** (if no command-line arg)
   ```python
   DEVICE_PREFERENCE = 'cuda'  # This is used if no command-line arg
   ```

3. **Auto-detection** (if DEVICE_PREFERENCE is 'auto')
   - Checks if CUDA is available
   - Uses GPU if available, CPU otherwise

---

## Checking Your System

To see what devices are available:

```bash
# Run the test script
python test_device.py

# Or check directly in Python
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### On Windows with NVIDIA GPU:
```
CUDA Available: True
CUDA Device Name: NVIDIA GeForce RTX 4090
```

### Without GPU or CPU only:
```
CUDA Available: False
(Will use CPU)
```

---

## Game Output

When you start the game, you'll see the device being used:

### CPU Mode Output:
```
DQN Missile Evasion Game
==================================================
Using device: cpu
DQN Network: input=9, hidden=128, actions=3
Replay Buffer Size: 10000
Training: epsilon 1.0 → 0.05, decay 0.995
```

### GPU Mode Output (if CUDA available):
```
DQN Missile Evasion Game
==================================================
Using device: cuda:0
DQN Network: input=9, hidden=128, actions=3
Replay Buffer Size: 10000
Training: epsilon 1.0 → 0.05, decay 0.995
```

### Fallback (GPU requested but not available):
```
DQN Missile Evasion Game
==================================================
⚠️  CUDA not available, falling back to CPU
Using device: cpu
DQN Network: input=9, hidden=128, actions=3
```

---

## Performance Comparison

| Device | Speed | Memory | Best For |
|--------|-------|--------|----------|
| CPU | Moderate | Minimal | Testing, CPU-only systems |
| GPU | 10-50x faster | Needs VRAM | Production, fast training |

---

## Troubleshooting

### GPU not detected even though you have NVIDIA GPU?

1. Check NVIDIA drivers:
```bash
nvidia-smi
```

2. Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory (OOM) on GPU?

- The game uses 10000 replay buffer size which requires ~500MB GPU memory
- If you get OOM errors, reduce `REPLAY_BUFFER_SIZE` in the code:
  ```python
  REPLAY_BUFFER_SIZE = 5000  # Smaller buffer
  ```

### Slow training on CPU?

- CPU training is normal, but slower
- For production use, consider using GPU
- You can also reduce `DQN_BATCH_SIZE`:
  ```python
  DQN_BATCH_SIZE = 32  # Smaller batches = faster
  ```

---

## Code Implementation Details

### Device Detection Method

The game uses a `_get_device()` method in the Game class:

```python
def _get_device(self):
    """Determine which device to use (CPU or GPU).
    Priority: command-line args > DEVICE_PREFERENCE constant > auto-detect
    """
    device_choice = DEVICE_PREFERENCE.lower()
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('cpu', 'cuda', 'gpu'):
            device_choice = 'cuda' if arg in ('cuda', 'gpu') else 'cpu'
    
    # Determine device
    if device_choice == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice in ('cuda', 'gpu'):
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:  # 'cpu'
        device = torch.device("cpu")
    
    return device
```

### Device Usage in DQN

The device is passed to the DQN agent during initialization:

```python
self.device = self._get_device()
self.dqn_agent = DQNAgent(DQN_STATE_SIZE, DQN_ACTION_SIZE, device=self.device)
```

All tensors are moved to the selected device using `.to(device)`:

```python
states_t = torch.from_numpy(states).float().to(self.device)
actions_t = torch.from_numpy(actions).long().to(self.device)
```

---

## Files

- `ver3.py` - Main game file with CPU/GPU support
- `test_device.py` - Device detection test script
- `README_DEVICE.md` - This file

---

## Summary

**Quick Start:**
```bash
# Auto-detect (recommended)
python ver3.py

# Force GPU
python ver3.py gpu

# Force CPU
python ver3.py cpu

# Show all options
python ver3.py help
```

The game will automatically display which device is being used when it starts!
