# 003 - Device Selection Flag

## Summary

Added a `--device` CLI argument to all `train.py` and `generate.py` scripts (both v1 and v2) so users can force CPU execution when GPU memory is insufficient.

## Changes

- **model_v1/train.py**: Added `--device` argument
- **model_v1/generate.py**: Added `--device` argument
- **model_v2/train.py**: Added `--device` argument
- **model_v2/generate.py**: Added `--device` argument

## Usage

```bash
# Default behavior (auto-detect GPU, fallback to CPU)
python train.py

# Force CPU
python train.py --device cpu

# Force CUDA
python train.py --device cuda
```

The `--device` flag accepts three values:
- `auto` (default): uses CUDA if available, otherwise CPU
- `cpu`: forces CPU execution
- `cuda`: forces CUDA (will error if no GPU is available)
