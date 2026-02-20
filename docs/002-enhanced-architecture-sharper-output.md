# 002 - Enhanced Architecture for Sharper Pokemon Generation

## Summary

Created `model_v2/` with architectural improvements to address blurry outputs from the v1 WGAN-GP after 1000 epochs. The original code is preserved in `model_v1/`.

## Problem

Generated Pokemon images from v1 were too blurry at 64x64, lacking fine detail and spatial coherence.

## Root Causes Addressed

1. **Checkerboard artifacts** from `ConvTranspose2d` overlap patterns
2. **No residual connections** — shallow effective depth limits detail
3. **No attention** — can't model long-range spatial dependencies (e.g., symmetry)
4. **No EMA** — generator weight oscillation degrades inference quality
5. **Limited capacity** — only 64 base feature maps

## Changes in model_v2/

### Architecture (models.py)

- **Upsample + Conv2d** replaces `ConvTranspose2d` in generator (except initial 1x1→4x4 projection). Eliminates checkerboard artifacts.
- **Residual blocks** after each up/downsampling stage in both G and D. `ResidualBlock` (G) uses BatchNorm + ReLU; `ResidualBlockDisc` (D) uses LayerNorm + LeakyReLU.
- **Self-attention** at 32x32 resolution in both G and D for global coherence.
- **Spectral normalization** on all Conv2d layers in the discriminator for training stability.
- **Increased capacity**: 64 → 96 base feature maps.

### Training (train.py, config.py)

- **EMA generator** (decay=0.999): maintained alongside the training generator; used for sample generation and saved in checkpoints.
- **TTUR**: generator LR reduced to 5e-5 while discriminator stays at 1e-4.
- **Epochs**: 1000 → 2000 for the deeper architecture to converge.

### Generation (generate.py)

- Loads EMA weights by default (`--no-ema` flag for raw generator).

## Parameter Counts

| Component     | v1        | v2         |
|---------------|-----------|------------|
| Generator     | ~2.8M     | ~8.6M      |
| Discriminator | ~2.8M     | ~21.2M     |

## File Structure

```
models/
  WGAN_small/    # Original v1 code (paths point to ../../dataset/ and ../../outputs/WGAN_small/)
  WGAN_large/    # Enhanced v2 code (paths point to ../../dataset/ and ../../outputs/WGAN_large/)
    config.py
    models.py
    dataset.py
    train.py
    generate.py
    utils.py
    plot_losses.py
outputs/
  WGAN_small/
    checkpoints/
    samples/
    losses.csv
  WGAN_large/
    checkpoints/
    samples/
    losses.csv
```

## Usage

```bash
cd models/WGAN_large
source ../../venv/bin/activate
python3 train.py                    # Start training
python3 train.py --resume ../../outputs/WGAN_large/checkpoints/checkpoint_epoch_0500.pt  # Resume
python3 generate.py --checkpoint ../../outputs/WGAN_large/checkpoints/checkpoint_epoch_1000.pt --num 64
```
