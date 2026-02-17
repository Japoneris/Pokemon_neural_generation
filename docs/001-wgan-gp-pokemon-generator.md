# 001 — WGAN-GP Pokemon Generator

## Summary

Implements a Pokemon image generator using a **Wasserstein GAN with Gradient Penalty (WGAN-GP)**. Trains on 492 official Sugimori-style Pokemon images and generates new 64x64 Pokemon-like images.

## Architecture

- **Generator**: 5-layer ConvTranspose network. Latent vector `z(128)` → progressively upsampled to `(3, 64, 64)`. Uses BatchNorm + ReLU, final Tanh activation.
- **Discriminator (Critic)**: 5-layer Conv network. Image `(3, 64, 64)` → scalar score. Uses LayerNorm (not BatchNorm — required for WGAN-GP correctness) + LeakyReLU. No sigmoid — outputs unbounded Wasserstein distance estimate.

## Dataset Handling

- PNG images have RGBA transparency — alpha-composited onto white background before converting to RGB
- Images have varying sizes — resized (shortest side) then center-cropped to 64x64
- Augmentation: horizontal flip, ±10° rotation, color jitter
- Normalized to [-1, 1] to match generator's Tanh output range

## Training

WGAN-GP loss: critic trained 5 steps per generator step, gradient penalty λ=10.

Key hyperparameters: batch size 64, lr 1e-4, Adam(β1=0.0, β2=0.9), 1000 epochs.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters |
| `dataset.py` | PokemonDataset + DataLoader |
| `models.py` | Generator + Discriminator |
| `train.py` | Training loop |
| `generate.py` | Inference from checkpoint |
| `utils.py` | Gradient penalty, weight init, image grid |

## Usage

```bash
# Train
python3 train.py

# Resume training
python3 train.py --resume outputs/checkpoints/checkpoint_epoch_0500.pt

# Generate images
python3 generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0950.pt --num 16
```

Samples saved to `outputs/samples/` during training. Checkpoints saved to `outputs/checkpoints/`.
