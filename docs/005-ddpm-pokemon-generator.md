# 005 - DDPM Pokemon Generator (Stable Diffusion)

## Summary

Added a new generative model using **Denoising Diffusion Probabilistic Models (DDPM)** under `models/stable_diffusion/`. Unlike WGAN models which use adversarial training, this model learns to reverse a gradual noising process to generate 64x64 Pokemon images.

## Architecture

### UNet Noise Prediction Network

The core model is a UNet that predicts the noise added to an image at a given timestep:

| Stage | Resolution | Channels | Features |
|-------|-----------|----------|----------|
| Encoder L0 | 64x64 | 64 | 2 ResBlocks |
| Encoder L1 | 32x32 | 128 | 2 ResBlocks |
| Encoder L2 | 16x16 | 256 | 2 ResBlocks + Self-Attention |
| Encoder L3 | 8x8 | 512 | 2 ResBlocks |
| Middle | 8x8 | 512 | ResBlock + Attention + ResBlock |
| Decoder | symmetric | symmetric | 3 ResBlocks/level + skip connections |

- **~58M parameters**
- **GroupNorm + SiLU** activations throughout
- **Sinusoidal time embedding** -> MLP -> additive injection in ResBlocks
- **Multi-head self-attention** (4 heads) at 16x16 resolution

### Diffusion Process

- **1000 timesteps** with linear beta schedule (1e-4 to 0.02)
- **Forward process**: Gradually adds Gaussian noise to training images
- **Training loss**: Simple MSE between predicted noise and actual noise
- **Sampling**: Supports both DDPM (1000 steps) and DDIM (50 steps, much faster)

## Key Differences from v1/v2

| Aspect | v1/v2 (WGAN-GP) | v3 (DDPM) |
|--------|-----------------|-----------|
| **Approach** | Adversarial (generator vs discriminator) | Denoising (predict noise to remove) |
| **Training** | Minimax game, can be unstable | Single MSE objective, very stable |
| **Generation** | Single forward pass (~instant) | Iterative denoising (50-1000 steps) |
| **Quality** | Can suffer mode collapse | Better diversity, no mode collapse |
| **Architecture** | Generator + Discriminator | Single UNet |

## Training Configuration

- **Batch size**: 32 (smaller than GAN due to UNet memory)
- **Optimizer**: AdamW, lr=2e-4
- **LR schedule**: 500-step linear warmup + cosine annealing
- **Gradient clipping**: max norm 1.0
- **EMA**: decay=0.9999, updated every step
- **Samples**: Generated every 10 epochs using DDIM (50 steps)

## Usage

### Training
```bash
cd models/stable_diffusion
python3 train.py                          # Train from scratch
python3 train.py --resume ../../outputs/stable_diffusion/checkpoints/checkpoint_epoch_0500.pt  # Resume
python3 train.py --device cuda --epochs 2000 --lr 1e-4  # Custom settings
```

### Generation
```bash
python3 generate.py --checkpoint ../../outputs/stable_diffusion/checkpoints/checkpoint_epoch_0500.pt
python3 generate.py --checkpoint <path> --num 64 --sampler ddim --ddim-steps 50
python3 generate.py --checkpoint <path> --sampler ddpm  # Full 1000-step sampling
```

## File Structure

```
models/stable_diffusion/
├── config.py       # All hyperparameters
├── dataset.py      # Same dataset pipeline as other models
├── models.py       # UNet with time conditioning (~58M params)
├── train.py        # DDPM training loop
├── generate.py     # DDPM/DDIM sampling CLI
└── utils.py        # GaussianDiffusion class, EMA, image saving
```
