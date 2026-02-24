# WGAN-GP Hyperparameter Tuning Guide

This guide is tailored to the two WGAN-GP setups in this repository
(`WGAN_small` and `WGAN_large`) and explains what to change and why
when results are not satisfactory.

---

## 1. Reading the Losses

Before touching any hyperparameter, interpret your `losses.csv` output.

| Signal | Meaning |
|---|---|
| `d_loss` steadily near 0, then diverges | Healthy early training, critic is learning |
| `d_loss` → large negative number and stays there | Critic dominates: generator is failing |
| `d_loss` → large positive number | Generator dominates: critic is failing |
| `g_loss` slowly decreasing (less negative) | Generator is improving |
| Both losses oscillate wildly | Learning rate too high or `LAMBDA_GP` too low |

In WGAN-GP, `d_loss = D(fake) - D(real) + λ·GP`. A well-behaved critic
should have `d_loss` near zero in steady state, because `D(fake) ≈ D(real)`
when the generator is good.

---

## 2. The Core Hyperparameters

### 2.1 `LAMBDA_GP` — Gradient Penalty Coefficient

**Current value:** `10.0` (both models — this is the canonical default)

This enforces the 1-Lipschitz constraint on the critic. It is the most
important stability parameter in WGAN-GP.

| Problem | Direction |
|---|---|
| Training unstable, wild oscillations | Increase: `10 → 20` |
| `d_loss` stuck near zero but images stay blurry | Decrease slightly: `10 → 5` |
| Gradient norm printed >> 1 | Increase |

Only change this if training is visually unstable. Do not touch it first.

---

### 2.2 `N_CRITIC` — Critic Steps per Generator Step

**Current value:** `5` (both models — standard)

The critic must be more trained than the generator so it provides a useful
gradient signal.

| Problem | Direction |
|---|---|
| Generator images collapse / all look alike | Increase: `5 → 7` or `10` |
| Generator hardly improves after many epochs | Decrease: `5 → 3` to let G update more often |
| Training very slow on small dataset | Decrease: `5 → 3` |

On a small dataset like Pokémon sprites, increasing `N_CRITIC` beyond 5
rarely helps and slows down the generator.

---

### 2.3 Learning Rates

**WGAN_small:** `LR_G = LR_D = 1e-4`
**WGAN_large:** `LR_G = 1e-5`, `LR_D = 1e-4` (Two Time-scale Update Rule, TTUR)

TTUR (slower generator, faster critic) is generally better. The large model
already uses it. If the small model is unstable or produces blurry results,
switch it to TTUR as well.

```
# WGAN_small — try TTUR
LEARNING_RATE_G = 1e-5
LEARNING_RATE_D = 1e-4
```

You can apply this to an existing run without restarting:
```bash
python train.py --resume checkpoints/checkpoint_epoch_XXXX.pt --lr 1e-5
```
Note: `--lr` currently sets both equally. To get true TTUR you must edit
`config.py` and resume — the checkpoint loader will read the updated
`config.LEARNING_RATE_G/D`.

**General rules:**
- If `d_loss` explodes → halve `LR_D`
- If `g_loss` never improves → halve `LR_G`
- Adam betas `(0.0, 0.9)` are correct for WGAN-GP — **do not change them**

---

### 2.4 `BATCH_SIZE`

**WGAN_small:** `64` | **WGAN_large:** `32`

The gradient penalty is computed on interpolated samples within a batch.
Larger batches give a more stable GP estimate.

| Problem | Direction |
|---|---|
| Training unstable even with high `LAMBDA_GP` | Try `BATCH_SIZE = 32 → 64` |
| GPU memory limited | Decrease; compensate with more `N_CRITIC` steps |

Pokémon datasets are small (~900 sprites). A batch of 64 covers ~7% of
the dataset per step, which is reasonable.

---

### 2.5 `LATENT_DIM`

**Current:** `128` (both models)

This controls the expressiveness of the generator's input space. It is
rarely the bottleneck unless images are very diverse.

- Increasing to `256` can help if generated images look repetitive/averaged.
- Decreasing to `64` can help if the model overfits to a small dataset.

Requires restarting training from scratch (changes architecture).

---

### 2.6 `GEN_FEATURE_MAPS` / `DISC_FEATURE_MAPS`

**WGAN_small:** `64` | **WGAN_large:** `96`

These scale model capacity. Both G and D should grow together. Do not make
D much larger than G — it will overpower G and training collapses.

Requires restarting from scratch.

---

## 3. Data Augmentation Parameters

Located in `config.py`:
```python
HORIZONTAL_FLIP_PROB = 0.5   # effective for sprites
ROTATION_DEGREES    = 10     # small rotations OK for Pokémon
COLOR_JITTER        = (0.1, 0.1, 0.1, 0.02)  # brightness, contrast, sat, hue
```

On a small dataset, augmentation is critical. If results look washed out or
have color artifacts:
- Reduce `COLOR_JITTER` hue: `0.02 → 0.01`
- Reduce `COLOR_JITTER` brightness/saturation: `0.1 → 0.05`

If images look too repetitive (mode collapse), increase augmentation slightly.

---

## 4. Diagnosing Failure Modes

### 4.1 Mode Collapse

**Symptom:** All generated images look nearly identical; `g_loss` very negative
and stable.

**Fixes (in order of priority):**
1. Increase `N_CRITIC` from 5 to 7–10
2. Increase `LATENT_DIM` (128 → 256)
3. Lower `LR_G` (1e-4 → 1e-5) — WGAN_large already does this
4. Increase `GEN_FEATURE_MAPS`

### 4.2 Blurry / Low-Frequency Images

**Symptom:** Generated images are recognizable shapes but soft and without
sharp edges or color blocks.

**Fixes:**
1. Train longer — WGAN-GP typically needs 500–2000 epochs for Pokémon
2. Switch from WGAN_small to WGAN_large (residual blocks + self-attention)
3. Lower `LR_G` slightly to slow down the generator

### 4.3 Checkerboard Artifacts

**Symptom:** Regular grid-like pixel patterns in generated images.

**Root cause:** `ConvTranspose2d` in WGAN_small.

**Fix:** Switch to WGAN_large, which replaces `ConvTranspose2d` with
`Upsample + Conv2d` in the generator — this eliminates checkerboard
patterns by design.

### 4.4 Training Collapse (Loss Explodes)

**Symptom:** `d_loss` or `g_loss` goes to ±∞ within a few epochs.

**Fixes:**
1. Lower learning rates by 5–10×
2. Increase `LAMBDA_GP` (10 → 20)
3. Reduce `BATCH_SIZE` temporarily
4. Check that BatchNorm is **not** used in the discriminator (correct in
   both models — discriminator uses LayerNorm)

---

## 5. Model-Specific Notes

### WGAN_small

- Good for fast experiments and architecture validation
- Checkerboard artifacts are expected due to `ConvTranspose2d`
- If training is stable and results plateau, migrate to WGAN_large

### WGAN_large

- Uses TTUR (`LR_G = 1e-5`, `LR_D = 1e-4`) — keep this asymmetry
- Spectral norm in discriminator + LayerNorm provides double Lipschitz
  regularization, so `LAMBDA_GP = 10` may be reducible to `5` if training
  is stable
- EMA (`EMA_DECAY = 0.999`) smooths the generator weights — the EMA model
  is what you should use for final image generation, not the raw generator

---

## 6. Quick Reference Cheat Sheet

```
Goal                          | Parameter to change
------------------------------|------------------------------------------
More stable training          | ↑ LAMBDA_GP (10 → 20)
Reduce mode collapse          | ↑ N_CRITIC (5 → 7), ↑ LATENT_DIM
Improve image sharpness       | Use WGAN_large, train longer
Fix checkerboard              | Use WGAN_large (Upsample+Conv)
Training too slow             | ↓ N_CRITIC (5 → 3)
Exploding losses              | ↓ LR (÷5 or ÷10), ↑ LAMBDA_GP
Color artifacts               | ↓ COLOR_JITTER, ↓ ROTATION_DEGREES
Generator not learning        | Apply TTUR: LR_G = 1e-5, LR_D = 1e-4
```

---

## 7. Recommended Tuning Order

1. **First:** Verify `d_loss` and `g_loss` trends look healthy (see §1)
2. **Second:** If unstable, adjust `LAMBDA_GP` before anything else
3. **Third:** If collapse, adjust `N_CRITIC` and learning rates
4. **Fourth:** If blurry/checkerboard, migrate to WGAN_large
5. **Last resort:** Increase model capacity (`GEN_FEATURE_MAPS`, `LATENT_DIM`)

Changing multiple parameters at once makes it impossible to know what helped.
Change one parameter at a time and run for at least 200 epochs before judging.
