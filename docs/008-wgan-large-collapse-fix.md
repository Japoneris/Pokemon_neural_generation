# WGAN-Large: Mode Collapse Diagnosis and Fix

## Observation

After 1260 epochs, WGAN_large generates uniform gray/beige squares with no
structure. WGAN_small at the same stage already shows recognizable Pokémon
shapes. The generator collapsed from epoch 0 and never recovered.

## Loss Analysis

| Metric | WGAN_large (epoch 1260) | WGAN_small (epoch 1260) |
|---|---|---|
| `d_loss` | ≈ -12 | ≈ -7 |
| `g_loss` | ≈ 65 (and rising) | ≈ 21 (stable) |

The climbing `g_loss` means `D(fake)` is increasingly negative — the critic
rejects fakes more and more harshly while the generator cannot respond.

## Root Cause: Generator Learning Speed 50× Slower Than Discriminator

The config set:
- `LR_G = 1e-5`, `LR_D = 1e-4` → 10× TTUR ratio
- `N_CRITIC = 5` → 5 critic updates per generator update

Combined, the discriminator was effectively learning **50× faster** than the
generator. The WGAN-large discriminator is also architecturally more powerful
(spectral norm + LayerNorm + residual blocks) than the small one. This gave the
critic an overwhelming advantage from batch 1, driving the generator to collapse
onto a constant gray output — the only "strategy" that produces a stable critic
score when you can't actually fool it.

The TTUR paper (Heusel et al. 2017) recommends a **2–4× ratio**, not 10×.
The original WGAN-GP paper uses **equal rates**.

The collapse is irreversible from a checkpoint: the Adam optimizer's moment
estimates have been shaped by 1260 epochs of uniform-output gradients and
cannot recover. A full restart is required.

## Changes Made to `config.py`

| Parameter | Old | New | Reason |
|---|---|---|---|
| `LEARNING_RATE_G` | `1e-5` | `1e-4` | Match LR_D; equal rates work in WGAN_small |
| `BATCH_SIZE` | `32` | `64` | More stable gradient penalty estimates; matches small |
| `N_CRITIC` | `5` | `3` | Generator updates more frequently relative to critic |
| `EMA_DECAY` | `0.999` | `0.995` | Faster EMA warm-up; 0.999 barely moved in early epochs |

Note: `LEARNING_RATE_D` stays at `1e-4`. The discriminator's spectral norm
already provides strong Lipschitz regularization on top of LAMBDA_GP=10,
so it does not need to be slowed down further.

## Restart Instructions

Do **not** resume from any existing checkpoint — the optimizer state is
poisoned. Start fresh:

```bash
cd models/WGAN_large
python train.py
```

## What to Watch For

- **Epoch 0–50**: `d_loss` should be in the range -50 to -100 initially, then
  converge toward 0. If it immediately stabilizes near -100 and `g_loss` climbs,
  the generator is collapsing again.
- **Epoch 50–200**: Sample images should show blurry but non-uniform blobs.
  If they are still uniform, increase `LR_G` further or reduce `N_CRITIC` to 2.
- **Healthy steady state**: `d_loss` ≈ -5 to -15, `g_loss` stable or gently
  decreasing.

## If It Collapses Again

The large architecture may simply need equal or near-equal learning rates.
Further tuning levers:
1. Lower `N_CRITIC` to 2
2. Raise `LR_G` to `2e-4` (slightly above LR_D)
3. Reduce `LAMBDA_GP` to 5 (spectral norm already provides regularization)
