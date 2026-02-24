# 006 - Denoising Process Visualization

## Overview

Added the ability to save intermediate denoising frames during image generation, allowing a step-by-step visualization of how the model progressively removes noise from a random signal to produce a final image.

## Changes

### `models/stable_diffusion/utils.py`

- `ddpm_sample`: added optional `callback` parameter (`callback(step_idx, total_steps, x)`), called after each of the 1000 reverse steps.
- `ddim_sample`: same `callback` parameter, called after each DDIM step (default 50).

### `models/stable_diffusion/generate.py`

- Added `make_denoise_callback(denoise_dir, nrow, every)` helper that creates and returns a callback saving image grids to `<denoise_dir>/step_NNNN.png`.
- Added two new CLI flags:
  - `--denoise-dir PATH` — directory where per-step frames are written (feature disabled when omitted).
  - `--denoise-every N` — save a frame every N steps (default `1`; useful to thin out DDPM's 1000 steps).

## Usage

```bash
# DDIM: save all 50 steps
python generate.py --checkpoint path/to/ckpt.pt --denoise-dir frames/

# DDPM: save every 10th step out of 1000
python generate.py --checkpoint path/to/ckpt.pt --sampler ddpm \
    --denoise-dir frames/ --denoise-every 10
```

Frames are named `step_0000.png` (noisiest) through `step_NNNN.png` (final image). The last step is always saved regardless of `--denoise-every`.

## Notes

- The callback approach avoids storing all intermediate tensors in memory simultaneously.
- The final output image (`--output`) is saved as before, unaffected by the new flags.
