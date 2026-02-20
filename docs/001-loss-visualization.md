# 001 - Loss Visualization Script

## Summary

Added `plot_losses.py` to visualize GAN training losses from a `losses.csv` file. Supports `--output-dir` to select the model output folder (v1 or v2).

## Details

The script reads the CSV file containing discriminator (`d_loss`) and generator (`g_loss`) losses per epoch and produces a two-panel plot:

- **Top panel**: Discriminator loss over training steps
- **Bottom panel**: Generator loss over training steps

Each panel shows the raw values (semi-transparent) overlaid with a 20-step rolling average for trend clarity.

The plot is saved as `losses_plot.png` inside the specified output directory. The plot title includes the folder name to identify the model.

## Usage

```bash
source venv/bin/activate
# WGAN_small (default)
python3 plot_losses.py --output-dir outputs/WGAN_small
# WGAN_large
python3 plot_losses.py --output-dir outputs/WGAN_large
# Stable Diffusion
python3 plot_losses.py --output-dir outputs/stable_diffusion
```
