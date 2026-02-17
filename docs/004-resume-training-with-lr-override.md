# 004 - Resume Training with Learning Rate Override

## Problem

After training for 1000 epochs with `lr=1e-4`, the generated images are blurry. Continuing training with a lower learning rate can help sharpen results without starting from scratch.

## Changes

Added two CLI flags to `train.py`:

- `--epochs N` — Override the total number of epochs (default: `config.NUM_EPOCHS`). This allows extending training beyond the original 1000 epochs when resuming.
- `--lr RATE` — Override the learning rate for both Generator and Discriminator. When used with `--resume`, the optimizer states are loaded first, then the learning rate is overwritten.

## Usage

Resume from the last checkpoint with a lower learning rate and train for 500 more epochs:

```bash
cd model_v1
python3 train.py --resume ../outputs/checkpoints/checkpoint_epoch_0950.pt --lr 2e-5 --epochs 2000
```

This loads the model and optimizer state from epoch 950, sets both learning rates to `2e-5`, and trains until epoch 2000.
