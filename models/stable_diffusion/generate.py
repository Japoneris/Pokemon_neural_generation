"""Generate new Pokemon images from a trained DDPM checkpoint."""

import argparse
import os
from pathlib import Path

import torch

import config
from models import UNet
from utils import GaussianDiffusion, save_image_grid


def make_denoise_callback(denoise_dir, nrow, every=1):
    """Return a callback that saves a grid image every *every* denoising steps.

    The callback signature matches what ddim_sample / ddpm_sample expect:
        callback(step_idx, total_steps, x)
    Frames are saved as  <denoise_dir>/step_<NNNN>.png  (step_idx 0 = noisiest).
    """
    os.makedirs(denoise_dir, exist_ok=True)

    def callback(step_idx, total_steps, x):
        if step_idx % every != 0 and step_idx != total_steps - 1:
            return
        path = os.path.join(denoise_dir, f"step_{step_idx:04d}.png")
        save_image_grid(x, path, nrow=nrow)

    return callback


def main():
    parser = argparse.ArgumentParser(description="Generate Pokemon images (v3 DDPM)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model instead of EMA")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use: auto (default), cpu, or cuda")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"],
                        help="Sampling method: ddim (fast, default) or ddpm (slow, 1000 steps)")
    parser.add_argument("--ddim-steps", type=int, default=config.DDIM_STEPS,
                        help="Number of DDIM steps (default: 50)")
    parser.add_argument("--ddim-eta", type=float, default=config.DDIM_ETA,
                        help="DDIM eta: 0.0=deterministic, 1.0=stochastic (default: 0.0)")
    parser.add_argument("--denoise-dir", type=str, default=None,
                        help="Directory to save per-step denoising frames (disabled by default)")
    parser.add_argument("--denoise-every", type=int, default=1,
                        help="Save a frame every N denoising steps (default: 1)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Auto-generate output filename if not provided
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        epoch_str = checkpoint_name.replace('checkpoint_', '')
        model_type = "ema" if not args.no_ema else "raw"
        if args.sampler == "ddim":
            args.output = f"generated_{epoch_str}_{model_type}_{args.sampler}_steps{args.ddim_steps}_eta{args.ddim_eta}_num{args.num}.png"
        else:
            args.output = f"generated_{epoch_str}_{model_type}_{args.sampler}_steps1000_num{args.num}.png"

    # Load model
    model = UNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if not args.no_ema and "ema_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded raw model weights")
    model.eval()

    # Diffusion
    diffusion = GaussianDiffusion(
        config.NUM_TIMESTEPS, config.BETA_START, config.BETA_END, config.BETA_SCHEDULE
    )

    shape = (args.num, config.NUM_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    nrow = min(8, args.num)
    print(f"Generating {args.num} images with {args.sampler} sampler...")

    callback = None
    if args.denoise_dir is not None:
        callback = make_denoise_callback(args.denoise_dir, nrow, every=args.denoise_every)
        print(f"Saving denoising frames to '{args.denoise_dir}' every {args.denoise_every} step(s)")

    if args.sampler == "ddim":
        images = diffusion.ddim_sample(model, shape, device, args.ddim_steps, args.ddim_eta,
                                       seed=args.seed, callback=callback)
    else:
        images = diffusion.ddpm_sample(model, shape, device, seed=args.seed, callback=callback)

    save_image_grid(images, args.output, nrow=nrow)
    print(f"Saved {args.num} generated images to {args.output}")
    if args.denoise_dir is not None:
        print(f"Denoising frames saved in '{args.denoise_dir}'")


if __name__ == "__main__":
    main()
