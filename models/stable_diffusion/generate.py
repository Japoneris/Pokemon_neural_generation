"""Generate new Pokemon images from a trained DDPM checkpoint."""

import argparse
import os
from pathlib import Path

import torch

import config
from models import UNet
from utils import GaussianDiffusion, save_image_grid


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
    print(f"Generating {args.num} images with {args.sampler} sampler...")

    if args.sampler == "ddim":
        images = diffusion.ddim_sample(model, shape, device, args.ddim_steps, args.ddim_eta)
    else:
        images = diffusion.ddpm_sample(model, shape, device)

    nrow = min(8, args.num)
    save_image_grid(images, args.output, nrow=nrow)
    print(f"Saved {args.num} generated images to {args.output}")


if __name__ == "__main__":
    main()
