"""Generate new Pokemon images from a trained checkpoint."""

import argparse

import torch

import config
from models import Generator
from utils import save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Generate Pokemon images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use: auto (default), cpu, or cuda")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load generator
    generator = Generator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # Generate
    with torch.no_grad():
        noise = torch.randn(args.num, config.LATENT_DIM, 1, 1, device=device)
        images = generator(noise)

    # Compute grid dimensions
    nrow = min(8, args.num)
    save_image_grid(images, args.output, nrow=nrow)
    print(f"Saved {args.num} generated images to {args.output}")


if __name__ == "__main__":
    main()
