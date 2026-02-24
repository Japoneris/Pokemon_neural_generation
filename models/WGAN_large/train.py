"""WGAN-GP training loop for Pokemon generation (v2 with EMA)."""

import argparse
import csv
import os
import sys

import torch
from tqdm import tqdm

# Add parent directory to path to import common module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from common.dataset import get_dataloader
from models import Generator, Discriminator
from utils import compute_gradient_penalty, weights_init, save_image_grid, create_ema, update_ema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use: auto (default), cpu, or cuda")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Total number of epochs (overrides config.NUM_EPOCHS)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for both G and D (overrides config values)")
    args = parser.parse_args()

    num_epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS
    lr_g = args.lr if args.lr is not None else config.LEARNING_RATE_G
    lr_d = args.lr if args.lr is not None else config.LEARNING_RATE_D

    # Setup
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    # Data
    dataloader = get_dataloader(
        root_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        augment=True,
        horizontal_flip_prob=config.HORIZONTAL_FLIP_PROB,
        rotation_degrees=config.ROTATION_DEGREES,
        color_jitter=config.COLOR_JITTER
    )
    print(f"Dataset: {len(dataloader.dataset)} images, {len(dataloader)} batches/epoch")

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # EMA generator
    ema_generator = create_ema(generator)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {g_params:,} params | Discriminator: {d_params:,} params")

    # Optimizers (TTUR: different LR for G and D)
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr_g, betas=(config.BETA1, config.BETA2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(config.BETA1, config.BETA2)
    )

    # Fixed noise for tracking progress
    fixed_noise = torch.randn(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, 1, 1, device=device)

    # Loss log CSV
    loss_log_path = os.path.join(config.OUTPUT_DIR, "losses.csv")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        ema_generator.load_state_dict(checkpoint["ema_generator_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # Override learning rate if --lr was provided
        if args.lr is not None:
            for param_group in optimizer_G.param_groups:
                param_group["lr"] = lr_g
            for param_group in optimizer_D.param_groups:
                param_group["lr"] = lr_d
            print(f"Resumed from epoch {checkpoint['epoch']} with new lr={args.lr}")
        else:
            print(f"Resumed from epoch {checkpoint['epoch']}")

    # Write CSV header if starting fresh (or file doesn't exist)
    if start_epoch == 0 or not os.path.exists(loss_log_path):
        with open(loss_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "d_loss", "g_loss"])

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        g_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # --- Train Discriminator ---
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise).detach()

            d_real = discriminator(real_images).mean()
            d_fake = discriminator(fake_images).mean()
            gp = compute_gradient_penalty(discriminator, real_images, fake_images, device)
            d_loss = d_fake - d_real + config.LAMBDA_GP * gp

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            d_loss_sum += d_loss.item()

            # --- Train Generator every N_CRITIC steps ---
            if (batch_idx + 1) % config.N_CRITIC == 0:
                noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise)
                g_loss = -discriminator(fake_images).mean()

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                g_loss_sum += g_loss.item()
                g_steps += 1

                # Update EMA
                update_ema(ema_generator, generator, config.EMA_DECAY)

            pbar.set_postfix(d_loss=f"{d_loss.item():.4f}", d_real=f"{d_real.item():.4f}")

        # Epoch logging
        avg_d = d_loss_sum / len(dataloader)
        avg_g = g_loss_sum / max(g_steps, 1)
        print(f"Epoch {epoch:4d} | D loss: {avg_d:.4f} | G loss: {avg_g:.4f}")

        # Append to CSV
        with open(loss_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_d:.6f}", f"{avg_g:.6f}"])

        # Save sample grid (using EMA generator)
        if epoch % config.SAMPLE_INTERVAL == 0:
            ema_generator.eval()
            with torch.no_grad():
                fake = ema_generator(fixed_noise)
            save_image_grid(fake, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:04d}.png"))
            ema_generator.train()

        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "ema_generator_state_dict": ema_generator.state_dict(),
                },
                os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pt"),
            )

    # Save final checkpoint if not already saved
    final_epoch = num_epochs - 1
    if final_epoch % config.SAVE_INTERVAL != 0:
        torch.save(
            {
                "epoch": final_epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "ema_generator_state_dict": ema_generator.state_dict(),
            },
            os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{final_epoch:04d}.pt"),
        )
        print(f"Saved final checkpoint at epoch {final_epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()
