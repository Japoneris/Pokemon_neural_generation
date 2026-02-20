"""DDPM training loop for Pokemon generation (v3)."""

import argparse
import csv
import math
import os

import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from models import UNet
from utils import GaussianDiffusion, create_ema, update_ema, save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon DDPM v3")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use: auto (default), cpu, or cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override NUM_EPOCHS")
    parser.add_argument("--lr", type=float, default=None, help="Override LEARNING_RATE")
    args = parser.parse_args()

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

    num_epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS
    lr = args.lr if args.lr is not None else config.LEARNING_RATE

    # Data
    dataloader = get_dataloader()
    print(f"Dataset: {len(dataloader.dataset)} images, {len(dataloader)} batches/epoch")

    # Model
    model = UNet().to(device)
    ema_model = create_ema(model)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"UNet: {param_count:,} params")

    # Diffusion
    diffusion = GaussianDiffusion(
        config.NUM_TIMESTEPS, config.BETA_START, config.BETA_END, config.BETA_SCHEDULE
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2),
    )

    # LR scheduler: linear warmup + cosine annealing
    total_steps = num_epochs * len(dataloader)

    def lr_lambda(step):
        if step < config.WARMUP_STEPS:
            return step / max(1, config.WARMUP_STEPS)
        progress = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss log CSV
    loss_log_path = os.path.join(config.OUTPUT_DIR, "losses.csv")

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Write CSV header if starting fresh
    if start_epoch == 0 or not os.path.exists(loss_log_path):
        with open(loss_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss"])

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        loss_sum = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Sample random timesteps
            t = torch.randint(0, config.NUM_TIMESTEPS, (batch_size,), device=device)

            # Compute loss
            loss = diffusion.p_losses(model, real_images, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # Update EMA
            update_ema(ema_model, model, config.EMA_DECAY)

            loss_sum += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Epoch logging
        avg_loss = loss_sum / len(dataloader)
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

        with open(loss_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}"])

        # Sample generation using EMA + DDIM
        if epoch % config.SAMPLE_INTERVAL == 0:
            ema_model.eval()
            samples = diffusion.ddim_sample(
                ema_model,
                (config.NUM_SAMPLE_IMAGES, config.NUM_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
                device,
                ddim_steps=config.DDIM_STEPS,
            )
            save_image_grid(samples, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:04d}.png"))
            ema_model.train()

        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pt"),
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
