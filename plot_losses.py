"""Visualize training losses from a losses.csv file (GAN or Stable Diffusion)."""

import argparse
import csv
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training losses")
    parser.add_argument("output_dir",
        help="Directory containing losses.csv"
    )
    return parser.parse_args()


def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def detect_mode(fieldnames):
    if "d_loss" in fieldnames and "g_loss" in fieldnames:
        return "gan"
    elif "loss" in fieldnames:
        return "sd"
    else:
        raise ValueError(f"Unrecognised CSV columns: {fieldnames}")


def plot_gan(epochs, d_losses, g_losses, model_name, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(epochs, d_losses, color="tab:blue", alpha=0.4, linewidth=0.8)
    ax1.plot(epochs, rolling_avg(d_losses), color="tab:blue", linewidth=2, label="D Loss (rolling avg)")
    ax1.set_ylabel("Discriminator Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"GAN Training Losses ({model_name})")

    ax2.plot(epochs, g_losses, color="tab:orange", alpha=0.4, linewidth=0.8)
    ax2.plot(epochs, rolling_avg(g_losses), color="tab:orange", linewidth=2, label="G Loss (rolling avg)")
    ax2.set_ylabel("Generator Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def plot_sd(epochs, losses, model_name, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(epochs, losses, color="tab:purple", alpha=0.4, linewidth=0.8)
    ax.plot(epochs, rolling_avg(losses), color="tab:purple", linewidth=2, label="Loss (rolling avg)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Stable Diffusion Training Loss ({model_name})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def main():
    args = parse_args()

    csv_path = os.path.join(args.output_dir, "losses.csv")
    model_name = os.path.basename(os.path.normpath(args.output_dir))
    out_path = os.path.join(args.output_dir, "losses_plot.png")

    data_by_epoch = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        mode = detect_mode(reader.fieldnames)
        for row in reader:
            epoch = int(row["epoch"])
            if mode == "gan":
                data_by_epoch[epoch] = (float(row["d_loss"]), float(row["g_loss"]))
            else:
                data_by_epoch[epoch] = float(row["loss"])

    sorted_epochs = sorted(data_by_epoch.keys())

    if mode == "gan":
        d_losses = [data_by_epoch[e][0] for e in sorted_epochs]
        g_losses = [data_by_epoch[e][1] for e in sorted_epochs]
        plot_gan(sorted_epochs, d_losses, g_losses, model_name, out_path)
    else:
        losses = [data_by_epoch[e] for e in sorted_epochs]
        plot_sd(sorted_epochs, losses, model_name, out_path)

    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
