"""Visualize GAN training losses from a losses.csv file."""

import argparse
import csv
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GAN training losses")
    parser.add_argument(
        "--output-dir",
        default="outputs/WGAN_small",
        help="Directory containing losses.csv (default: outputs/WGAN_small)",
    )
    return parser.parse_args()


def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def main():
    args = parse_args()

    csv_path = os.path.join(args.output_dir, "losses.csv")

    # Use a dict keyed by epoch to deduplicate (keep last occurrence)
    data_by_epoch = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            data_by_epoch[epoch] = (float(row["d_loss"]), float(row["g_loss"]))

    # Sort by epoch to get correct x-axis ordering
    sorted_epochs = sorted(data_by_epoch.keys())
    epochs = sorted_epochs
    d_losses = [data_by_epoch[e][0] for e in sorted_epochs]
    g_losses = [data_by_epoch[e][1] for e in sorted_epochs]
    model_name = os.path.basename(os.path.normpath(args.output_dir))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Discriminator loss
    ax1.plot(epochs, d_losses, color="tab:blue", alpha=0.4, linewidth=0.8)
    ax1.plot(epochs, rolling_avg(d_losses), color="tab:blue", linewidth=2, label="D Loss (rolling avg)")
    ax1.set_ylabel("Discriminator Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"GAN Training Losses ({model_name})")

    # Generator loss
    ax2.plot(epochs, g_losses, color="tab:orange", alpha=0.4, linewidth=0.8)
    ax2.plot(epochs, rolling_avg(g_losses), color="tab:orange", linewidth=2, label="G Loss (rolling avg)")
    ax2.set_ylabel("Generator Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "losses_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
