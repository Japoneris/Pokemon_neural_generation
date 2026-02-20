"""Plot training losses from the CSV log."""

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot WGAN-GP v2 training losses")
    parser.add_argument("--input", type=str, default="../outputs_v2/losses.csv", help="Path to losses CSV")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    epochs, d_losses, g_losses = [], [], []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            d_losses.append(float(row["d_loss"]))
            g_losses.append(float(row["g_loss"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(epochs, d_losses, linewidth=0.8)
    ax1.set_ylabel("Discriminator Loss")
    ax1.set_title("WGAN-GP v2 Training Losses")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, g_losses, linewidth=0.8, color="tab:orange")
    ax2.set_ylabel("Generator Loss")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
