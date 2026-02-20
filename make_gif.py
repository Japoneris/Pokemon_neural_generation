#!/usr/bin/env python3
"""Create a GIF from images in a folder, taking every k-th image."""

import argparse
import glob
import os

from PIL import Image


def make_gif(folder: str, k: int, output: str, duration: int):
    """Create a GIF by selecting every k-th image from sorted files in folder."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()

    if not files:
        print(f"No images found in {folder}")
        return

    selected = files[::k]
    print(f"Found {len(files)} images, selecting every {k}-th -> {len(selected)} frames")

    frames = [Image.open(f) for f in selected]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from a folder of images")
    parser.add_argument("folder", help="Folder containing the images")
    parser.add_argument("-k", type=int, default=10, help="Take every k-th image (default: 10)")
    parser.add_argument("-o", "--output", default=None, help="Output GIF path (default: <folder>/animation.gif)")
    parser.add_argument("-d", "--duration", type=int, default=200, help="Duration per frame in ms (default: 200)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.folder, "animation.gif")
    make_gif(args.folder, args.k, output, args.duration)
