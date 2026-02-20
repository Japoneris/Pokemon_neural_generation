# GIF Generation Script

## Summary

Added `make_gif.py` — a CLI script that creates an animated GIF from a folder of images, selecting every k-th image.

## Usage

```bash
python3 make_gif.py <folder> [-k K] [-o OUTPUT] [-d DURATION]
```

- `folder`: path to the directory containing images
- `-k`: take every k-th image (default: 10)
- `-o`: output GIF path (default: `<folder>/animation.gif`)
- `-d`: duration per frame in milliseconds (default: 200)

## Example

```bash
python3 make_gif.py outputs/samples/ -k 10
```

Creates a 21-frame GIF from 202 training sample images.
