"""Pokemon dataset with RGBA compositing and augmentation.

This is a shared dataset module used by all models (WGAN_small, WGAN_large, stable_diffusion).
"""

import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PokemonDataset(Dataset):
    """Loads Pokemon PNGs, composites RGBA onto white, applies transforms."""

    def __init__(self, root_dir, image_size, augment=True, 
                 horizontal_flip_prob=0.5, rotation_degrees=15, 
                 color_jitter=(0.2, 0.2, 0.2, 0.1)):
        """
        Args:
            root_dir: Directory containing Pokemon PNG images
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
            horizontal_flip_prob: Probability of horizontal flip
            rotation_degrees: Max rotation angle in degrees
            color_jitter: Tuple of (brightness, contrast, saturation, hue) jitter values
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.file_paths = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )

        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if augment:
            transform_list += [
                transforms.RandomHorizontalFlip(horizontal_flip_prob),
                transforms.RandomRotation(rotation_degrees, fill=(255, 255, 255)),
                transforms.ColorJitter(*color_jitter),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, img)
        rgb = composited.convert("RGB")
        return self.transform(rgb)


def get_dataloader(root_dir, image_size, batch_size, augment=True,
                   horizontal_flip_prob=0.5, rotation_degrees=15,
                   color_jitter=(0.2, 0.2, 0.2, 0.1)):
    """Create a DataLoader for the Pokemon dataset.
    
    Args:
        root_dir: Directory containing Pokemon PNG images
        image_size: Target image size (square)
        batch_size: Batch size for training
        augment: Whether to apply data augmentation
        horizontal_flip_prob: Probability of horizontal flip
        rotation_degrees: Max rotation angle in degrees
        color_jitter: Tuple of (brightness, contrast, saturation, hue) jitter values
    
    Returns:
        DataLoader instance
    """
    dataset = PokemonDataset(
        root_dir, 
        image_size, 
        augment=augment,
        horizontal_flip_prob=horizontal_flip_prob,
        rotation_degrees=rotation_degrees,
        color_jitter=color_jitter
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
