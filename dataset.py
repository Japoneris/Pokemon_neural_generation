"""Pokemon dataset with RGBA compositing and augmentation."""

import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class PokemonDataset(Dataset):
    """Loads Pokemon PNGs, composites RGBA onto white, applies transforms."""

    def __init__(self, root_dir, image_size=config.IMAGE_SIZE, augment=True):
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
                transforms.RandomHorizontalFlip(config.HORIZONTAL_FLIP_PROB),
                transforms.RandomRotation(config.ROTATION_DEGREES),
                transforms.ColorJitter(*config.COLOR_JITTER),
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


def get_dataloader(root_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, augment=True):
    """Create a DataLoader for the Pokemon dataset."""
    dataset = PokemonDataset(root_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
