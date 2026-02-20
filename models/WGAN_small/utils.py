"""Utility functions: gradient penalty, weight init, image saving."""

import torch
import torch.nn as nn
from torchvision.utils import save_image


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def weights_init(m):
    """DCGAN-style weight initialization."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def save_image_grid(tensor, path, nrow=8):
    """Save a tensor as an image grid, denormalizing from [-1,1] to [0,1]."""
    save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))
