"""Utility functions: gradient penalty, weight init, EMA, image saving."""

import copy

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
    """DCGAN-style weight initialization (skips spectral-normed layers)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Spectral norm wraps the weight, so check for weight_orig
        if hasattr(m, 'weight_orig'):
            return
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def create_ema(model):
    """Create an EMA copy of a model."""
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def update_ema(ema_model, model, decay):
    """Update EMA weights: ema = decay * ema + (1 - decay) * model."""
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(model_p, alpha=1.0 - decay)


def save_image_grid(tensor, path, nrow=8):
    """Save a tensor as an image grid, denormalizing from [-1,1] to [0,1]."""
    save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))
