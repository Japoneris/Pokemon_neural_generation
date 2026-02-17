"""Generator and Discriminator (Critic) for WGAN-GP."""

import torch.nn as nn

import config


class Generator(nn.Module):
    """Maps latent vector z to a 64x64 RGB image."""

    def __init__(self, latent_dim=config.LATENT_DIM, ngf=config.GEN_FEATURE_MAPS):
        super().__init__()
        self.main = nn.Sequential(
            # (latent_dim, 1, 1) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(ngf, config.NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Maps a 64x64 RGB image to a scalar critic score (no sigmoid)."""

    def __init__(self, ndf=config.DISC_FEATURE_MAPS):
        super().__init__()
        self.main = nn.Sequential(
            # (3, 64, 64) -> (ndf, 32, 32)
            nn.Conv2d(config.NUM_CHANNELS, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 32, 32) -> (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16) -> (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8) -> (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4) -> (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)
