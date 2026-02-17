"""Enhanced Generator and Discriminator for WGAN-GP v2.

Improvements over v1:
- Upsample+Conv2d instead of ConvTranspose2d (no checkerboard artifacts)
- Residual blocks after each stage
- Self-attention at 32x32 resolution
- Spectral normalization in discriminator
- Increased capacity (96 base feature maps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SelfAttention(nn.Module):
    """Self-attention layer for capturing long-range spatial dependencies."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(B, -1, H * W)                       # (B, C//8, HW)
        attn = F.softmax(torch.bmm(q, k), dim=-1)                # (B, HW, HW)

        v = self.value(x).view(B, -1, H * W)                     # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN + skip connection."""

    def __init__(self, channels, use_spectral_norm=False):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)
        self.block = nn.Sequential(
            norm_fn(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            norm_fn(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class ResidualBlockDisc(nn.Module):
    """Residual block for discriminator with LayerNorm and LeakyReLU."""

    def __init__(self, channels, spatial_size, use_spectral_norm=True):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)
        self.block = nn.Sequential(
            norm_fn(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.LayerNorm([channels, spatial_size, spatial_size]),
            nn.LeakyReLU(0.2, inplace=True),
            norm_fn(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.LayerNorm([channels, spatial_size, spatial_size]),
        )

    def forward(self, x):
        return F.leaky_relu(self.block(x) + x, 0.2)


class Generator(nn.Module):
    """Enhanced generator: Upsample+Conv, residual blocks, self-attention."""

    def __init__(self, latent_dim=config.LATENT_DIM, ngf=config.GEN_FEATURE_MAPS):
        super().__init__()

        # Project: (latent_dim, 1, 1) -> (ngf*8, 4, 4)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        # Stage 1: 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        self.res1 = ResidualBlock(ngf * 4)

        # Stage 2: 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.res2 = ResidualBlock(ngf * 2)

        # Stage 3: 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.res3 = ResidualBlock(ngf)

        # Self-attention at 32x32
        self.attn = SelfAttention(ngf)

        # Stage 4: 32x32 -> 64x64
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, config.NUM_CHANNELS, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.project(z)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.attn(x)
        return self.up4(x)


class Discriminator(nn.Module):
    """Enhanced discriminator: spectral norm, residual blocks, self-attention."""

    def __init__(self, ndf=config.DISC_FEATURE_MAPS):
        super().__init__()
        sn = nn.utils.spectral_norm

        # (3, 64, 64) -> (ndf, 32, 32)
        self.down1 = nn.Sequential(
            sn(nn.Conv2d(config.NUM_CHANNELS, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResidualBlockDisc(ndf, 32)

        # Self-attention at 32x32
        self.attn = SelfAttention(ndf)

        # (ndf, 32, 32) -> (ndf*2, 16, 16)
        self.down2 = nn.Sequential(
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResidualBlockDisc(ndf * 2, 16)

        # (ndf*2, 16, 16) -> (ndf*4, 8, 8)
        self.down3 = nn.Sequential(
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResidualBlockDisc(ndf * 4, 8)

        # (ndf*4, 8, 8) -> (ndf*8, 4, 4)
        self.down4 = nn.Sequential(
            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResidualBlockDisc(ndf * 8, 4)

        # (ndf*8, 4, 4) -> (1, 1, 1)
        self.out = nn.Sequential(
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.res1(self.down1(x))
        x = self.attn(x)
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        x = self.res4(self.down4(x))
        return self.out(x)
