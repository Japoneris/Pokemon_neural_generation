"""UNet noise-prediction network for DDPM."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding for integer timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal encoding -> MLP projection."""

    def __init__(self, model_channels, time_emb_dim):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(model_channels)
        self.mlp = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def group_norm(channels):
    """GroupNorm with at most 32 groups."""
    return nn.GroupNorm(min(32, channels), channels)


class ResBlock(nn.Module):
    """Residual block with time-embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    """Strided convolution downsampling (halves spatial dims)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample + conv (doubles spatial dims)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """UNet noise-prediction model with time conditioning.

    Architecture for 64x64 images with channel_mult=(1,2,4,8):
        Encoder: 64→32→16→8  (channels: 64→128→256→512)
        Middle:  8×8, 512ch
        Decoder: 8→16→32→64  (symmetric with skip connections)
    """

    def __init__(
        self,
        in_channels=config.NUM_CHANNELS,
        model_channels=config.MODEL_CHANNELS,
        out_channels=config.NUM_CHANNELS,
        channel_mult=config.CHANNEL_MULT,
        num_res_blocks=config.NUM_RES_BLOCKS,
        attention_resolutions=config.ATTENTION_RESOLUTIONS,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        time_emb_dim=config.TIME_EMB_DIM,
    ):
        super().__init__()

        self.time_embedding = TimeEmbedding(model_channels, time_emb_dim)
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Track channels at each skip-connection point
        self._build_encoder(model_channels, channel_mult, num_res_blocks,
                            attention_resolutions, num_heads, dropout, time_emb_dim)
        self._build_middle(time_emb_dim, num_heads, dropout)
        self._build_decoder(model_channels, channel_mult, num_res_blocks,
                            attention_resolutions, num_heads, dropout, time_emb_dim)

        self.out = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    # -- Encoder -----------------------------------------------------------

    def _build_encoder(self, model_channels, channel_mult, num_res_blocks,
                       attention_resolutions, num_heads, dropout, time_emb_dim):
        self.down_blocks = nn.ModuleList()
        self.down_block_types = []  # "res", "attn", "down"
        self.skip_channels = [model_channels]  # from input_conv

        ch = model_channels
        resolution = config.IMAGE_SIZE  # 64

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                self.down_block_types.append("res")
                ch = out_ch
                if resolution in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch, num_heads))
                    self.down_block_types.append("attn")
                self.skip_channels.append(ch)

            if level < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                self.down_block_types.append("down")
                self.skip_channels.append(ch)
                resolution //= 2

        self._bottleneck_ch = ch

    # -- Middle ------------------------------------------------------------

    def _build_middle(self, time_emb_dim, num_heads, dropout):
        ch = self._bottleneck_ch
        self.middle_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.middle_attn = AttentionBlock(ch, num_heads)
        self.middle_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

    # -- Decoder -----------------------------------------------------------

    def _build_decoder(self, model_channels, channel_mult, num_res_blocks,
                       attention_resolutions, num_heads, dropout, time_emb_dim):
        self.up_blocks = nn.ModuleList()
        self.up_block_types = []  # "res", "attn", "up"

        ch = self._bottleneck_ch
        resolution = config.IMAGE_SIZE // (2 ** (len(channel_mult) - 1))

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            # num_res_blocks + 1 to account for skip concat from encoder
            for i in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                self.up_block_types.append("res")
                ch = out_ch
                if resolution in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch, num_heads))
                    self.up_block_types.append("attn")

            if level > 0:
                self.up_blocks.append(Upsample(ch))
                self.up_block_types.append("up")
                resolution *= 2

    # -- Forward -----------------------------------------------------------

    def forward(self, x, t):
        """
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) integer timesteps
        Returns:
            (B, C, H, W) predicted noise
        """
        t_emb = self.time_embedding(t)
        h = self.input_conv(x)
        skips = [h]

        # Encoder
        for block, btype in zip(self.down_blocks, self.down_block_types):
            if btype == "res":
                h = block(h, t_emb)
                skips.append(h)
            elif btype == "attn":
                h = block(h)
                skips[-1] = h  # update last skip with post-attention features
            else:  # "down"
                h = block(h)
                skips.append(h)

        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        # Decoder
        for block, btype in zip(self.up_blocks, self.up_block_types):
            if btype == "res":
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
            else:  # "attn" or "up"
                h = block(h)

        return self.out(h)
