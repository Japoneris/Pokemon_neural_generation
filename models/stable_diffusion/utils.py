"""Diffusion utilities, EMA helpers, and image saving."""

import copy
import math

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(num_timesteps, beta_start, beta_end):
    """Linear schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine schedule as proposed in Improved DDPM (Nichol & Dhariwal 2021)."""
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


# ---------------------------------------------------------------------------
# Gaussian diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """Holds the diffusion schedule and provides forward/reverse process methods."""

    def __init__(self, num_timesteps, beta_start, beta_end, schedule="linear"):
        if schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.num_timesteps = num_timesteps
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precomputed quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        """Extract values from 1-D tensor *a* at indices *t*, reshape for broadcasting."""
        out = a.to(t.device).gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # -- Forward process ---------------------------------------------------

    def q_sample(self, x_0, t, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    # -- Training loss -----------------------------------------------------

    def p_losses(self, model, x_0, t):
        """Simplified DDPM loss: MSE between predicted and actual noise."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    # -- Reverse process (DDPM) --------------------------------------------

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """Single DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t)
        mean = sqrt_recip_alpha * (x_t - betas_t / sqrt_one_minus_alpha * predicted_noise)

        if t[0] > 0:
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise
        return mean

    @torch.no_grad()
    def ddpm_sample(self, model, shape, device, seed=None, callback=None):
        """Full DDPM sampling: T steps from pure noise to image.

        Args:
            seed: Optional random seed for reproducible noise generation.
            callback: Optional callable(step_idx, total_steps, x) called after each step.
        """
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(shape, device=device, generator=generator)
        else:
            x = torch.randn(shape, device=device)

        total = self.num_timesteps
        for step_idx, i in enumerate(reversed(range(total))):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if callback is not None:
                callback(step_idx, total, x)
        return x

    # -- Reverse process (DDIM) --------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, ddim_steps=50, eta=0.0, seed=None, callback=None):
        """DDIM sampling: accelerated generation in fewer steps.

        Args:
            seed: Optional random seed for reproducible noise generation.
            callback: Optional callable(step_idx, total_steps, x) called after each step.
        """
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        alphas_cumprod = self.alphas_cumprod.to(device)

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(shape, device=device, generator=generator)
        else:
            x = torch.randn(shape, device=device)

        total = len(timesteps)
        for step_idx, i in enumerate(reversed(range(total))):
            t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
            alpha_t = self._extract(alphas_cumprod, t, x.shape)

            predicted_noise = model(x, t)

            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            if i > 0:
                t_prev = torch.full((shape[0],), timesteps[i - 1], device=device, dtype=torch.long)
                alpha_t_prev = self._extract(alphas_cumprod, t_prev, x.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            sigma = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * predicted_noise
            noise = torch.randn_like(x) if i > 0 else 0
            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + sigma * noise

            if callback is not None:
                callback(step_idx, total, x)

        return x


# ---------------------------------------------------------------------------
# EMA helpers (same as v2)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_image_grid(tensor, path, nrow=8):
    """Save a tensor as an image grid, denormalizing from [-1,1] to [0,1]."""
    save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))
