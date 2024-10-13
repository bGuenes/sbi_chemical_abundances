import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GaussianFourierEmbedding():
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        output_dim (int, optional): Output dimesion. Defaults to 128.
    """

    def __init__(self, output_dim=128):
        self.output_dim = output_dim

    def __call__(self, inputs):
        term1 = torch.cos(2 * np.pi * inputs)
        term2 = torch.sin(2 * np.pi * inputs)
        out = torch.cat([term1, term2], dim=-1)

        return out[..., : self.output_dim]


class Simformer(nn.Module):
    def __init__(self, timesteps):
        super(Simformer, self).__init__()

        self.betas = self.linear_beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)