import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# --------------------------------------------------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        output_dim (int, optional): Output dimesion. Defaults to 128.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        term1 = torch.sin(x_proj)
        term2 = torch.cos(x_proj)
        out = torch.cat([term1, term2], dim=-1)
        return out


# --------------------------------------------------------------------------------------------------
# Stochastic Differential Equations

class BaseSDE():
      """A base class for SDEs. We assume that the SDE is of the form:

    dX_t = f(t, X_t)dt + g(t, X_t)dW_t

    where f and g are the drift and diffusion functions respectively. We assume that the initial distribution is given by p0 at time t=0.

    Args:
        drift (Callable): Drift function
        diffusion (Callable): Diffusion function
    """
      def __init__(self, drift, diff):
          self.drift = drift
          self.diff = diff
      
      def diffusion(self, x, t):
        eps = torch.randn_like(x)
        return x + (self.drift(t) + eps.mT * self.diff(t)).mT


class VPSDE(BaseSDE):
    def __init__(self, beta_min=0.01, beta_max=10.0):
        """
        Variance Preserving Stochastic Differential Equation (VPSDE) class.
        The VPSDE is defined as:
            Drift     -> f(x,t) = -1/2 * beta_t * x
            Diffusion -> g(t)   = sqrt(beta_t)
        """
        drift = lambda t: -0.5 * (beta_min + (beta_max - beta_min) * t)
        diff = lambda t: torch.sqrt(beta_min + (beta_max - beta_min) * t)

        super().__init__(drift, diff)


class VESDE(BaseSDE):
    def __init__(self, sigma_min=0.0001, sigma_max=15.0):
        """
        Variance Exploding Stochastic Differential Equation (VESDE) class.
        The VESDE is defined as:
            Drift     -> f(x,t) = 0
            Diffusion -> g(t)   = sigma^t
        """
        drift = lambda t: torch.zeros_like(t)
        
        _const = torch.sqrt(2 * torch.log(torch.tensor([sigma_max / sigma_min])))
        diff = lambda t: sigma_min * (sigma_max / sigma_min) ** t * _const

        super().__init__(drift, diff)


# --------------------------------------------------------------------------------------------------

class Simformer(nn.Module):
    def __init__(self, timesteps, sde_type="vesde",
                  sigma=[0.0001, 15.0], beta=[0.01, 10.0]):
        """
        Simformer class

        Args:
            timesteps (int): Number of timesteps
            sde_type (str): Type of SDE to use (VESDE or VPSDE)
            sigma (tuple): Sigma values for VESDE
            beta (tuple): Beta values for the VPSDE
        """

        super(Simformer, self).__init__()

        self.normed_t = torch.linspace(0., 1., timesteps)

        self.time_embedding = GaussianFourierEmbedding(64)

        if sde_type == "vesde":
            self.sde = VESDE(sigma[0], sigma[1])
        elif sde_type == "vpsde":
            self.sde = VPSDE(beta[0], beta[1])
        else:
            raise ValueError("Invalid SDE type")


    def forward_diffusion_sample(self, x_0, T, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        t = self.normed_t[T]
        x_1 = self.sde.diffusion(x_0, t)
        return x_1
