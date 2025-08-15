# from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
# from kappamodules.functional.pos_embed import get_sincos_1d_from_grid
import torch
from torch import nn


def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000, scale : int = 256):
    # The grid is assumed to be [-1 ,1] or [0, 1] (small)
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    # generate frequencies for sin/cos (e.g. dim=8 -> omega = [1.0, 0.1, 0.01, 0.001])
    omega = 1.0 / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim).to(
        grid.device
    )
    # create grid of frequencies with timesteps
    # Example seqlen=5 dim=8
    # [0, 0.0, 0.00, 0.000]
    # [1, 0.1, 0.01, 0.001]
    # [2, 0.2, 0.02, 0.002]
    # [3, 0.3, 0.03, 0.003]
    # [4, 0.4, 0.04, 0.004]
    # Note: supports cases where grid is more than 1d
    out = scale *grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return torch.concat([emb, padding], dim=-1)


class ConditionerTimestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cond_dim = dim * 4
        self.dim = dim
        self.cond_dim = cond_dim
        """
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=num_timesteps, dim=dim),
        )
        """
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, timestep):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten().double()
        # embed
        # embed = self.mlp(self.timestep_embed[timestep])
        embed = self.mlp(get_sincos_1d_from_grid(timestep, dim=self.dim))
        return embed


import einops
import torch
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim, ndim, max_wavelength: int = 10000, dtype=torch.float32, scale : int = 256):
        super().__init__()
        # assume the domain is small, like [-1, 1]^n or [0, 1]^n, add scale to enhance the performance position encoding
        # by careful of the Nyquist frequency (scale should be too large, otherwise bring aliasing) 
        self.dim = dim
        self.ndim = ndim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1. / max_wavelength ** (torch.arange(0, effective_dim_per_wave, 2, dtype=dtype) / effective_dim_per_wave),
        )
        self.scale = scale

    def forward(self, coords):
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        assert self.ndim == ndim
        out = self.scale * coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        if coords.ndim == 3:
            emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
        elif coords.ndim == 2:
            emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
        else:
            raise NotImplementedError
        emb = emb.to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"
