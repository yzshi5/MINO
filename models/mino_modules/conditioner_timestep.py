#from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
#from kappamodules.functional.pos_embed import get_sincos_1d_from_grid
import torch
from torch import nn

def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000):
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    # generate frequencies for sin/cos (e.g. dim=8 -> omega = [1.0, 0.1, 0.01, 0.001])
    omega = 1. / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim).to(grid.device)
    # create grid of frequencies with timesteps
    # Example seqlen=5 dim=8
    # [0, 0.0, 0.00, 0.000]
    # [1, 0.1, 0.01, 0.001]
    # [2, 0.2, 0.02, 0.002]
    # [3, 0.3, 0.03, 0.003]
    # [4, 0.4, 0.04, 0.004]
    # Note: supports cases where grid is more than 1d
    out = grid.unsqueeze(-1) @ omega.unsqueeze(0)
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
        #embed = self.mlp(self.timestep_embed[timestep])
        embed = self.mlp(get_sincos_1d_from_grid(timestep, dim=self.dim))
        return embed
