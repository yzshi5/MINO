from functools import partial

import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from torch import nn
import math


class DecoderPerceiver(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            ndim,
            dim,
            depth,
            num_heads,
            unbatch_mode="dense_to_sparse_unpadded",
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            init_gate_zero=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.cond_dim = cond_dim
        self.init_weights = init_weights
        self.unbatch_mode = unbatch_mode

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = partial(PerceiverBlock, kv_dim=dim, cond_dim=cond_dim)
        else:
            block_ctor = partial(DitPerceiverBlock, kv_dim=dim, cond_dim=cond_dim)

        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_heads,
                init_weights=init_weights,
                init_gate_zero=init_gate_zero,
            )
            for _ in range(depth)
        ])
        
        # DiT perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        
        self.val_embed = nn.Sequential(
            LinearProjection(output_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )        
     
        if cond_dim is None:
            block_ctor = partial(PerceiverBlock, init_gate_zero=init_gate_zero)
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim, init_gate_zero=init_gate_zero)

        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim*2, perc_dim*2, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim*2, perc_dim, init_weights=init_weights),
        )
        
        # projection
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, output_val, condition=None):
        
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition


        x = self.input_proj(x)

        # Query = Linear(MLP(f_t), P_emb))
        query_pos = self.pos_embed(output_pos)
        query_val = self.val_embed(output_val)
        query = self.query_proj(torch.cat([query_pos, query_val], dim=-1))
        
        # apply the cross attention DiT
        for block in self.blocks:
            x = block(q=query, kv=x, **cond_kwargs)

        x = self.pred(x)
        
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> batch_size dim seqlen",
            )
        elif self.unbatch_mode == "image":
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x
