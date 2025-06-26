from functools import partial

from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from mino_modules.modules.supernode_pooling_gno import SupernodePooling
from torch import nn


class EncoderSupernodes(nn.Module):
    def __init__(
            self,
            input_dim,
            ndim,
            radius,
            enc_dim,
            enc_depth,
            enc_num_heads,
            cond_dim=None,
            init_weights="truncnormal",
            init_gate_zero=False, # set it as true
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.condition_dim = cond_dim
        self.init_weights = init_weights
        self.init_gate_zero=init_gate_zero

        # supernode pooling, use nonlinear kernel GNO (modified from GINO code) 
        self.supernode_pooling = SupernodePooling(
            radius=radius,
            input_dim=input_dim,
            hidden_dim=enc_dim,
            ndim=ndim,
        ) 
        
        # blocks
        self.enc_proj = LinearProjection(enc_dim, enc_dim, init_weights=init_weights, optional=True)
        
        # percieverBlock and DiT perceiver Block
        if cond_dim is None:
            block_ctor = partial(PerceiverBlock, kv_dim=enc_dim, init_weights=init_weights, init_gate_zero=init_gate_zero)
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim, kv_dim=enc_dim, 
                                 init_weights=init_weights, init_gate_zero=init_gate_zero)
            
        self.first_block = block_ctor(dim=enc_dim, num_heads=enc_num_heads)
        
        self.rest_block = nn.ModuleList(
            block_ctor(dim=enc_dim, num_heads=enc_num_heads)
            for _ in range(enc_depth-1))
        

    def forward(self, input_feat, input_pos, query_pos, condition=None):

        assert len(input_feat) == len(input_pos), "expected input_feat and input_pos to have same length"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            query_pos=query_pos
        )
   
        # project to encoder dimension
        x = self.enc_proj(x)                            #"batch_size seqlen dim 

        h = self.first_block(kv=x, q=x, **cond_kwargs)  # [B, L, enc_dim]

        for blk in self.rest_block:                      # N-1 times
            h = blk(q=h, kv=x, **cond_kwargs)
            
        return h