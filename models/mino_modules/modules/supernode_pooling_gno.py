import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from torch import nn
from torch_geometric.nn.pool import radius_graph
from torch_scatter import segment_csr
from .gno_block import GNOBlock
import torch.nn.functional as F

class SupernodePooling(nn.Module):
    def __init__(
            self,
            radius,
            input_dim,
            hidden_dim,
            ndim,
            init_weights="torch",
    ):
        super().__init__()
        self.radius = radius
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights
        
        self.message = GNOBlock(
            in_channels=input_dim,
            out_channels=hidden_dim,
            coord_dim=ndim,
            pos_embedding_type='transformer',
            pos_embedding_channels=hidden_dim,
            pos_embedding_max_positions=10000,
            radius=radius,
            channel_mlp_layers=[hidden_dim*2, hidden_dim],
            channel_mlp_non_linearity=F.gelu,
            transform_type="nonlinear_kernelonly",
            use_torch_scatter_reduce=True,
            use_open3d_neighbor_search=False,
        )
                
        self.output_dim = hidden_dim
    
    def forward(self, input_feat, input_pos, query_pos):
        
        x = self.message(y=input_pos[0], x=query_pos[0], f_y=input_feat)
    
        return x


