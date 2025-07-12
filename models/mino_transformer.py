import einops
import torch
from torch import nn

# the code is modified from UPT 
class MINO(nn.Module):
    def __init__(self, encoder,  decoder, conditioner=None):
        super().__init__()
        self.conditioner = conditioner
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            input_feat, 
            input_pos,
            query_pos, # fixed
            timestep=None
    ):        
        
        x_dim = input_pos.shape[1]
        n_chan = input_feat.shape[1]
        batch_size = len(input_feat)

        if self.conditioner is not None:
            if timestep.dim() == 0 or timestep.numel() == 1:
                timestep = torch.ones(batch_size, device=timestep.device) * timestep
            condition = self.conditioner(timestep) # [batch_size, dim*4]
        else:
            condition = None

        output_pos = input_pos.permute(0, 2, 1) 
        output_feat = input_feat.permute(0, 2, 1)
        

        input_pos = einops.rearrange(input_pos, "batch_size dim seq_len -> batch_size seq_len dim ",
                                     batch_size = batch_size,
                                     dim = x_dim)
        input_feat = einops.rearrange(input_feat, "batch_size dim seq_len -> batch_size seq_len dim",
                                      batch_size = batch_size,
                                      dim = n_chan) 
        query_pos = einops.rearrange(query_pos, "batch_size dim seq_len -> batch_size seq_len dim ",
                                     batch_size = batch_size,
                                     dim = x_dim)

        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            query_pos=query_pos,
            condition=condition,
        )

        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            output_val=output_feat,
            condition=condition,
        )

        return pred

    
