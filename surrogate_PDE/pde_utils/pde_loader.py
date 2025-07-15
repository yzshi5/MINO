import torch
import numpy as np
import einops
from torch.utils.data import TensorDataset, DataLoader, Dataset

class SimDataset_PDE(Dataset):
    def __init__(self, input_data, output_data, pos, query_pos):
        super().__init__()
        self.input_data = input_data.permute(0, 2, 1) # [batch, n_seq, n_chan]
        self.output_data = output_data.permute(0, 2, 1) # [batch, n_seq, n_chan]
        self.pos = pos.permute(0, 2, 1)
        self.query_pos = query_pos.unsqueeze(0).repeat(len(pos), 1, 1).permute(0, 2, 1)
        
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, idx):
        return dict(input_feat=self.input_data[idx], output_feat=self.output_data[idx], input_pos=self.pos[idx], query_pos=self.query_pos[idx])
    
def SimulationCollator_PDE(batch):
    
    collated_batch = {}

    # inputs to sparse tensors
    # position: batch_size * (num_inputs, ndim) -> (batch_size * num_inputs, ndim)
    # features: batch_size * (num_inputs, dim) -> (batch_size * num_inputs, dim)
    input_pos = []
    input_query_pos = []
    input_feat = []
    input_lens = []
    output_feat = []

    batch_size = len(batch)
    for i in range(len(batch)):
        pos = batch[i]["input_pos"]
        query_pos = batch[i]['query_pos']
        feat = batch[i]["input_feat"]
        out_feat = batch[i]["output_feat"]
        assert len(pos) == len(pos)
        input_pos.append(pos)
        input_feat.append(feat)
        output_feat.append(out_feat)
        input_query_pos.append(query_pos)

    x_dim = pos.shape[-1]
    n_chan = feat.shape[-1]
    n_chan_out = out_feat.shape[-1]

    concat_pos = torch.concat(input_pos)
    concat_query_pos = torch.concat(input_query_pos)
    concat_feat = torch.concat(input_feat)
    concat_out_feat = torch.concat(output_feat)

    collated_batch["input_pos"] = einops.rearrange(concat_pos, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                  batch_size=batch_size,
                                                  dim=x_dim)
    collated_batch["input_feat"] = einops.rearrange(concat_feat, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                   batch_size=batch_size,
                                                   dim=n_chan)
    collated_batch["output_feat"] = einops.rearrange(concat_out_feat, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                   batch_size=batch_size,
                                                   dim=n_chan_out)
    collated_batch['query_pos']= einops.rearrange(concat_query_pos, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                  batch_size=batch_size,
                                                  dim=x_dim)
    # inquired positin 
    return collated_batch