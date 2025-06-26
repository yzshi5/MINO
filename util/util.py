import torch
import numpy as np
import einops
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt

def reshape_for_batchwise(x, k):
        # need to do some ugly shape-hacking here to get appropriate number of dims
        # maps tensor (n,) to (n, 1, 1, ..., 1) where there are k 1's
        return x.view(-1, *[1]*k)
    
def plot_loss_curve(tr_loss, save_path, te_loss=None, te_epochs=None, logscale=True):
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(tr_loss, label='tr')
    else:
        ax.plot(tr_loss, label='tr')
    if te_loss is not None:
        te_epochs = np.asarray(te_epochs)
        if logscale:
            ax.semilogy(te_epochs-1, te_loss, label='te')  # assume te_epochs is 1-indexed
        else:
            ax.plot(te_epochs-1, te_loss, label='te')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close(fig)

def make_grid(dims, x_min=0, x_max=1):
    """ Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        grid = make_2d_grid(dims)   
    elif len(dims) == 3: 
        grid = make_3d_grid(dims)
        
    return grid


def make_2d_grid(dims, x_min=0, x_max=1):
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1)),
        dim=1)
    return grid

def make_3d_grid(dims, x_min=0, x_max=1):
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x3 = torch.linspace(x_min, x_max, dims[2])
    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1),
        x3.contiguous().view(x3.numel(), 1)),
        dim=1)    
    return grid



class SimDataset(Dataset):
    def __init__(self, data, pos, query_pos):
        super().__init__()
        self.data = data.permute(0, 2, 1) # [batch, n_seq, n_chan]
        self.pos = pos.permute(0, 2, 1)
        self.query_pos = query_pos.unsqueeze(0).repeat(len(data), 1, 1).permute(0, 2, 1)
        
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, idx):
        return dict(input_feat=self.data[idx], input_pos=self.pos[idx], query_pos=self.query_pos[idx])
    
def SimulationCollator(batch):
    
    collated_batch = {}

    # inputs to sparse tensors
    # position: batch_size * (num_inputs, ndim) -> (batch_size * num_inputs, ndim)
    # features: batch_size * (num_inputs, dim) -> (batch_size * num_inputs, dim)
    input_pos = []
    input_query_pos = []
    input_feat = []
    input_lens = []

    batch_size = len(batch)
    for i in range(len(batch)):
        pos = batch[i]["input_pos"]
        query_pos = batch[i]['query_pos']
        feat = batch[i]["input_feat"]
        assert len(pos) == len(pos)
        input_pos.append(pos)
        input_feat.append(feat)
        input_query_pos.append(query_pos)

    x_dim = pos.shape[-1]
    n_chan = feat.shape[-1]

    concat_pos = torch.concat(input_pos)
    concat_query_pos = torch.concat(input_query_pos)
    concat_feat = torch.concat(input_feat)

    collated_batch["input_pos"] = einops.rearrange(concat_pos, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                  batch_size=batch_size,
                                                  dim=x_dim)
    collated_batch["input_feat"] = einops.rearrange(concat_feat, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                   batch_size=batch_size,
                                                   dim=n_chan)
    collated_batch['query_pos']= einops.rearrange(concat_query_pos, "(batch_size seq_len) dim -> batch_size dim seq_len",
                                                  batch_size=batch_size,
                                                  dim=x_dim)
    # inquired positin 
    return collated_batch