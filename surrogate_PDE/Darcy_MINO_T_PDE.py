#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

## load path
import os
import sys

sys.path.append('../')
sys.path.append('../models')
from pathlib import Path

## load utils
from util.util import *
import time
import scipy.io as scio
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

# load PDE utils
from pde_utils.testloss import TestLoss
from pde_utils.normalizer import UnitTransformer
from pde_utils.pde_loader import *

## load modules
from models.mino_transformer import MINO
from models.mino_modules.decoder_perceiver import DecoderPerceiver
from models.mino_modules.encoder_supernodes_gno_cross_attention import EncoderSupernodes


# ## Parameters

# In[2]:
plt.plot([1, 2, 3, 4])

torch.cuda.is_available()


# In[4]:
# print current hostname (computation node)
import socket

print(socket.gethostname())


# In[10]:


dims = [85, 85]
query_dims = [16, 16]
x_dim = 2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
spath = Path('/home/yshi/PDE_solving/dataset/saved/MINO_T_Darcy_PDE')

spath.mkdir(parents=True, exist_ok=True)
saved_model = False  # True # save model
save_int = 500

# model hyperparameters
## conditional time step..
dim = 256
num_heads = 4

## training parameters
epochs = 500
batch_size = 4


# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# ### Dataset and Dataloder
#
# `x_train` : Input function, contains N samples, with shape (N, n_chan, n_seq), where `n_seq` is the number of discretizations
#
# `y_train` : output function, with shape (N, n_chan_2, n_seq)
#
# `pos_data` : (N, x_dim, n_seq), `x_dim` is the dimension for the domain, for 2D `x_dim` = 2
#
# `query_pos` : inquired position for GNO, has a shape of (x_dim, n_node). n_node << n_seq.
#

# In[5]:


# datapath
data_path = Path('/home/yshi/PDE_solving/dataset/MINO_PDE/Darcy_421')
train_path = data_path / 'piececonst_r421_N1024_smooth1.mat'
test_path = data_path / 'piececonst_r421_N1024_smooth2.mat'


# In[6]:


#### We follow the pipeline of Transolver for processing the Darcy flow data
## the resolution is [85, 85]

ntrain = 1000
ntest = 200
downsample = 5

r = downsample
h = int(((421 - 1) / r) + 1)  # 85
s = h
dx = 1.0 / s

train_data = scio.loadmat(train_path)
x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
x_train = torch.from_numpy(x_train).float()
x_train = torch.flatten(x_train.unsqueeze(1), start_dim=2)

y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
y_train = torch.from_numpy(y_train)
y_train = torch.flatten(y_train.unsqueeze(1), start_dim=2)

test_data = scio.loadmat(test_path)
x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
x_test = torch.from_numpy(x_test).float()
x_test = torch.flatten(x_test.unsqueeze(1), start_dim=2)

y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
y_test = torch.from_numpy(y_test)
y_test = torch.flatten(y_test.unsqueeze(1), start_dim=2)

x_normalizer = UnitTransformer(x_train)
y_normalizer = UnitTransformer(y_train)

x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_train = y_normalizer.encode(y_train)

# y_test is not encoded

x_normalizer.to(device)
y_normalizer.to(device)

x = np.linspace(0, 1, s)
y = np.linspace(0, 1, s)
x, y = np.meshgrid(x, y)
pos = np.c_[x.ravel(), y.ravel()]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
pos = pos.permute(0, 2, 1)

pos_train = pos.repeat(ntrain, 1, 1)
pos_test = pos.repeat(ntest, 1, 1)

## latent position
query_pos = make_2d_grid(query_dims).permute(1, 0)  # [2, 16x16]

train_dataset = SimDataset_PDE(
    input_data=x_train, output_data=y_train, pos=pos_train, query_pos=query_pos
)
test_dataset = SimDataset_PDE(
    input_data=x_test, output_data=y_test, pos=pos_test, query_pos=query_pos
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=SimulationCollator_PDE,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=SimulationCollator_PDE,
)
print('Dataloading is over.')


# ## Model Initialization
# %%

# %%

model = MINO(
    encoder=EncoderSupernodes(
        input_dim=1,  # co-domain
        ndim=2,  # dimension of domain
        radius=0.07,
        enc_dim=dim,
        enc_num_heads=num_heads,
        enc_depth=5,
    ),
    decoder=DecoderPerceiver(
        input_dim=dim,
        output_dim=1,
        ndim=2,
        dim=dim,
        num_heads=num_heads,
        depth=2,  # 2 layers
        unbatch_mode='dense_to_sparse_unpadded',
    ),
)
model = model.to(device)
print(f'parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

# %%

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

myloss = TestLoss(size_average=False)
de_x = TestLoss(size_average=False)
de_y = TestLoss(size_average=False)

# In[11]:


for ep in range(1, epochs + 1):
    model.train()
    train_loss = 0
    reg = 0
    test_losses = []
    for batch_pack in train_loader:
        batch = batch_pack['input_feat'].to(device)  # [batch_size, n_chan, n_seq]
        pos = batch_pack['input_pos'].to(device)  # [batch_size, x_dim, n_seq]
        query_pos = batch_pack['query_pos'].to(device)
        out_batch = batch_pack['output_feat'].to(device)

        optimizer.zero_grad()

        out = model(input_feat=batch, input_pos=pos, query_pos=query_pos)
        out = y_normalizer.decode(out)  # 'b, c, (h w)'
        out_batch = y_normalizer.decode(out_batch)  # 'b, c, (h w)'

        l2loss = myloss(out, out_batch)

        # derivative loss for Darcy Flow dataset
        """
        out = rearrange(out, 'b c (h w) -> b c h w', h=s)
        out = out[..., 1:-1, 1:-1].contiguous()
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = rearrange(out, 'b c h w -> b c (h w)')
        gt_grad_x, gt_grad_y = central_diff(out_batch, dx, s)
        pred_grad_x, pred_grad_y = central_diff(out, dx, s)
        deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
        loss = 0.1 * deriv_loss + l2loss
        loss.backward()   
        """
        l2loss.backward()

        optimizer.step()
        train_loss += l2loss.item()
        # reg += deriv_loss.item()
    scheduler.step()

    train_loss /= ntrain
    reg /= ntrain
    print('Epoch {} Reg : {:.5f} Train loss : {:.5f}'.format(ep, reg, train_loss))

    if ep % 5 == 0:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Show input batch
        im0 = axs[0].imshow(batch[0, 0].reshape(85, 85).detach().cpu().numpy())
        axs[0].set_title('Input')
        fig.colorbar(im0, ax=axs[0])

        # Show model output
        im1 = axs[1].imshow(out[0, 0].reshape(85, 85).detach().cpu().numpy())
        axs[1].set_title(f'Epoch {ep} - Predicted')
        fig.colorbar(im1, ax=axs[1])

        # Show ground truth
        im2 = axs[2].imshow(out_batch[0, 0].reshape(85, 85).detach().cpu().numpy())
        axs[2].set_title(f'Epoch {ep} - Ground Truth')
        fig.colorbar(im2, ax=axs[2])

        # Save figure
        plt.tight_layout()
        plt.savefig(spath / f'epoch_{ep}_train.png')
        plt.close(fig)

    model.eval()
    rel_err = 0.0
    with torch.no_grad():
        for batch_pack in test_loader:
            batch = batch_pack['input_feat'].to(device)  # [batch_size, n_chan, n_seq]
            pos = batch_pack['input_pos'].to(device)  # [batch_size, x_dim, n_seq]
            query_pos = batch_pack['query_pos'].to(device)
            out_batch = batch_pack['output_feat'].to(device)

            out = model(input_feat=batch, input_pos=pos, query_pos=query_pos)
            out = y_normalizer.decode(out)  # 'b, c, (h w)'

            tl = myloss(out, out_batch).item()
            rel_err += tl

        if ep % 5 == 0:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Show model output
            im0 = axs[0].imshow(out[0, 0].reshape(85, 85).detach().cpu().numpy())
            axs[0].set_title(f'Epoch {ep} - Predicted')
            fig.colorbar(im0, ax=axs[0])

            # Show ground truth
            im1 = axs[1].imshow(out_batch[0, 0].reshape(85, 85).detach().cpu().numpy())
            axs[1].set_title(f'Epoch {ep} - Ground Truth')
            fig.colorbar(im1, ax=axs[1])

            # Save figure
            plt.tight_layout()
            plt.savefig(spath / f'epoch_{ep}_val.png')
            # print('yes, val {}'.format(spath / f'epoch_{ep}_val.png'))
            plt.close(fig)

        rel_err /= ntest
        print('rel_err:{}'.format(rel_err))

    test_losses.append(rel_err)
    ##### BOOKKEEPING
    if saved_model == True:
        if ep % save_int == 0:
            torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')
            np.save(save_path / 'test_loss_epoch.npy', np.array(test_losses))


# ## Evaluation

# In[7]:


# load the trained model
"""
spath = Path('/net/ghisallo/scratch1/yshi5/OFM_PDE/GITO_exp/GITO_T_NS')

for param in model.parameters():
    param.requires_grad = False
    
model_path = os.path.join(spath, 'epoch_500.pt')
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint, strict=False)

"""
