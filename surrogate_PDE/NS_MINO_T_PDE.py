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

torch.cuda.is_available()


# In[4]:
# print current hostname (computation node)
import socket

print(socket.gethostname())


# In[10]:


dims = [64, 64]
query_dims = [16, 16]
x_dim = 2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
spath = Path('/home/yshi/PDE_solving/dataset/saved/MINO_T_NS_time_PDE')

spath.mkdir(parents=True, exist_ok=True)
saved_model = True # save model
save_int = 500

# model hyperparameters
## conditional time step..
dim = 256
num_heads = 4

## training parameters
epochs = 500
batch_size = 2


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
data_path = Path('/home/yshi/PDE_solving/dataset/MINO_PDE/NS_time')
data_path = data_path / 'NavierStokes_V1e-5_N1200_T20.mat'
#train_path = data_path / 'piececonst_r421_N1024_smooth1.mat'
#test_path = data_path / 'piececonst_r421_N1024_smooth2.mat'


# In[6]:


#### We follow the pipeline of Transolver for processing the Darcy flow data
## the resolution is [85, 85]

ntrain = 1000
ntest = 200

T_in = 10
T = 10
step = 1 # one step prediction 

downsample = 1

r = downsample
h = int(((64- 1) / r) + 1)  # 85

data = scio.loadmat(data_path)
train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
train_a = torch.from_numpy(train_a)
train_a = train_a.permute(0, 2, 1)

train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
train_u = torch.from_numpy(train_u)
train_u = train_u.permute(0, 2, 1)


test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
test_a = torch.from_numpy(test_a)
test_a = test_a.permute(0, 2, 1)

test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
test_u = torch.from_numpy(test_u)
test_u = test_u.permute(0, 2, 1)

x = np.linspace(0, 1, h)
y = np.linspace(0, 1, h)
x, y = np.meshgrid(x, y)
pos = np.c_[x.ravel(), y.ravel()]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

pos = pos.permute(0, 2 ,1)
pos_train = pos.repeat(ntrain, 1, 1)
pos_test = pos.repeat(ntest, 1, 1)

## latent position
query_pos = make_2d_grid(query_dims).permute(1, 0)  # [2, 16x16]

#%%
# channel = 10

#%%
train_dataset = SimDataset_PDE(
    input_data=train_a, output_data=train_u, pos=pos_train, query_pos=query_pos
)
test_dataset = SimDataset_PDE(
    input_data=test_a, output_data=test_u, pos=pos_test, query_pos=query_pos
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


# %%

model = MINO(
    encoder=EncoderSupernodes(
        input_dim=10,  # co-domain
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
        in_out_dim_same=False,
        val_dim=10,
    ),
)
model = model.to(device)
print(f'parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

# %%

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

myloss = TestLoss(size_average=False)

# In[11]:


for ep in range(1, epochs + 1):
    model.train()
    train_l2_step = 0
    train_l2_full = 0

    for batch_pack in train_loader:
        batch = batch_pack['input_feat'].to(device)  # [batch_size, n_chan, n_seq]
        pos = batch_pack['input_pos'].to(device)  # [batch_size, x_dim, n_seq]
        query_pos = batch_pack['query_pos'].to(device)
        out_batch = batch_pack['output_feat'].to(device)

        bsz = batch.shape[0]
        loss = 0

        for t in range(0, T, step):
            one_step_forward = out_batch[:, t:t + step, :]

            one_step_predict = model(input_feat=batch, input_pos=pos, query_pos=query_pos)
            # progressively update the batch 
            loss += myloss(one_step_predict.reshape(bsz, -1), one_step_forward.reshape(bsz, -1))
            if t == 0:
                pred = one_step_predict
            else:
                pred = torch.cat((pred, one_step_predict), 1) #concatenate them along channel dim

            batch = torch.cat((batch[:,step:,:], one_step_forward), dim=1) # ground truth

        train_l2_step += loss.item()
        train_l2_full += myloss(pred.reshape(bsz, -1), out_batch.reshape(bsz, -1)).item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # reg += deriv_loss.item()
    scheduler.step()

    ## 
    # plot save

    test_l2_step = 0
    test_l2_full = 0

    model.eval()

    with torch.no_grad():
        for batch_pack in test_loader:
            batch = batch_pack['input_feat'].to(device)  # [batch_size, n_chan, n_seq]
            pos = batch_pack['input_pos'].to(device)  # [batch_size, x_dim, n_seq]
            query_pos = batch_pack['query_pos'].to(device)
            out_batch = batch_pack['output_feat'].to(device)

            bsz = batch.shape[0]
            loss = 0

            for t in range(0, T, step):
                one_step_forward = out_batch[:, t:t + step, :]
                one_step_predict = model(input_feat=batch, input_pos=pos, query_pos=query_pos)
                # progressively update the batch 
                loss += myloss(one_step_predict.reshape(bsz, -1), one_step_forward.reshape(bsz, -1))
                if t == 0:
                    pred = one_step_predict
                else:
                    pred = torch.cat((pred, one_step_predict), 1) #concatenate them along channel dim

                batch = torch.cat((batch[:,step:,:], one_step_predict), dim=1) # ground truth
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bsz, -1), out_batch.reshape(bsz, -1)).item()
            
    if ep % 5 == 0:
        batch_idx = 0
        even_indices = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

        predict_snapshots = pred[batch_idx, even_indices, :].detach().cpu().numpy()     # shape: [5, 4096]
        out_snapshots = out_batch[batch_idx, even_indices, :].detach().cpu().numpy()       # shape: [5, 4096]

        fig, axes = plt.subplots(2, 5, figsize=(18, 6))

        for i in range(5):
            # Predict row (first)
            axes[0, i].imshow(predict_snapshots[i].reshape(64, 64), cmap='viridis')
            axes[0, i].set_title(f'Predict #{even_indices[i]}')
            axes[0, i].axis('off')
            # Out_batch row (second)
            axes[1, i].imshow(out_snapshots[i].reshape(64, 64), cmap='viridis')
            axes[1, i].set_title(f'GT #{even_indices[i]}')
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel("Predict", fontsize=14)
        axes[1, 0].set_ylabel("Ground Truth", fontsize=14)
        plt.tight_layout()

        plt.savefig(spath / f'epoch_{ep}_test.png')
        plt.close(fig)

    print(
        "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(
            ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
                test_l2_full / ntest))
    ##### BOOKKEEPING
    if saved_model == True:
        if ep % save_int == 0:
            torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')
            np.save(save_path / 'test_loss_epoch.npy', np.array(test_losses))


# ## Evaluation

