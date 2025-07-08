import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

## reduce the batch_size to fit the GPU memoery, batch_size=96 ~ 40 GB memoery

## load path
import os
import sys
sys.path.append('./')
sys.path.append('./models')
from pathlib import Path

## load utils 
from util.util import *
# we provide two realizations of GP on sphere, the commented one is slower
#from util.ofm_OT_likelihood_sphere_seq_mino import * 
from util.true_gaussian_process_seq import *
from util.ofm_OT_likelihood_seq_mino import *
from util.metrics import *
import time
from mpl_toolkits.mplot3d import Axes3D  

## load modules 


from models.mino_transformer import MINO
from models.mino_modules.decoder_perceiver import DecoderPerceiver
from models.mino_modules.encoder_supernodes_gno_cross_attention import EncoderSupernodes
from models.mino_modules.conditioner_timestep import ConditionerTimestep

import argparse

parser = argparse.ArgumentParser('Training MINO-T on Global Climate dataset')
parser.add_argument('-model', type=str)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--saved_model', type=int, default=1) #True
parser.add_argument('--data_path', type=str)
parser.add_argument('--spath', type=str)

# data parameters
parser.add_argument('--x_dim', type=int, default=2)
parser.add_argument('--query_dims', type=int, nargs='+') #[16, 16]
parser.add_argument('--co_domain', type=int)
parser.add_argument('--radius', type=float)

# GP hyperparameters
parser.add_argument('--kernel_length', type=float, default=0.01)
parser.add_argument('--kernel_variance', type=float, default=1.0)
parser.add_argument('--nu', type=float, default=0.5)
parser.add_argument('--sigma_min', type=float, default=1e-4)

# training and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--step_size', type=int, default=25)
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--eval', type=int, default=0, help='evaluation mode') ##1 for inference only

# model hyperparameters
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--enc_depth', type=int)
parser.add_argument('--dec_depth', type=int)


args = parser.parse_args()

data_path = args.data_path + '/train_climate.npy'
data_test_path = args.data_path + '/test_climate.npy'

spath = Path(args.spath)
spath.mkdir(parents=True, exist_ok=True)

def gen_n_pos(_NUM_LONGS=90, _NUM_LATS=46):
    # follow the github GeometricKernel tutorial
    longs, lats =  np.mgrid[0:2*np.pi:(_NUM_LONGS+1)*1j, 0:np.pi:_NUM_LATS*1j]

    longs, lats = longs[:-1,:], lats[:-1,:]

    other_points_xs = np.sin(lats) * np.cos(longs)
    other_points_ys = np.sin(lats) * np.sin(longs)
    other_points_zs = np.cos(lats)

    other_points = np.c_[np.ravel(other_points_xs),
                         np.ravel(other_points_ys),
                         np.ravel(other_points_zs)]

    n_pos = other_points
    return n_pos

def latent_query_sphere(num_longs, num_lats):
    longs, lats =  np.mgrid[0:2*np.pi:(num_longs+1)*1j, 0:np.pi:num_lats*1j]

    longs, lats = longs[:-1,2:-2], lats[:-1,2:-2] # remove queried nodes in (near) the poles

    print("qeury pos:{}".format(longs.shape))
    other_points_xs = np.sin(lats) * np.cos(longs)
    other_points_ys = np.sin(lats) * np.sin(longs)
    other_points_zs = np.cos(lats)

    query_pos = np.c_[np.ravel(other_points_xs),
                         np.ravel(other_points_ys),
                         np.ravel(other_points_zs)]
    
    return query_pos

def plot_3d_scatter(pos, s=1):
    x, y, z = pos.T
    
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z,
               s=s,            # marker size
               alpha=0.8)       # slight transparency

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3‑D scatter plot")

    plt.tight_layout()
    plt.show()    
    

def gen_meta_info(batch_size, query_pos, n_pos):

    
    n_pos = n_pos
    pos_data = n_pos.unsqueeze(0).repeat(batch_size, 1, 1)
    
    #query_pos = make_3d_grid(query_dims, x_min=-1, x_max=1)
    query_pos_data = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)

    collated_batch = {}
    collated_batch["input_pos"] = pos_data.permute(0, 2, 1)
    collated_batch['query_pos']= query_pos_data
    
    return collated_batch

def main():
    
    # dataloader
    x_train = np.load(data_path)
    x_train = torch.Tensor(x_train[:,2:3]).permute(0,1,3,2) # 0:2 (longitude, latitude)
    x_train = torch.flatten(x_train, start_dim=2)

    n_pos = gen_n_pos()
    n_pos = torch.Tensor(n_pos)
    pos_data = n_pos.unsqueeze(0).repeat(len(x_train), 1, 1).permute(0,2,1) 

    # [32, 16], remove queried nodes around poles
    query_pos_input = latent_query_sphere(num_longs=args.query_dims[0], num_lats=args.query_dims[1]+4)
    #plot_3d_scatter(query_pos_input)
    
    query_pos_input = torch.Tensor(query_pos_input).permute(1,0)
    train_dataset = SimDataset(x_train, pos_data, query_pos_input)

    loader_tr =  DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SimulationCollator,
    )
    
    x_test = np.load(data_test_path)
    x_test = torch.Tensor(x_test) #total 2000 samples
    x_test = torch.Tensor(x_test[:,2:3]).permute(0,1,3,2) # (longitude, latitude)
    x_test = torch.flatten(x_test, start_dim=2)

    print('Dataloading is over')
    
    conditioner = ConditionerTimestep(
        dim=args.dim
    )
    model = MINO(
        conditioner=conditioner,
        encoder=EncoderSupernodes(
            input_dim=args.co_domain, # co-domain 
            ndim=args.x_dim, # dimension of domain
            radius= args.radius,
            enc_dim=args.dim,
            enc_num_heads=args.num_heads,
            enc_depth=args.enc_depth,
            cond_dim=conditioner.cond_dim,
        ),
        decoder=DecoderPerceiver(
            input_dim=args.dim,
            output_dim=args.co_domain,
            ndim=args.x_dim,
            dim=args.dim,
            num_heads=args.num_heads,
            depth=args.dec_depth, # 2 layers
            unbatch_mode="dense_to_sparse_unpadded",
            cond_dim=conditioner.cond_dim,
        ),
    )
    model = model.to(args.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")   
    if args.eval:
        # 
        # skip traininig
        print('start evaluating')
        for param in model.parameters():
            param.requires_grad = False
        model_path = os.path.join(spath, 'epoch_480.pt')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        fmot = OFMModel(model, kernel_length=args.kernel_length, kernel_variance=args.kernel_variance, nu=args.nu, sigma_min=args.sigma_min, device=args.device, x_dim=args.x_dim, n_pos=n_pos)
        
    else:    
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
        fmot = OFMModel(model, kernel_length=args.kernel_length, kernel_variance=args.kernel_variance, nu=args.nu, sigma_min=args.sigma_min, device=args.device, x_dim=args.x_dim, n_pos=n_pos)

        fmot.train(loader_tr, optimizer, epochs=args.epochs, scheduler=scheduler, eval_int=int(0), save_int=int(args.epochs), generate=False, save_path=spath,saved_model=args.saved_model)
    
        print("Training is over, start evaluating")

    start = time.time()    
    with torch.no_grad():

        X_alt = []
        # generate 5000 synthetic samples
        for i in range(26):
            collated_batch =  gen_meta_info(batch_size=100, n_pos=n_pos,  query_pos=query_pos_input)
            pos, query_pos = collated_batch['input_pos'], collated_batch['query_pos']
            X_temp = fmot.sample(pos=pos.to(args.device), query_pos=query_pos.to(args.device), n_samples=100, n_channels=args.co_domain, n_eval=2).cpu()

            X_alt.append(X_temp)

        X_alt = torch.vstack(X_alt).squeeze()
    end = time.time()
    print("Generation time :{} s".format(end-start))
    
    X_alt = X_alt[:len(x_test)]
    ## print metrics

    ## general metric 
    swd_value = swd_stable(X=X_alt, Y=x_test)
    mmd_value = unbiased_mmd2_torch(X=X_alt, Y=x_test, device=args.device)  

    print('swd:{:.3f}, mmd:{:.3f}'.format(swd_value, mmd_value))

if __name__ == "__main__":
    main()