import numpy as np
import torch
from torchdiffeq import odeint
#from util.gaussian_process import GPPrior
from util.true_gaussian_process_seq import true_GPPrior
from torchcfm.optimal_transport import OTPlanSampler
from util.util import reshape_for_batchwise, plot_loss_curve

import time

class OFMModel:
    def __init__(self, model, kernel_length=0.01, kernel_variance=1.0, nu=0.5, sigma_min=1e-4, device='cpu', dtype=torch.double,
                 x_dim=None, n_pos=None):
        
        # x_dim = 1,2, 3
        # n_pos = [n_points, x_dim]
        
        self.model = model
        self.device = device
        self.dtype = dtype
        self.n_pos = n_pos.to(device)
        #self.gp = true_GPPrior(lengthscale=kernel_length, var=kernel_variance, nu=nu, device=device, grids=make_grid(dims))
        self.gp = true_GPPrior(lengthscale=kernel_length, var=kernel_variance, nu=nu, device=device, x_dim=x_dim, n_pos=n_pos)
        self.ot_sampler = OTPlanSampler(method="exact")
        self.sigma_min = sigma_min

    def sample_gp_noise(self, x_data):
        # sample GP noise with OT 
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]

        x_0 = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels) 
        x_0, x_data = self.ot_sampler.sample_plan(x_0, x_data)
        
        return x_0, x_data
        
    def simulate(self, t, x_0, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, n_point]
        # samples from p_t(x | x_data)
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        #dims = x_data.shape[2:]
        #n_dims = len(dims)
        
        # Sample from prior GP
        noise = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels)
    
        # Construct mean/variance parameters
        t = reshape_for_batchwise(t, 2) # [batch, n_chan, n_pos]
        
        mu = t * x_data + (1 - t) * x_0
        samples = mu + self.sigma_min * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, x0, x1):
        # computes v_t(x_noisy | x_data)
        # x_data, x_noisy: (batch_size, n_channels, *dims)

        return x1 - x0

    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None, saved_model=False):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        model = self.model
        device = self.device
        dtype = self.dtype

        first = True
        for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            # rewrite the train_loader, it should be a list
            for batch_pack in train_loader:
                    
                batch = batch_pack['input_feat'].to(device) # [batch_size, n_chan, n_seq]
                pos = batch_pack['input_pos'].to(device) # [batch_size, x_dim, n_seq]
                query_pos = batch_pack['query_pos'].to(device)
                #supernode_idxs = batch_pack['supernode_idxs'].to(device) # [batch_size*n_super_nodes]
                #batch_idx = batch_pack['batch_idx'].to(device) # [batch_size*n_seq]
                
                #pos = pos[0:1].to(device) # requirment of GINO
                batch_size = batch.shape[0]
                    
                # GP noise with OT reorder
                x_0, x_data = self.sample_gp_noise(batch)
        
                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                
                x_t = self.simulate(t, x_0, x_data)
                # Get conditional vector fields
                target = self.get_conditional_fields(x_0, x_data)

                x_t = x_t.to(device)
                target = target.to(device)         

                # Get model output
                #print('t before the model :{}'.format(t))
                model_out = model(input_feat=x_t, input_pos=pos, query_pos=query_pos, timestep=t)

                # Evaluate loss and do gradient step
                #print('target:{}, model_out:{}'.format(target.shape, model_out.shape))
                optimizer.zero_grad()
                loss = torch.mean((model_out - target)**2 ) 
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()


            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            
            ##### BOOKKEEPING
            if saved_model == True:
                if ep % save_int == 0:
                    torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')
                    np.save(save_path / 'train_loss.npy', np.array(tr_losses))

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')


    @torch.no_grad()
    def sample(self, pos, query_pos, n_channels=1, n_samples=1, n_eval=2, return_path=False, rtol=1e-5,
               atol=1e-5, method = 'dopri5'):
        
        # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2. 
        n_pos = pos[0].permute(1,0).cpu()
        
        t = torch.linspace(0, 1, n_eval, device=self.device)
        x0 = self.gp.sample(n_pos, n_samples=n_samples, n_channels=n_channels)

        def sample_func(t, x_t):
            return self.model(input_feat=x_t, input_pos=pos,  query_pos=query_pos, timestep=t) 
            
        out = odeint(sample_func, x0, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]
