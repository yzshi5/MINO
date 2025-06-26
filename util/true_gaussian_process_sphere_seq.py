import torch
import numpy as np

import geometric_kernels

# Note: if you are using a backend other than numpy,
# you _must_ uncomment one of the following lines
# import geometric_kernels.tensorflow
# import geometric_kernels.torch
# import geometric_kernels.jax

# Import a space and an appropriate kernel.
from geometric_kernels.spaces import Hypersphere
from geometric_kernels.kernels import MaternGeometricKernel

from geometric_kernels.kernels import default_feature_map
from geometric_kernels.sampling import sampler
from geometric_kernels.kernels import default_feature_map

# Gaussian Process sequence. 

class true_GPPrior(torch.distributions.distribution.Distribution):
    
    """ Wrapper around some torch utilities that makes prior sampling easy.
    """

    def __init__(self, kernel=None, mean=None, lengthscale=None, var=None, nu=0.5, device='cpu', x_dim=2, n_pos=None):
        """
        n_pos : [n_points, x_dim] 
        """
        assert var == 1, 'variance is not 1' 
        
        sphere = Hypersphere(dim=2) # only work for S^2
        kernel = MaternGeometricKernel(sphere)
        params = kernel.init_params()
        params["lengthscale"] = np.array([lengthscale])
        params["nu"] = np.array([nu])
        
        self.params = params
        self.n_pos = n_pos.numpy()
        self.feature_map = default_feature_map(kernel=kernel)
        self.device = device
        self.scale = 5

    
    def sample(self, n_pos, n_samples=1, n_channels=1):
        """ Draws samples from the GP prior.
        n_pos: torch.Tensor([n_points, x_dim])
        n_samples: number of samples to draw
        n_channels: number of independent channels to draw samples for

        returns: samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        sample_paths = sampler(self.feature_map, s=n_samples*n_channels)
        _, samples = sample_paths(n_pos.numpy(), self.params)
        
        samples = torch.Tensor(samples).to(self.device)
        samples = samples.permute(1,0).reshape(n_samples, n_channels, -1)
                
        return samples*self.scale
        
    
    def sample_from_prior(self, n_samples=1, n_channels=1):
        """
        fixed prior
        """
        sample_paths = sampler(self.feature_map, s=n_samples*n_channels)
        _, samples = sample_paths(self.n_pos, self.params)
        
        samples = torch.Tensor(samples).to(self.device)
        samples = samples.permute(1,0).reshape(n_samples, n_channels, -1)
        
        # [batch_size, n_chan, n_seq]
        return samples*self.scale           
    
    def sample_train_data(self, n_samples=1, n_channels=1, nbatch=1000):
        """
        calculation in cuda, but saved in cpu.
        iteratively 
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])
              
        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(nbatch, n_channels).cpu()
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)
                
        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all
        