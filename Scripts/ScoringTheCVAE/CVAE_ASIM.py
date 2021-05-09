#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:05:38 2021

@author: rcj
"""
import torch
import numpy as np
from typing import *# The Dict class for the Forward pass is in the typing package

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,
        
    def forward(self, x):
        return x.view(*self.shape)
    
class CVAENet(torch.nn.Module):
    def __init__(self, input_shape:torch.Size, latent_features:int, n, batch_size:int):
        super(CVAENet, self).__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.batch_size = batch_size
        # Define Encoder
        self.Encode = torch.nn.Sequential(
            # First conv1
            torch.nn.Conv1d(in_channels=1,
                            out_channels=3,
                            kernel_size=4,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(3),                
            
            # Conv to half size
            torch.nn.Conv1d(in_channels=3,
                            out_channels=15,
                            kernel_size=4,
                            stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(15),
            # Conv to 1/4th
            torch.nn.Conv1d(in_channels=15,
                            out_channels=9,
                            kernel_size=4,
                            stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(9),
            
            # Scale to 1/8th
            torch.nn.Conv1d(in_channels=9,
                            out_channels=3,
                            kernel_size=4,
                            stride=2),
                            
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(3), # BNBeforeLinLayer
           # torch.nn.Dropout(p=0.3), # NoDropout
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=n, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=250),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=250, out_features=latent_features*2),

        )        
        self.Decode = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_features, out_features=250),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=250, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=n),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1), # BNBeforeLinLayer
            # Upscale to 1/8
            View([batch_size, 3, -1]),
            #GetSize(),
            torch.nn.ConvTranspose1d(in_channels=3,
                            out_channels=9,
                            kernel_size=4,
                            stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.BatchNorm1d(9),
            
            # Upscale to 1/4
            torch.nn.ConvTranspose1d(in_channels=9,
                            out_channels=15,
                            kernel_size=4,
                            stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.BatchNorm1d(15),
            
            # Upscale to 1/2
            torch.nn.ConvTranspose1d(in_channels=15,
                            out_channels=3,
                            kernel_size=4,
                            stride=2
            ),
            torch.nn.ReLU(),
            # There appears to be a problem with simply using scale_factor 2 here, 
            # because it results in an even number, where the encoder rounds down to an uneven. Hardcode it instead.
            torch.nn.Upsample(14997), 
            # Last conv to original data size
            torch.nn.BatchNorm1d(3),
            torch.nn.ConvTranspose1d(in_channels=3,
                            out_channels=3,
                            kernel_size=4,
                            stride=2,
                            output_padding = 1
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(3),
            
            # Double the length for alpha and beta parameters in beta distribution
            torch.nn.ConvTranspose1d(in_channels=3,
                            out_channels=1,
                            kernel_size=4,
                            stride=1,
                            output_padding = 0
            ),
    )
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

    def posterior(self, x:torch.Tensor) -> torch.distributions.Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.Encode(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> torch.distributions.Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:torch.Tensor) -> torch.distributions.Distribution:
        """return the distribution `p(x|z)`"""
        # Parameters for beta distribution. Chunk the double length, take exp() and return Beta distribution.
        alpha,beta = self.Decode(z).chunk(2,dim=-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return torch.distributions.beta.Beta(alpha,beta)

    def forward(self, x, batch_size:int=100) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample().view(batch_size, 1, -1) 
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


def reduce(x:torch.Tensor) -> torch.Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(torch.nn.Module):
    def __init__(self, b:float):
        super().__init__()
        self.b = b
        
    def forward(self, model:torch.nn.Module, x:torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        log_px = reduce(px.log_prob(torch.clamp(x, 0.000001, 0.99999))) # Clamp the data to be sure we're within [0,1 range]. 0 and 1 are apparently not included in the Beta Distribution
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO
        kl = (torch.sum(qz.sigma**2 + qz.mu**2 - 2*torch.log(qz.sigma)-1, dim=-1)/2).mean()
        reconLoss = torch.sum(px.log_prob(torch.clamp(x, 0.000001, 0.99999)), dim=-1).mean()
        beta_elbo = reconLoss - self.b*kl
        
        # loss
        loss = -beta_elbo.mean()
       # print('loss: ', loss) 
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs
        
class ReparameterizedDiagonalGaussian(torch.distributions.Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: torch.Tensor, log_sigma:torch.Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> torch.Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> torch.Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> torch.Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon() # <- your code
        
    def log_prob(self, z:torch.Tensor) -> torch.Tensor:
        """return the log probability: log `p(z)`"""
        pz = torch.distributions.normal.Normal(self.mu, self.sigma)
        return pz.log_prob(z)