import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import GAT

class Posterior(nn.Module):

    def __init__(self, h_dim, post_hid_dim, z_dim):
        super(Posterior, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.post_hid_dim = post_hid_dim

        self.post_encoder = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.post_hid_dim),
            nn.ReLU(),
            nn.Linear(self.post_hid_dim, self.post_hid_dim),
            nn.ReLU()
        )
        self.post_mean = nn.Linear(self.post_hid_dim, self.z_dim)
        self.post_std = nn.Sequential(
            nn.Linear(self.post_hid_dim, self.z_dim),
            nn.Softplus()) 

    def forward(self, phi_x_t, h):

        enc_post = self.post_encoder(torch.cat([phi_x_t, h], dim=1))
        post_mean = F.relu(self.post_mean(enc_post))
        post_std = self.post_std(enc_post)

        return post_mean, post_std
    
class Prior(nn.Module):

    def __init__(self, h_dim, prior_hid_dim, z_dim):
        super(Prior, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.prior_hid_dim = prior_hid_dim

        self.prior_encoder = nn.Sequential(
            nn.Linear(self.h_dim, self.prior_hid_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.prior_hid_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.prior_hid_dim, self.z_dim),
            nn.Softplus())

    def forward(self, h):

        enc_prior = self.prior_encoder(h)
        prior_mean = F.relu(self.prior_mean(enc_prior))
        prior_std = self.prior_std(enc_prior)

        return prior_mean, prior_std
    
class AttrDecoder(nn.Module):

    def __init__(self, h_dim, attr_hid_dim,x_dim,dec_method='mlp',no_neg=True,device='cpu'):
        super(AttrDecoder, self).__init__()

        self.h_dim = h_dim
        self.attr_hid_dim = attr_hid_dim
        self.x_dim = x_dim
        self.dec_method=dec_method 
        self.no_neg=no_neg
        self.device=device

        if self.dec_method=='mlp':
            
            self.attr_decoder = nn.Sequential(
                nn.Linear(self.h_dim + self.h_dim, self.attr_hid_dim),
                nn.ReLU(),
                nn.Linear(self.attr_hid_dim, self.attr_hid_dim),
                nn.ReLU())
        elif self.dec_method=='gnn':
            self.attr_decoder=GAT(self.h_dim + self.h_dim,
                                  self.attr_hid_dim,
                                  self.attr_hid_dim,device=self.device)
        else:
            raise ValueError('Wrong Mehtod!')
        
        self.attr_decoder_std = nn.Sequential(
            nn.Linear(self.attr_hid_dim, self.x_dim),
            nn.Softplus()) 
        self.attr_decoder_mean = nn.Linear(
            self.attr_hid_dim, self.x_dim)  

    def forward(self, phi_z_t, h, gen_A_t):

        if self.dec_method=='mlp':
            dec_attr = self.attr_decoder(torch.cat([phi_z_t, h], dim=1))
        elif self.dec_method=='gnn':
            dec_attr=self.attr_decoder(torch.cat([phi_z_t, h], dim=1),gen_A_t)
        else:
            raise ValueError('Wrong Method!')
        
        if self.no_neg:
            dec_attr_mean = F.relu(self.attr_decoder_mean(dec_attr)) 
        else:
            dec_attr_mean = self.attr_decoder_mean(dec_attr)
        dec_attr_std = self.attr_decoder_std(dec_attr)

        return dec_attr_mean, dec_attr_std