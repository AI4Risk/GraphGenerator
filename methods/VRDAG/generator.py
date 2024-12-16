import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BidirectionalEncoder,Time2Vec,GIN
from .mix_bernoulli import MixtureBernoulli
from .var_dist import *
import warnings
warnings.filterwarnings('ignore')


class VRDAG(nn.Module):

    def __init__(self, config):

        super(VRDAG, self).__init__()
        
        self.x_dim = len(config['attr_col'][config['data']])
        self.h_dim = config['h_dim']  
        self.z_dim = config['z_dim']  
        self.enc_hid_dim = config['enc_hid_dim']  
        self.n_encoder_layer = config['n_encoder_layer'] 
        self.bi_flow=config['bi_flow']
        self.post_hid_dim = config['post_hid_dim'] 
        self.prior_hid_dim = config['prior_hid_dim']  
        self.attr_hid_dim = config['attr_hid_dim']  
        self.bernoulli_hid_dim = config['bernoulli_hid_dim']  
        self.n_rnn_layer = config['n_rnn_layer']  
        self.max_num_nodes = config['max_num_nodes']  
        self.num_mix_component = config['num_mix_component']  
        self.ini_method=config['ini_method'] 
        self.attr_optimize=config['attr_optimize']
        self.seq_len = config['seq_len']
        self.neg_num=config['neg_num'] 
        self.dec_method=config['dec_method'] 
        self.no_neg=config['no_neg'] 
        self.eps=config['eps']
        self.pos_weight=config['pos_weight'] 
        self.reduce = config['reduce'] 
        self.device = config['device']  

        self.activation = config['activation']  
        self.time_embed_size = config['h_dim']  
        self.is_vectorize=config['is_vectorize'] 

        self.time_to_vec=Time2Vec(self.activation,self.time_embed_size,self.device)

        if config['ini_method']=='embed':
            self.id_embedding = nn.Embedding(
                self.max_num_nodes, self.h_dim).to(self.device)
        else:
            self.id_embedding=None
        
        if self.bi_flow:
            self.phi_x = BidirectionalEncoder(self.x_dim,
                                            self.enc_hid_dim,
                                            self.h_dim,
                                            self.n_encoder_layer,
                                            self.device)
        else:
            self.phi_x=GIN(self.x_dim,
                           self.enc_hid_dim,
                           self.h_dim,
                           layer_num=self.n_encoder_layer,
                           device=self.device)
            
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU()).to(self.device)

        self.post_dist = Posterior(
            self.h_dim, self.post_hid_dim, self.z_dim).to(self.device)

        self.prior_dist = Prior(
            self.h_dim, self.prior_hid_dim, self.z_dim).to(self.device)

        self.attr_decoder = AttrDecoder(
            self.h_dim, self.attr_hid_dim, self.x_dim,
            self.dec_method,self.no_neg,device=self.device).to(self.device)

        self.topo_decoder = MixtureBernoulli(self.h_dim+self.h_dim,
                                             self.bernoulli_hid_dim,
                                             self.num_mix_component,
                                             self.max_num_nodes).to(self.device)

        self.rnn = nn.GRU(self.h_dim+self.h_dim, self.h_dim,
                          self.n_rnn_layer).to(self.device)

        self.attr_loss_func = nn.MSELoss(reduction='sum')
        self.adj_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight],device=self.device),reduction='sum')
        self.recon_loss=nn.MSELoss(reduction='sum')
        
    def forward(self, A_t,X_t,h,t_vec,n_nodes):

        phi_x_t = self.phi_x(X_t, A_t) 

        if torch.is_tensor(t_vec):
            phi_x_t+=t_vec
        
        # inference stage
        post_mean_t, post_std_t = self.post_dist(phi_x_t, h)

        # prior distribution
        prior_mean_t, prior_std_t = self.prior_dist(h)

        # sampling and reparameterization 
        z_t = self._reparameterized_sample(post_mean_t, post_std_t)
        phi_z_t = self.phi_z(z_t)


        # topology decoder：
        log_alpha, log_theta,gen_adj = self.topo_decoder(
            phi_z_t, h, is_sampling=False)
        
        # attr_decoder：
        attr_mean_t, attr_std_t = self.attr_decoder(phi_z_t, h, gen_adj)
        
        _, h = self.rnn(
            torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h.unsqueeze(0))
        h = h.squeeze(0)  

        attr_loss=self._loss_gauss_attr(attr_mean_t, attr_std_t, X_t)
        
        struc_loss=self._loss_mixbernoulli_topo(log_alpha, log_theta, A_t, gen_adj, n_nodes)

        kld_loss_t = self._kld_gauss(post_mean_t,
                                    post_std_t, prior_mean_t, prior_std_t)
        
        return h, kld_loss_t, struc_loss, attr_loss
        
    def _sampling(self, seq_len):

        with torch.no_grad():

            sample = []  
            
            if self.ini_method=='zero':
                h = torch.zeros(self.max_num_nodes,self.h_dim,device=self.device)  
            elif self.ini_method=='embed':
                h = self.id_embedding
            else:
                raise ValueError('Wrong initialization method!')

            for t in range(seq_len):

                prior_mean_t, prior_std_t = self.prior_dist(h)

                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                phi_z_t = self.phi_z(z_t)
                
                adj_t = self.topo_decoder(phi_z_t, h, is_sampling=True)
                
                attr_mean_t, _ = self.attr_decoder(phi_z_t, h,adj_t)

                phi_x_t = self.phi_x(attr_mean_t, adj_t)
                
                if self.is_vectorize:
                    t_vec=self.time_to_vec(torch.FloatTensor([t]).to(self.device))
                    phi_x_t+=t_vec

                _, h = self.rnn(
                    torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h.unsqueeze(0))
                h = h.squeeze(0)
                
                sample.append((adj_t.data,
                              attr_mean_t.data))

        return sample

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device,
                          dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, p_mu, p_std, q_mu, q_std):

        p = torch.distributions.normal.Normal(p_mu, p_std)
        q = torch.distributions.normal.Normal(q_mu, q_std)

        kld_loss = 0.5 * torch.distributions.kl_divergence(p, q).sum()

        return kld_loss

    def _loss_gauss_attr(self, dist_mean_t, dist_std_t, x):

        if self.attr_optimize=='mse':
            dist = torch.distributions.normal.Normal(dist_mean_t, dist_std_t)
            x_tau = dist.sample()  # 分布采样
            attr_loss = self.attr_loss_func(x, x_tau)
        
        elif self.attr_optimize=='kld':
            
            gauss_diff=torch.log(2*torch.FloatTensor([torch.pi]).to(self.device))/2
            attr_loss=torch.sum(torch.log(dist_std_t + self.eps) + gauss_diff + (x - dist_mean_t).pow(2)/(2*(dist_std_t+self.eps).pow(2)))
        
        elif self.attr_optimize=='sce':
            loss_func=nn.CosineSimilarity()
            attr_loss=(1-loss_func(dist_mean_t,x)).pow(2).sum()
        
        else:
            raise ValueError('Wrong optimization method!')
        
        return attr_loss

    def _loss_mixbernoulli_topo(self, log_alpha, log_theta, A_t, gen_adj, n_nodes):
        
        topo_loss = None
        
        real_log_alpha=F.log_softmax(torch.sum(log_alpha,dim=1),dim=1)
        
        mask_alpha=torch.ones_like(real_log_alpha,dtype=torch.bool,device=self.device)
        mask_alpha[:n_nodes]=torch.zeros_like(mask_alpha[:n_nodes],device=self.device)
        real_log_alpha.masked_fill(mask_alpha,value=0)
        
        real_log_alpha=real_log_alpha[:n_nodes]
        
        mask_theta=torch.ones_like(log_theta,dtype=torch.bool,device=self.device)
        for node in range(n_nodes):
            pos_sample_ids=A_t[node].nonzero()
            neg_sample_ids=torch.randint(0,n_nodes,size=(1,self.neg_num))
            if torch.numel(pos_sample_ids):
                mask_theta[node][pos_sample_ids]=torch.zeros_like(mask_theta[node][pos_sample_ids]) 
            mask_theta[node][neg_sample_ids]=torch.zeros_like(mask_theta[node][neg_sample_ids]) 
        log_theta.masked_fill(mask_theta,value=-1e9)
        
        log_theta=log_theta[:n_nodes,:n_nodes]
        
        real_adj_loss=torch.cat([torch.stack([self.adj_loss_func(log_theta[:, :, k][i], A_t[i][:n_nodes]) for i in range(n_nodes)]).unsqueeze(1) for k in range(self.num_mix_component)],dim=1)
        
        if self.reduce == 'sum':
            topo_loss = - \
                torch.logsumexp(real_log_alpha-real_adj_loss, dim=1).sum()
        elif self.reduce == 'mean':
            topo_loss = - \
                torch.logsumexp(real_log_alpha-real_adj_loss, dim=1).mean()
        else:
            raise ValueError(
                '{} reduce method does not exist!'.format(self.reduce))

        
        recon_loss=self.recon_loss(A_t[:n_nodes,:n_nodes],gen_adj[:n_nodes,:n_nodes])
        topo_loss+=recon_loss
        
        return topo_loss
