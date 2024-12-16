import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureBernoulli(nn.Module):

    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 num_mix_component,
                 max_num_nodes,
                 ):
        super(MixtureBernoulli, self).__init__()

        self.embed_dim = embed_dim  
        self.hidden_dim = hidden_dim  
        self.max_num_nodes = max_num_nodes  
        self.num_mix_component = num_mix_component  

        self.output_theta = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component)
        )

        self.output_alpha = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component)
        )

    def _inference(self, phi_z_t, h):
        node_state = torch.cat([phi_z_t, h], dim=1)

        diff = node_state.unsqueeze(1)-node_state.unsqueeze(0)

        log_alpha = self.output_alpha(diff)  # [N(t+1) x N(t+1) x num_mix_component]
        log_theta = self.output_theta(diff)  # [N(t+1) x N(t+1) x num_mix_component]

        prob_alpha = F.softmax(torch.sum(log_alpha,dim=1), -1)  
        alpha = torch.multinomial(prob_alpha, 1).long()  

        log_theta = self.output_theta(diff)

        probs = torch.sigmoid(torch.cat([log_theta[i, :, alpha[i]] for i in range(
            self.max_num_nodes)], dim=1))  # [N(t+1) x N(t+1)]

        adj = torch.bernoulli(probs)

        return log_alpha, log_theta, adj

    def _sampling(self, phi_z_t, h):

        with torch.no_grad():

            node_state = torch.cat([phi_z_t, h], dim=1)

            diff = node_state.unsqueeze(1)-node_state.unsqueeze(0)

            log_alpha = self.output_alpha(diff)
            prob_alpha = F.softmax(torch.sum(log_alpha,dim=1), -1)  
            alpha = torch.multinomial(prob_alpha, 1).long()  

            log_theta = self.output_theta(diff)

            probs = torch.sigmoid(torch.cat([log_theta[i, :, alpha[i]] for i in range(
                self.max_num_nodes)], dim=1))  # [N(t+1) x N(t+1)]

            Adj = torch.bernoulli(probs)

            return Adj 

    def forward(self, phi_z_t, h, is_sampling=False):

        if is_sampling == False:

            log_alpha, log_theta, adj = self._inference(phi_z_t, h)

            return log_alpha, log_theta, adj

        else:
            adj = self._sampling(phi_z_t, h)  

            return adj

