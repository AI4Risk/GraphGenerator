import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

########## submodule for LinkGenerator ##########
class GraphEncoder4Link(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(GraphEncoder4Link, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index) + x)
        x = F.leaky_relu(self.mlp(x) + x)
        return x
    
########## main module for LinkGenerator ##########
class LinkGenerator(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, device):
        super(LinkGenerator, self).__init__()
        self.device = device
        self.encoder = GraphEncoder4Link(in_features, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data, gen_data = None,lambd = 0.5):
        data = data.to(self.device)
        embed = self.encoder(data.x, data.edge_index)
        
        if gen_data != None:
            out = lambd*embed+(1-lambd)*gen_data
        else :
            out = embed
         
        return self.proj(out)

