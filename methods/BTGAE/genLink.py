import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

########## submodule for LinkGenerator ##########
class GraphEncoder4Link(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(GraphEncoder4Link, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv_layer_num = 5

        self.convs = torch.nn.ModuleList()
        for i in range(self.conv_layer_num):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        for i in range(self.conv_layer_num - 2):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
        x = self.convs[self.conv_layer_num - 2](x, edge_index, edge_weight) 
        return x
    
########## main module for LinkGenerator ##########
class LinkGenerator(torch.nn.Module):
    def __init__(self, in_features, hidden_dim ,max_num_nodes,device):
        super(LinkGenerator, self).__init__()
        self.device = device
        self.encoder_a = GraphEncoder4Link(in_features, hidden_dim)

    def forward(self, data, gen_data = None,lambd = 0.5):
        data = data.to(self.device)
        embed_a = self.encoder_a(data.x, data.edge_index, data.edge_attr,data.batch)
        
        if gen_data != None:
            out_a = lambd*embed_a+(1-lambd)*gen_data
        else :
            out_a = embed_a
       
        return out_a

