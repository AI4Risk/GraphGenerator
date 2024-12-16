import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class Time2Vec(nn.Module):
    def __init__(self,activation,time_embed_size,device='cpu'):
        super(Time2Vec,self).__init__()
       
        if activation=="sin":
            self.func=torch.sin
        elif activation=="cos":
            self.func=torch.cos
        
        self.time_emb_size=time_embed_size
        self.tau_to_embed_zero=nn.Linear(1,1).to(device) 
        self.tau_to_embed_periodic=nn.Linear(1,self.time_emb_size-1).to(device) 
        self.W = nn.Parameter(torch.randn(self.time_emb_size)).to(device)
        
    
    def forward(self,t):
        
        f_0=self.tau_to_embed_zero(t)
        f_r=self.func(self.tau_to_embed_periodic(t))
        
        t_vec=torch.cat([f_0,f_r],axis=-1)
        t_vec=t_vec*self.W
        
        return t_vec

class BidirectionalEncoder(nn.Module):

    def __init__(self, in_dim,
                 hid_dim,
                 encode_dim,
                 layer_num=3,
                 device='cpu'):
       
        super(BidirectionalEncoder, self).__init__()

        self.device = device
        self.layer_num = layer_num  # 图编码器层数

        self.linear = nn.Linear(in_dim, hid_dim).to(device)

       
        self.in_flow_encoder = [
            GINLayer(hid_dim, hid_dim, device=self.device) for _ in range(self.layer_num)]

        
        self.out_flow_encoder = [
            GINLayer(hid_dim, hid_dim, device=self.device) for _ in range(self.layer_num)]

        self.aggregator = nn.Linear(2*hid_dim, encode_dim).to(device)
        
        self.pooling = nn.Linear(encode_dim*layer_num, encode_dim).to(device)

    def forward(self, x, A):

        self.hierarchy = []  # 层次信息

        x = self.linear(x)

        for layer_id in range(self.layer_num):

            x_in = self.in_flow_encoder[layer_id](x, A)

            x_out = self.out_flow_encoder[layer_id](x, A.T)

            biflow = torch.cat([x_in, x_out], dim=1)
            biflow = F.leaky_relu(self.aggregator(biflow))
            self.hierarchy.append(biflow)

        global_flow = torch.cat(self.hierarchy, dim=1)
        encoded_states = self.pooling(global_flow)

        return encoded_states

class FCLayer(nn.Module):
   
    def __init__(self, in_size, out_size, alpha=0.2, dropout=0.3, 
                 b_norm=True, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.is_norm=b_norm 
        self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = nn.LeakyReLU(alpha)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.activation(self.linear(x))
        h = self.dropout(h)
        if self.is_norm:
            h = F.elu(self.b_norm(h))
        else:
            h=F.elu(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'

class MLP(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, layers, alpha=0.2,b_norm=True,
                 dropout=0.3, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, alpha=alpha, b_norm=b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, alpha=alpha, b_norm=b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, alpha=alpha,
                                                    b_norm=b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, alpha=alpha, b_norm=b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'

class GINLayer(nn.Module):

    def __init__(self, in_features, out_features, fc_layers=2, hid_mlp=16, 
                 alpha=0.2,b_norm=True,dropout=0.3,device='cpu'):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param hid_features:    size of the hidden_size per node in MLP
        :param fc_layers:       number of fully connected layers after the sum aggregator
        :param device:          device used for computation
        """
        super(GINLayer, self).__init__()

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = nn.Parameter(torch.zeros(size=(1,), device=device))
        self.post_transformation = MLP(in_size=in_features, hidden_size=hid_mlp,
                                       out_size=out_features, layers=fc_layers, alpha=alpha,b_norm=b_norm,
                                        dropout=dropout, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        self.epsilon.data.fill_(0.1)

    def forward(self, input, adj):
        N= adj.shape[0]

        # sum aggregation
        mod_adj = adj + torch.eye(N, device=self.device) * (1 + self.epsilon)
        support = torch.matmul(mod_adj, input)
        
        # post-aggregation transformation
        return self.post_transformation(support)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
               
# 多层GIN
class GIN(nn.Module):
    def __init__(self, in_features, hid_features, out_features, fc_layers=2, hid_mlp=16, 
                 alpha=0.2,b_norm=True,dropout=0.3,layer_num=2,device='cpu'):
        
        super(GIN, self).__init__()
        self.dropout=dropout
        self.in_layer=GINLayer(in_features, hid_features,fc_layers=fc_layers, hid_mlp=hid_mlp, 
                 alpha=alpha,b_norm=b_norm,dropout=dropout,device=device)
        
        self.stacks=[GINLayer(hid_features, hid_features,fc_layers=fc_layers, hid_mlp=hid_mlp, 
                 alpha=alpha,b_norm=b_norm,dropout=dropout,device=device) for i in range(layer_num-1)]
        
        self.out_layer=GINLayer(hid_features, out_features,fc_layers=fc_layers, hid_mlp=hid_mlp, 
                 alpha=alpha,b_norm=b_norm,dropout=dropout,device=device)
    
    def forward(self,x,adj):
        x=self.in_layer(x,adj)
        for GNNLayer in self.stacks:
            x=GNNLayer(x,adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(x, adj)
        return x
    
class GATLayer(nn.Module):
    
    def __init__(self,in_features, out_features, dropout, alpha, concat=True) -> None:
        super(GATLayer,self).__init__()
        self.in_features = in_features   
        self.out_features = out_features   
        self.dropout = dropout    
        self.alpha = alpha     
        self.concat = concat   
        
        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_normal_(self.W.data,gain=1.414) 
        # attention
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_normal_(self.a.data,gain=1.414)
        self.leakyrelu=nn.LeakyReLU(self.alpha)    
        
        self.bn=nn.BatchNorm1d(out_features) 
    
    def forward(self,inp,adj):
 
        h=torch.mm(inp,self.W) # [N,out_features]
        N=h.size(0) 
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)   
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -1e12 * torch.ones_like(e)    
        attention = torch.where(adj>0, e, zero_vec)  
        
        attention = F.softmax(attention, dim=1)    
        attention = F.dropout(attention, self.dropout, training=self.training)   
        h_prime = self.bn(torch.matmul(attention, h))  
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
class GAT(nn.Module):
    def __init__(self, in_features, hid_features, out_features, 
                 alpha=0.2,dropout=0.3,layer_num=2,device='cpu'):
       
        super(GAT, self).__init__()
        self.dropout=dropout
        self.in_layer=GATLayer(in_features, hid_features,
                 dropout=dropout,alpha=alpha).to(device)
        
        self.stacks=[GATLayer(hid_features, hid_features,
                 dropout=dropout,alpha=alpha).to(device) for i in range(layer_num-1)]
        
        self.out_layer=GATLayer(hid_features, out_features,
                 dropout=dropout,alpha=alpha).to(device)
    
    def forward(self,x,adj):
        x=self.in_layer(x,adj)
        for GNNLayer in self.stacks:
            x=GNNLayer(x,adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(x, adj)
        return x