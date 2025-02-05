import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from scipy.sparse import csr_matrix ,lil_matrix


########## submodules for  CGraphVAE ##########
class MLP_VAE(nn.Module):
    def __init__(self, hidden_dim, embedding_size, y_size, device, debug=False):
        super(MLP_VAE, self).__init__()
        self.device = device
        self.debug = debug

        # Encoder: Shared feature extraction
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, embedding_size),
            nn.LeakyReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU()
        )

        # Encoder heads: Separate layers for mu and logvar
        self.encode_mu = nn.Linear(embedding_size, embedding_size)
        self.encode_logvar = nn.Linear(embedding_size, embedding_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size + hidden_dim, embedding_size),
            nn.LeakyReLU(),
            nn.Linear(embedding_size, y_size)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, graph_emb, condition):    
        # Concatenate input features
        cat_feat = torch.cat((graph_emb, condition), dim=1)

        # Shared encoder
        hidden = self.shared_encoder(cat_feat)
        # Get mu and logvar
        z_mu = self.encode_mu(hidden)
        z_logvar = self.encode_logvar(hidden)
        z_sgm = torch.exp(z_logvar * 0.5)
           
        # Reparameterization trick
        eps = torch.randn_like(z_sgm).to(self.device)
        z = eps * z_sgm + z_mu

        # Decoder input
        cat_z = torch.cat((z, condition), dim=1)
        y = self.decoder(cat_z)
            
        return y, z_mu, z_logvar
    
class Projector(nn.Module):
    def __init__(self, N, H):
        super().__init__()
        self.linear1 = nn.Linear(N, H)
        self.linear2 = nn.Linear(H, H)
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x) + x)
        return x
    
class GraphEncoder4Subgraph(torch.nn.Module):
    # not used
    def __init__(self, in_channels, hidden_dim):
        super(GraphEncoder4Subgraph, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index) + x)
        x = global_mean_pool(x, batch)  # Aggregate node embeddings to graph embeddings
        return x

########## main module for CGraphVAE ##########
class CGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes,label_size, link_hidden_dim, device = 'cuda'):
        '''
        Args:
            input_dim: input feature dimension for node.
            
            latent_dim: dimension of the latent representation of graph.
        '''
        super(CGraphVAE, self).__init__()
        # config
        self.device = device
        self.label_size = label_size 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim    
        self.max_num_nodes = max_num_nodes
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2 # The number of elements in an upper triangular matrix.
        
        # models
        
        self.vae = MLP_VAE(hidden_dim, latent_dim, output_dim,device=self.device)
        
        self.graph_encoder = GraphEncoder4Subgraph(input_dim, hidden_dim)
        
        self.label_embedding = nn.Linear(label_size, hidden_dim)
        
        self.projector = Projector(max_num_nodes, link_hidden_dim)

        # activation fuctions
        self.f_act = nn.Softmax(dim=1)
        # self.f_act = nn.Sigmoid()
        # self.f_act = nn.ReLU()
        
    def forward(self, adj, graph_pyg, spectral):
        graph_emb = self.graph_encoder(graph_pyg.x, graph_pyg.edge_index, graph_pyg.batch)
        
        spectral_embed = self.label_embedding(spectral)
            
        h_decode, z_mu, z_lsgms = self.vae(graph_emb, spectral_embed)
        
        out = self.f_act(h_decode)

        adj_permuted = adj
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes))== 1].squeeze_()
        adj_vectorized_var = adj_vectorized.to(self.device)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
        
        loss = loss_kl + adj_recon_loss
        loss = loss * self.max_num_nodes

        adj_out = self.recover_full_adj_from_lower(self.recover_adj_lower(h_decode))
        adj_out = adj_out.detach()
        adj_out.requires_grad = True
        gen_emb = self.projector(adj_out)
        return loss, gen_emb

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_pred,adj_truth)/(self.max_num_nodes * self.max_num_nodes)

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes).to(self.device)
        
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag    

    def sample(self, num_samples, spectral):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        spectral_embed = self.label_embedding(spectral)
        cat_z = torch.cat((z, spectral_embed), 1)
        y = self.vae.decoder(cat_z)
        out = self.f_act(y)
        adj_out = self.recover_full_adj_from_lower(self.recover_adj_lower(y))
        gen_emb = self.projector(adj_out)
        return out, gen_emb

########## Graph Sampler ##########
class GraphSampler:
    def __init__(self, model = None):
        """
        Parameters:
        - method: str
            The method to use for cleaning the graphs. Options are 'remove_isolated' and 'extract_lcc'.
        """
        self.model = model

    def generate_single_graph(self, adj_input, spectral,max_num_node):
        target_edges = int(adj_input.sum())//2
        node_num = adj_input.shape[0]
        with torch.no_grad():
            out, gen_emb = self.model.sample(1,spectral)
            out = out.cpu()
            # Get the i-th sample's edge probabilities
            edge_probs = out[0]
            # n = int((- 1 + np.sqrt(1 + 8 * len(edge_probs))) / 2)  # Solve for n in n*(n-1)/2 = len(edge_probs)
            n = max_num_node
            adj_tmp = torch.zeros((n, n), dtype=torch.float32)
            triu_indices = torch.triu_indices(row=n, col=n, offset=0)
            adj_tmp[triu_indices[0], triu_indices[1]] = edge_probs
            adj_tmp = adj_tmp[:node_num, :node_num]
            # print(adj_tmp.shape)
            edge_probs = adj_tmp[torch.triu(torch.ones(node_num,node_num) ).bool()].squeeze_()      
            # print(edge_probs.shape)
            if node_num == 1:
                edge_probs = edge_probs.unsqueeze(0)

            # set the diagonal to 0 , edge_probs is a 1D tensor
            indices = [0]
            # start from n to 2
            for i in range(node_num, 1, -1):
                indices.append(indices[-1] + i)

            edge_probs[indices] = 0
            # Determine threshold to keep top target_edges
            threshold_value = torch.topk(edge_probs, k=target_edges)[0][-1]  # Get the k-th largest value            
            # Threshold the probabilities to get binary edges
            edge_binary = (edge_probs >= threshold_value).int()
            
            # Convert 1D edge representation back to a 2D adjacency matrix
            adj_matrix = torch.zeros((node_num, node_num), dtype=torch.int)
            triu_indices = torch.triu_indices(row=node_num, col=node_num, offset=0)
            adj_matrix[triu_indices[0], triu_indices[1]] = edge_binary

            # Make the adjacency matrix symmetric to represent an undirected graph
            adj_matrix = adj_matrix + adj_matrix.t() - torch.diag(adj_matrix.diagonal())*2 # no self loops
            
            # Convert to csr_matrix and add to the list
            tmp_graph = csr_matrix(adj_matrix.numpy())
            
        return tmp_graph, gen_emb

    def sample(self, upper_bound, graphs_train,spectrals):
        self.model.eval()
        generated_graphs = []
        gen_embs = []
        max_num_nodes = max([g.shape[0] for g in graphs_train])
        with torch.no_grad():
            for i in range(len(graphs_train)):
                tmp_graph, gen_emb = self.generate_single_graph(graphs_train[i],spectrals[i],max_num_nodes) 
                
                generated_graphs.append(tmp_graph)
                gen_embs.append(gen_emb)
        self.model.train()
        return generated_graphs, gen_embs   


