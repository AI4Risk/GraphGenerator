import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv,global_mean_pool
from scipy.sparse import csr_matrix ,lil_matrix
import networkx as nx

########## submodules for  CGraphVAE ##########
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size,label_size, embedding_size, y_size,device,debug = False):
        super(MLP_VAE_plain, self).__init__()
        self.device = device
        self.debug = debug
        self.encode_11 = nn.Sequential(
            nn.Linear(h_size+label_size, embedding_size),
            nn.LayerNorm(embedding_size)  # Apply LayerNorm after the linear layer
        )
        self.encode_12 = nn.Sequential(
            nn.Linear(h_size+label_size, embedding_size),
            nn.LayerNorm(embedding_size)  # Apply LayerNorm after the linear layer
        )
        self.decode_1 = nn.Sequential(
            nn.Linear(embedding_size+label_size, embedding_size+label_size),
            nn.LayerNorm(embedding_size+label_size)
        )
        self.decode_2 = nn.Sequential(
            nn.Linear(embedding_size+label_size, y_size), # make edge prediction (reconstruct)
            nn.LayerNorm(y_size)
        )
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, graph, condition):
        cat_feat = torch.cat((graph, condition),1)

        # encode
        z_mu = self.encode_11(cat_feat)
        z_lsgms = self.encode_12(cat_feat)
        z_sgm = z_lsgms.mul(0.5).exp_()

        # sample
        eps = torch.randn(z_sgm.size()).to(self.device)
        z = eps * z_sgm + z_mu # reparameterization trick
        
        cat_z = torch.cat((z, condition), 1)
        # decode
        y = self.decode_1(cat_z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

class Projector(nn.Module):
    def __init__(self, N, H):
        super(Projector, self).__init__()
        self.linear = nn.Linear(N, H)  # Linear transformation for each row

    def forward(self, x):
        # x is expected to be of shape (N, N)
        output = self.linear(x)  # Apply linear layer to all rows at once
        return output

class GraphEncoder4Subgraph(torch.nn.Module):
    # not used
    def __init__(self, in_channels, hidden_dim):
        super(GraphEncoder4Subgraph, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
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
        self.latent_dim = latent_dim    
        self.max_num_nodes = max_num_nodes
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2 # The number of elements in an upper triangular matrix.
        
        # models
        self.vae = MLP_VAE_plain(input_dim * input_dim,label_size, latent_dim, output_dim,device=self.device)
        self.label_embedding = nn.Linear(label_size, label_size)
        self.projector = Projector(max_num_nodes, link_hidden_dim)

        # activation fuctions
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, adj, spectral):
        graph_h = adj.view(-1, self.max_num_nodes * self.max_num_nodes)
        spectral_embed = self.label_embedding(spectral)
        
        h_decode, z_mu, z_lsgms = self.vae(graph_h, spectral_embed)
        
        out = self.sigmoid(h_decode)

        adj_permuted = adj
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
        adj_vectorized_var = adj_vectorized.to(self.device)
  
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize

        loss = loss_kl + adj_recon_loss
        loss = loss * self.max_num_nodes

        adj_out = self.recover_adj_lower(h_decode)
        gen_emb = self.projector(adj_out)

        return loss, gen_emb

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_pred,adj_truth)/(self.max_num_nodes * self.max_num_nodes)

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes).to(self.device)
        # print(l.shape)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag    

    def sample(self, num_samples, spectral):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        spectral_embed = self.label_embedding(spectral)
        cat_z = torch.cat((z, spectral_embed), 1)
        y = self.vae.decode_1(cat_z)
        y = self.vae.relu(y)
        y = self.vae.decode_2(y)
        out = self.sigmoid(y)
        return out

########## Graph Sampler ##########
class GraphSampler:
    def __init__(self, method='remove_isolated',model = None):
        """
        Initializes the GraphPruner with a specified method.

        Parameters:
        - method: str
            The method to use for cleaning the graphs. Options are 'remove_isolated' and 'extract_lcc'.
        """
        self.method = method
        self.model = model

    def prune(self, generated_graph):
        """
        Cleans the generated graphs based on the specified method.

        Parameters:
        - generated_graphs: List[csr_matrix]
            A list of generated graph adjacency matrices in CSR format.

        Returns:
        - List[csr_matrix]
            A list of cleaned adjacency matrices in CSR format.
        """
        if self.method == 'remove_isolated':
            return self._remove_isolated_nodes(generated_graph)
        elif self.method == 'extract_lcc':
            return self._extract_lccs(generated_graph)
        else:
            raise ValueError("Unsupported cleaning method specified.")

    def _remove_isolated_nodes(self, adj_matrix):
        G = nx.from_scipy_sparse_array(adj_matrix)
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        cleaned_adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=int)
        return cleaned_adj_matrix

    def _extract_lccs(self, adj_matrix):
        G = nx.from_scipy_sparse_array(adj_matrix)
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()
        lcc_adj_matrix = nx.to_scipy_sparse_array(subgraph, format='csr', dtype=int)
        return lcc_adj_matrix
    
    def generate_single_graphs(self, adj_input, spectral,max_num_node):
        target_edges = int(adj_input.sum())//2
        node_num = adj_input.shape[0]
        with torch.no_grad():
            out = self.model.sample(1,spectral)
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
            
            #! prune the graph
            #! If we prune the isolated nodes, the nodes of pred will differ from the target.
            #! When pruning isolated nodes, do not calculate edge overlap.
            
            # tmp_graph = self.prune(tmp_graph)
            
        return tmp_graph

    def sample(self, upper_bound, graphs_train,spectrals):
        self.model.eval()
        generated_graphs = []
        max_num_nodes = max([g.shape[0] for g in graphs_train])
        with torch.no_grad():
            for i in range(len(graphs_train)):
                
                tmp_graph = self.generate_single_graphs(graphs_train[i],spectrals[i],max_num_nodes) 
                
                generated_graphs.append(tmp_graph)
                
        self.model.train()
        return generated_graphs       


def prune_and_pad_subgraphs_csr(subgraph_adj_list, t):
    """
    Prune and pad subgraphs represented as CSR adjacency matrices.

    Parameters:
    - subgraph_adj_list: List[csr_matrix]
        A list of subgraph adjacency matrices in CSR format.
    - t: int
        The minimum number of nodes for a subgraph to be kept.

    Returns:
    - List[csr_matrix]
        A list of pruned and padded CSR adjacency matrices.
    """
    # Filter out subgraphs with fewer than t nodes
    filtered_adj_list = [adj for adj in subgraph_adj_list if adj.shape[0] >= t]
    
    # Find the maximum number of nodes among the remaining subgraphs
    max_nodes = max(adj.shape[0] for adj in filtered_adj_list)
    
    # Pad subgraphs with fewer nodes than max_nodes
    padded_adj_list = []
    for adj in filtered_adj_list:
        if adj.shape[0] < max_nodes:
            # Convert to LIL format for easier manipulation
            padded_adj = lil_matrix((max_nodes, max_nodes))
            padded_adj[:adj.shape[0], :adj.shape[1]] = adj
            padded_adj_list.append(padded_adj.tocsr())  # Convert back to CSR format
        else:
            padded_adj_list.append(adj)
    
    return padded_adj_list
