import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack, hstack
import scipy.sparse as sp
from .utils import csr_to_pyg_graph
import torch

########## main module for GraphReconstructor ##########
class GraphReconstructor:
    def __init__(self,method = 'random',BiGAES = None):
        """
        Initializes the GraphReconstructor instance.
        """
        self.method = method
        self.addition_method = {
            'random': self._add_additional_edges_random
            # Add other partitioning methods here as needed.
        }
        self.BiGAES = BiGAES
    
    def addition(self,big_graph, additional_edges):
        
        addition_method = self.addition_method.get(self.method)
        if not addition_method:
            raise ValueError(f"Partition method {self.method} not recognized.")
        return addition_method(big_graph, additional_edges)

    def aggregate_subgraphs_random(self, subgraphs, additional_edges):
        """
        Aggregates subgraphs into a single large graph, adding a specified number of additional edges.

        Parameters:
        - subgraphs: List[csr_matrix]
            A list of subgraph adjacency matrices in CSR format.
        - additional_edges: int
            The total number of additional edges to add between subgraphs.

        Returns:
        - csr_matrix: The adjacency matrix of the aggregated graph in CSR format.
        """
        # Step 1: Stack subgraphs vertically and horizontally to form the larger sparse matrix
        big_graph = lil_matrix((0, 0))
        for subgraph in subgraphs:
            big_graph = self._expand_and_add(big_graph, subgraph)

        # Step 2: Distribute additional edges
        if additional_edges > 0:
            big_graph = self.addition(big_graph, additional_edges)

        return big_graph.tocsr()

    def sample(self, i,j,subgraph_i, subgraph_j,n,m):
        """
        Generate a new adjacency matrix for the given pair of subgraphs.

        Args:
        - subgraph_i (scipy.sparse.csr_matrix): The adjacency matrix of the first subgraph.
        - subgraph_j (scipy.sparse.csr_matrix): The adjacency matrix of the second subgraph.

        Returns:
        - scipy.sparse.csr_matrix: The predicted adjacency matrix for the pair of subgraphs.
        """
        
        with torch.no_grad():
            z_i = self.BiGAES[i](subgraph_i)
            z_j = self.BiGAES[j](subgraph_j)
            output = torch.mm(z_i,z_j.t())
            output = torch.sigmoid(output)

            output = output[:n, :m]
            # print(output.shape,n,m)
        return output

    def aggregate_subgraphs_synth(self, adj, subgraphs, max_num_nodes):    
        for biage in self.BiGAES:
            biage.eval()
        
        num_graphs = len(subgraphs)
        final_adj_blocks = [[None for _ in range(num_graphs)] for _ in range(num_graphs)]
        subgraph_list = [csr_to_pyg_graph(subgraph, max_num_nodes) for subgraph in subgraphs]
        numlist = [subgraph.shape[0] for subgraph in subgraphs]
        row = 0
        for i in range(num_graphs):
            final_adj_blocks[i][i] = subgraphs[i]
        for i in range(num_graphs):
            num_i = subgraphs[i].shape[0]
            col = row+num_i
            for j in range(i+1,num_graphs):
                prob = self.sample(i,j,subgraph_list[i],subgraph_list[j],numlist[i],numlist[j])
                num_j = subgraphs[j].shape[0]
                num_edges = int(adj[row:row+num_i, col:col+num_j].sum())
                edge_probability = prob / prob.sum()
                # print(num_edges)
                top = self.create_top_k_adjacency(num_i, num_j, num_edges, edge_probability)
                # print(top)
                # Assign the generated matrices to the appropriate blocks
                # print(top.sum(),num_i,num_j,num_edges)
                final_adj_blocks[i][j] = top
                final_adj_blocks[j][i] = top.transpose()
                col += num_j
            row += num_i
        final_adj = sp.bmat(final_adj_blocks, format='csr')
        return final_adj
    
    def aggregate_subgraphs_cellsbm(self, super_adj, subgraphs):
        num_subgraphs = len(subgraphs)

        # Initialize a 2D list structure for the blocks, filled with None
        final_adj_blocks = [[None for _ in range(num_subgraphs)] for _ in range(num_subgraphs)]

        # Fill in the diagonal blocks with the original subgraphs
        for i in range(num_subgraphs):
            final_adj_blocks[i][i] = subgraphs[i]

        # Off-diagonal blocks: simulate edges based on the stochastic block model
        for i in range(num_subgraphs):
            for j in range(i + 1, num_subgraphs):
                num_edges_ij = int(super_adj[i, j])  # Number of edges between subgraphs i and j
                num_i, num_j = subgraphs[i].shape[0], subgraphs[j].shape[0]
                total_possible_connections = num_i * num_j
                edge_probability = num_edges_ij / total_possible_connections


                # Generate a random adjacency matrix for the connections between subgraphs i and j
                top = self.create_top_k_adjacency(num_i, num_j, num_edges_ij, edge_probability)
                # Assign the generated matrices to the appropriate blocks
                final_adj_blocks[i][j] = top
                final_adj_blocks[j][i] = top.transpose()

        # Assemble the final adjacency matrix from the structured block list
        final_adj = sp.bmat(final_adj_blocks, format='csr')

        return final_adj
    
    def _expand_and_add(self, big_graph, subgraph):
        """
        Expands the big graph and adds the subgraph to the bottom-right.
        """
        subgraph_lil = subgraph.tolil()
        new_rows = lil_matrix((subgraph.shape[0], big_graph.shape[1]))
        new_cols = lil_matrix((big_graph.shape[0] + subgraph.shape[0], subgraph.shape[1]))
        big_graph = vstack([big_graph, new_rows]).tolil()
        big_graph = hstack([big_graph, new_cols]).tolil()
        big_graph[-subgraph.shape[0]:, -subgraph.shape[1]:] = subgraph_lil
        return big_graph

    def _add_additional_edges_random(self, big_graph, additional_edges):
        """
        Randomly adds additional edges to the graph.
        """
        num_nodes = big_graph.shape[0]
        possible_edges = np.triu_indices(n=num_nodes, k=1)
        edge_indices = np.random.choice(range(len(possible_edges[0])), size=additional_edges, replace=False)

        for edge_index in edge_indices:
            i = possible_edges[0][edge_index]
            j = possible_edges[1][edge_index]
            big_graph[i, j] = big_graph[j, i] = 1

        return big_graph
    
    def create_top_k_adjacency(self,num_i, num_j, num_edges, edge_probability):
        """
        Create a sparse adjacency matrix where exactly `num_edges` are set to 1 based on the highest probabilities.

        Args:
        - num_i (int): Number of rows in the adjacency matrix.
        - num_j (int): Number of columns in the adjacency matrix.
        - num_edges (int): Number of edges (entries set to 1) in the adjacency matrix.
        - edge_probability (float): Base probability used to generate edge weights.

        Returns:
        - csr_matrix: The generated sparse adjacency matrix in CSR format.
        """
        if num_edges > num_i * num_j:
            raise ValueError("Number of edges exceeds the number of possible matrix entries.")
        # Create an adjacency matrix of zeros

        # problem is here
        if num_edges == 0:
            return csr_matrix((num_i, num_j), dtype=int)

        # Flatten the matrix and get the indices of the top k highest values
        flat_probabilities = edge_probability.flatten().cpu()
        top_k_indices = np.argpartition(flat_probabilities, -num_edges)[-num_edges:]

        flat_adjacency = np.zeros_like(flat_probabilities, dtype=int)
        
        # Set the top k positions to 1
        flat_adjacency[top_k_indices] = 1

        # Reshape to the original matrix shape
        adjacency_matrix = flat_adjacency.reshape(num_i, num_j)

        # Convert to CSR format
        return csr_matrix(adjacency_matrix)
