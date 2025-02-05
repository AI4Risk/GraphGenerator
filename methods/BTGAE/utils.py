import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
import torch
from torch_geometric.data import Data
from scipy.sparse import csgraph, linalg
import networkx as nx
from scipy.sparse import lil_matrix
import random
import logging

def edge_overlap(A, B):
    return A.multiply(B).sum() / A.sum()

def weighted_edge_overlap(A, B):
    min_weights = sp.csr_matrix.minimum(A, B)
    overlap = min_weights.sum()/2
    return overlap

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)
    
def logPeakGPUMem(device):
    max_allocated = torch.cuda.max_memory_allocated(device=device)
    max_reserved = torch.cuda.max_memory_reserved(device=device)
    
    log(f"Peak GPU Memory Cached    : {max_reserved / (1024 ** 3):.2f} GB")
    log(f"Peak GPU Memory Allocated : {max_allocated / (1024 ** 3):.2f} GB")
    log(f"Peak GPU Memory Reserved  : {max_reserved / (1024 ** 3):.2f} GB")

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def csr_to_pyg_graph(csr_matrix, max_num_nodes=None):
    csr_matrix = csr_matrix + sp.eye(csr_matrix.shape[0])  # Add self-loops
    csr_matrix[csr_matrix > 1] = 1  # Clip values to 1
    coo = csr_matrix.tocoo()
    # First convert to np.array
    edge_index = np.vstack([coo.row, coo.col])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)
    
    # Compute node degrees from the csr_matrix
    node_degrees = np.array(csr_matrix.sum(axis=1)).flatten()  # Sum of each row to get degree
    
    # If max_num_nodes is not None, zero pad it
    if max_num_nodes is not None:
        
        node_degrees = np.pad(node_degrees, (0, max_num_nodes - len(node_degrees)), mode='constant')
    
    # Create a diagonal matrix of node degrees
    diag_matrix = np.diag(node_degrees)
    
    # Convert the diagonal matrix to a PyTorch tensor
    x = torch.tensor(diag_matrix, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    return data

def pad_sparse_matrix_with_numpy(sparse_matrix, max_num_nodes):
    """
    Pad a sparse matrix to a specified size using NumPy's pad function, then convert back to sparse.

    Args:
    - sparse_matrix (csr_matrix): The sparse matrix to be padded.
    - max_num_nodes (int): The desired number of rows and columns in the padded matrix.

    Returns:
    - csr_matrix: A new padded matrix in CSR format.
    """
    # Convert sparse matrix to dense array
    dense_matrix = sparse_matrix.toarray()

    # Calculate padding amounts
    pad_rows = max(0, max_num_nodes - dense_matrix.shape[0])
    pad_cols = max(0, max_num_nodes - dense_matrix.shape[1])

    # Pad the dense matrix using numpy's pad function
    padded_matrix = np.pad(dense_matrix, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

    return padded_matrix

def csr_delete_diagonal(csr_matrix):
    """
    Delete the diagonal elements of a CSR matrix.

    Args:
    - csr_matrix (csr_matrix): The input CSR matrix.

    Returns:
    - csr_matrix: A new CSR matrix with the diagonal elements removed.
    """
    # Convert the CSR matrix to a dense matrix
    dense_matrix = csr_matrix.toarray()

    # Set the diagonal elements to zero
    np.fill_diagonal(dense_matrix, 0)

    # Convert the dense matrix back to a CSR matrix
    return sp.csr_matrix(dense_matrix)

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


def _remove_isolated_nodes(adj_matrix):
    G = nx.from_scipy_sparse_array(adj_matrix)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    cleaned_adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=int)
    return cleaned_adj_matrix

def _extract_lccs(adj_matrix):
    G = nx.from_scipy_sparse_array(adj_matrix)
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc).copy()
    lcc_adj_matrix = nx.to_scipy_sparse_array(subgraph, format='csr', dtype=int)
    return lcc_adj_matrix

def prune(method, generated_graph):
    """
    Cleans the generated graphs based on the specified method.

    Parameters:
    - generated_graphs: List[csr_matrix]
        A list of generated graph adjacency matrices in CSR format.

    Returns:
    - List[csr_matrix]
        A list of cleaned adjacency matrices in CSR format.
    """
    if method == 'remove_isolated':
        return _remove_isolated_nodes(generated_graph)
    elif method == 'extract_lcc':
        return _extract_lccs(generated_graph)
    elif method == 'no_prune':
        return generated_graph
    else:
        raise ValueError("Unsupported cleaning method specified.")

########## Compute the first k eigenvalues of the Laplacian matrix ##########
def compute_eigen_csr(A_csr, k):
    # A_csr is expected to be a csr_matrix

    # Step 1: Compute the Laplacian Matrix L
    L = csgraph.laplacian(A_csr, normed=False)
    if L.dtype.char not in ['f', 'd']:
        L = L.astype('float64')

    # Step 2: Compute the first k smallest eigenvalues and their corresponding eigenvectors
    # 'eigsh' is used because L is symmetric; it finds 'k' smallest eigenvalues efficiently
    # which='SM' indicates the smallest magnitude eigenvalues
    tk = min(k, A_csr.shape[0]-1)

    # Set a higher number of max iterations and a smaller tolerance
    maxiter = L.shape[0] * 20  # Adjust this as needed, default is usually 10*n
    tol = 1e-5  # Smaller tolerance

    try:
        # Compute the first k smallest eigenvalues and their corresponding eigenvectors
        eigenvalues, eigenvectors = linalg.eigsh(L, k=tk, which='SM', maxiter=maxiter, tol=tol)
    except linalg.ArpackNoConvergence as e:
        print("ARPACK did not converge! Returning the eigenvalues/eigenvectors that did converge.")
        eigenvalues, eigenvectors = e.eigenvalues, e.eigenvectors

    dim = eigenvalues.shape[0]
    if dim < k:
        eigenvalues = np.pad(eigenvalues, (0, k - dim), mode='constant', constant_values=0)
        eigenvectors = np.pad(eigenvectors, ((0, 0), (0, k - dim)), mode='constant', constant_values=0)

    return eigenvalues
