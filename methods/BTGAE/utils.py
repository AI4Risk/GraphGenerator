import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
import torch
from torch_geometric.data import Data
from scipy.sparse import csgraph, linalg
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
