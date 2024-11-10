import torch
import numpy as np
import scipy.sparse as sp
import random
import copy
import logging
    
def edge_from_scores(scores_matrix, n_edges):
    degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.
    B = scores_matrix.shape[0]
    N = scores_matrix.shape[1]
    target_g = sp.lil_matrix(scores_matrix.shape)
    probs = copy.deepcopy(scores_matrix)
    for n in range(B):  # Iterate over the nodes
        target = np.random.choice(N, p=scores_matrix[n] / degrees[n], size=1)
        target_g[n, target] = 1
        probs[n, target] = 0
    diff = np.round(n_edges - target_g.sum())
    if diff > 0:
        probs = probs.reshape(-1)
        extra_edges = np.random.choice(probs.shape[0], replace=False, p=probs/probs.sum(), size=int(diff))
        target_g[extra_edges//N, extra_edges%N] = 1
    return target_g

def edge_overlap(A, B):
    """
    Compute edge overlap between two graphs (amount of shared edges).
    Args:
        A (sp.csr.csr_matrix): First input adjacency matrix.
        B (sp.csr.csr_matrix): Second input adjacency matrix.
    Returns:
        Edge overlap.
    """
    return A.multiply(B).sum() / A.sum()

def random_seed(seed=42):
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