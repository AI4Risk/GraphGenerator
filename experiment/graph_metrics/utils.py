import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.stats as ss
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance, spearmanr

def _trans_graph_format(G):
    if isinstance(G, nx.classes.graph.Graph):
        return nx.to_scipy_sparse_array(G)
    elif isinstance(G, np.ndarray):
        return sp.csr_matrix(G)
    else:
        return G
    
def trans_format(X):
    if isinstance(X, list):
        return [_trans_graph_format(G) for G in X]
    else:
        return _trans_graph_format(X)
    

def calculate_mmd(x1, x2, beta):
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    L=pairwise_distances(x1,x2).reshape(-1)
    return np.exp(-beta*np.square(L))

def get_mean_var_std(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr,ddof=1)
    return arr_mean ,arr_std

def hellinger_distance(p, q):
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    
    distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2)) / np.sqrt(2)
    
    return distance

def JS_divergence(p,q):
    M = (p+q)/2
    return 0.5 * ss.entropy(p,M,base=2) + 0.5*ss.entropy(q,M,base=2)