import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality

def max_degree(A):
    """
    Compute the maximum degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Maximum degree.
    """
    degrees = A.sum(axis=-1)
    return np.max(degrees)


def min_degree(A):
    """
    Compute the minimum degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Minimum degree.
    """
    degrees = A.sum(axis=-1)
    return np.min(degrees)


def average_degree(A):
    """
    Compute the average degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Average degree.
    """
    degrees = A.sum(axis=0)
    return np.mean(degrees)


def LCC(A):
    """
    Compute the size of the largest connected component (LCC).
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Size of the largest connected component.
    """
    G = nx.Graph(A)
    return max([len(c) for c in nx.connected_components(G)])


def n_component(A):
    """
    Compute the number of connected components (N-Component).
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Number of connected components.
    """
    G = nx.Graph(A)
    return len(list(nx.connected_components(G)))


def wedge_count(A):
    """
    Compute the wedge count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Wedge count.
    """
    degrees = np.array(A.sum(axis=-1))
    return 0.5 * np.dot(degrees.T, degrees - 1).reshape([])


def claw_count(A):
    """
    Compute the claw count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Claw count.
    """
    degrees = np.array(A.sum(axis=-1))
    return 1 / 6 * np.sum(degrees * (degrees - 1) * (degrees - 2))


def triangle_count(A):
    """
    Compute the triangle count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Triangle count.
    """
    A_graph = nx.Graph(A)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def square_count(A):
    """
    Compute the square count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Square count.
    """
    A_squared = A @ A
    common_neighbors = sp.triu(A_squared, k=1).tocsr()
    num_common_neighbors = np.array(
        common_neighbors[common_neighbors.nonzero()]
    ).reshape(-1)
    return np.dot(num_common_neighbors, num_common_neighbors - 1) / 4


def power_law_alpha(A):
    """
    Compute the power law coefficient of the degree distribution of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Power law coefficient.
    """
    degrees = np.array(A.sum(axis=0)).flatten()
    degrees = degrees[degrees>0]
    return powerlaw.Fit(
        degrees, xmin=max(np.min(degrees), 1), verbose=False
    ).power_law.alpha


def power_law_exp(A_in,flow='out'):
    '''
    For digraph.
    Compute the same thing as power_law_alpha.
    '''
    if flow=='out':
        degrees=A_in.sum(axis=0)
    elif flow=='in':
        degrees=A_in.sum(axis=1)
    else:
        raise ValueError('This flow direction does not exist!')
    
    degrees = np.array(degrees).flatten()
    degrees = degrees[degrees>0]
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def gini(A, flow='out'):
    """
    Compute the Gini coefficient of the degree distribution of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Gini coefficient.
    """
    N = A.shape[0]
    if flow=='out':
        degrees = A.sum(axis=0)
    else:
        degrees = A.sum(axis=1)
    degrees_sorted = np.sort(np.array(degrees).flatten())
    return (
        2 * np.dot(degrees_sorted, np.arange(1, N + 1)) / (N * np.sum(degrees_sorted))
        - (N + 1) / N
    )


def edge_distribution_entropy(A):
    """
    Compute the relative edge distribution entropy of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Relative edge distribution entropy.
    """
    N = A.shape[0]
    degrees = np.array(A.sum(axis=-1)).flatten()
    degrees /= degrees.sum()
    eps = 1e-18
    return -np.dot(np.log(degrees + eps), degrees) / np.log(N)


def assortativity(A):
    """
    Compute the assortativity of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Assortativity.
    """
    G = nx.Graph(A)
    return nx.degree_assortativity_coefficient(G)


def clustering_coefficient(A):
    """
    Compute the global clustering coefficient of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Clustering coefficient.
    """
    n_wedges = wedge_count(A)
    if n_wedges == 0:
        return 0
    n_triangles = triangle_count(A)
    return 3 * n_triangles / n_wedges


def cpl(A):
    """
    Compute the characteristic path length of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Characteristic path length.
    """
    P = sp.csgraph.shortest_path(A)
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(bool)].mean()


def node_div_dist(A_in):
    
    out_degree=A_in.sum(axis=0) 
    in_degree=A_in.sum(axis=1) 
    
    max_degree=np.where(out_degree>in_degree,out_degree,in_degree)
    max_degree=np.where(max_degree>0,max_degree,1)
    
    return ((out_degree-in_degree)/max_degree).reshape(-1)


def deg_dist(A_in, flow='in'):
    if flow=='out':
        return A_in.sum(axis=1)
    elif flow=='in':
        return A_in.sum(axis=0)


def clus_dist(A_in):
    
    return list(nx.clustering(nx.from_scipy_sparse_array(A_in)).values())


#! Note: BC and CC needs long running time! 
def mean_betweeness_centrality(A_in): # Mean Betweeness Centrality
    G = nx.from_scipy_sparse_matrix(A_in)
    return np.mean(list(betweenness_centrality(G).values())) 
    

def mean_closeness_centrality(A_in): # Mean Closeness Centrality
    G = nx.from_scipy_sparse_matrix(A_in)
    return np.mean(list(closeness_centrality(G).values())) 