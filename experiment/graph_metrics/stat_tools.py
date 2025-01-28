from .metrics import *
from .utils import trans_format
import time
from concurrent.futures import ProcessPoolExecutor

global_A = None

def cal_metrics(metric):
    return metric(global_A)

def compute_statistics(A, A_dir=None):
    """
    Compute a selection of graph statistics for the input graph.
    
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
          
    Returns:
        Dictionary containing the following statistics:
                 * Maximum, minimum, mean degree of nodes
                 * Size of the largest connected component (LCC)
                 * Wedge count
                 * Claw count
                 * Triangle count
                 * Square count
                 * Power law exponent
                 * Gini coefficient
                 * Relative edge distribution entropy
                 * Assortativity
                 * Clustering coefficient
                 * Characteristic path length
    """
    A = trans_format(A)
    A = A + A.T
    A[A > 1] = 1
    
    if isinstance(A, list):
        # return [compute_statistics(a) for a in A] # compute statistics for each graph in the list
        raise ValueError("Input graph must be a single graph, not a list of graphs.")
    
    A.indptr = A.indptr.astype('int32')
    A.indices = A.indices.astype('int32')
    A.data = A.data.astype('float64')
    
    if A_dir is None:
        num_edges = A.nnz // 2
    else:
        num_edges = A_dir.nnz

    statistics = {
        "d_max": max_degree,
        "d_min": min_degree,
        "d_mean": average_degree,
        "n_component": n_component,
        "LCC": LCC,
        "wedge_count": wedge_count,
        "claw_count": claw_count,
        "triangle_count": triangle_count,
        "power_law_exp": power_law_alpha,
        "gini": gini,
        "rel_edge_distr_entropy": edge_distribution_entropy,
        "assortativity": assortativity,
        }
    if A.shape[0] < 100000: # avoid computing these statistics for large graphs
        statistics["square_count"] = square_count
        statistics["cpl"] = cpl
    
    global global_A
    global_A = A
    metric_list = list(statistics.values())
    
    time_start = time.time()
    with ProcessPoolExecutor() as executor:
        results = executor.map(cal_metrics, metric_list)
        res = {k: v for k, v in zip(statistics.keys(), results)}
    
    res["n_nodes"] = A.shape[0]
    res["n_edges"] = num_edges
    
    # avoid computing wedge_count and triangle_count twice
    n_wedges = res["wedge_count"]
    n_triangles = res["triangle_count"]
    if n_wedges == 0:
        global_cluster_coef = 0
    else: 
        global_cluster_coef = 3 * n_triangles / n_wedges 
    res["global_cluster_coef"] = global_cluster_coef
    
    time_end = time.time()
    print(f"Time taken to compute statistics: {time_end - time_start} seconds")
    
    res = {k: int(v) if isinstance(v, int) else round(float(v), 4) for k, v in res.items()}
    return res
