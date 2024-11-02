import scipy.sparse as sp
from eval_tools.stats1graph import *
import os
from os.path import join, abspath
import json

def compute_graph_statistics(A, A_dir=None):
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
    
    if A_dir is None:
        num_edges = A.nnz // 2
    else:
        num_edges = A_dir.nnz
        
    statistics = {
        'n_nodes': A.shape[0],
        'n_edges': num_edges,
        "d_max": max_degree(A),
        "d_min": min_degree(A),
        "d_mean": average_degree(A),
        "gini": gini(A),
        "power_law_exp": power_law_alpha(A),
        # "LCC": LCC(A),
        "wedge_count": wedge_count(A),
        "claw_count": claw_count(A),
        # "triangle_count": triangle_count(A),
        # "square_count": square_count(A),
        # "rel_edge_distr_entropy": edge_distribution_entropy(A),
        # "assortativity": assortativity(A),
        # "clustering_coefficient": clustering_coefficient(A),
        # "n_component": n_component(A),
        # "cpl": cpl(A),  # 性能原因
    }
    
    statistics = {k: int(v) if isinstance(v, int) else round(float(v), 4) for k, v in statistics.items()}
    return statistics


if __name__ == '__main__':
    
    prefix = abspath(join(os.path.dirname(__file__), '..', 'data'))

    datasets = ['citeseer', 'cora', 'pubmed', 'Epinions', 'google', 'YelpChi']
    
    print('Computing graph statistics..., note that the graphs are undirected.')
    for dataset in datasets:
        data_dir = join(prefix, dataset)
        
        undirected_path = join(data_dir, f'{dataset}_undirected_csr.npz')
        directed_path = join(data_dir, f'{dataset}_directed_csr.npz')
        
        A = sp.load_npz(undirected_path)
        A_dir = None
        if os.path.exists(directed_path):
            A_dir = sp.load_npz(directed_path)
        
        print('\nComputing statistics for', dataset)
        statistics = compute_graph_statistics(A, A_dir)
        
        print(f'{dataset}: {json.dumps(statistics, indent=4)}')