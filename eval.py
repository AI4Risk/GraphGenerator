import scipy.sparse as sp
from eval_tools.stats1graph import *

def compute_graph_statistics(A):
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
    statistics = {
        'n_nodes': A.shape[0],
        'n_edges': A.nnz,
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
    
    return statistics


if __name__ == '__main__':
    dic = {
        'google': 'data/google/google_graph_csr.npz',
        'citeseer': 'data/citeseer/citeseer_graph_csr.npz',
        'cora': 'data/cora/cora_graph_csr.npz',
        'pubmed': 'data/pubmed/pubmed_graph_csr.npz'
    }
    print('Computing graph statistics..., note that the graphs are undirected.')
    for dataset, path in dic.items():
        A = sp.load_npz(path)
        statistics = compute_graph_statistics(A)
        
        print(f'{dataset}: {statistics}')
    