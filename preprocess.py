import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from itertools import count
import pickle as pkl
import os

########## csv file ##########
path_csv_file = {
    'google': 'data/google/web-Google.txt',
}

def reorder_index(src: list, tgt: list) -> tuple:
    """
    Reorder the index of the nodes in the graph.
    """
    
    all_nodes = np.unique(np.concatenate((src, tgt)))
    num_nodes = len(all_nodes)
    
    cnt = count()
    mapper = { node: next(cnt) for node in all_nodes }
    src = [mapper[s] for s in src]
    tgt = [mapper[t] for t in tgt]
    
    return src, tgt, num_nodes

def from_csv_to_coo(df: pd.DataFrame) -> sp.coo_matrix:
    """
    Convert the csv file to a sparse matrix.
    """
    
    src = df[0].values
    tgt = df[1].values
    src, tgt, num_nodes = reorder_index(src, tgt)
    
    undirected_src = np.concatenate((src, tgt))
    undirected_tgt = np.concatenate((tgt, src))
    coo = list(zip(undirected_src, undirected_tgt))
    
    # deduplicate
    coo_set = set(coo)
    
    src = [s for s, _ in list(coo_set)]
    tgt = [t for _, t in list(coo_set)]
    
    sp_adj = sp.coo_matrix((np.ones(len(src)), (src, tgt)), 
                           shape=(num_nodes, num_nodes))
    # deduplicate
    
    
    return sp_adj

def preprocess_csv(file_name):
    """
    Preprocess the csv file.
    """
    
    path_npz = os.path.join('data', file_name, f'{file_name}_graph_csr.npz')
    if os.path.exists(path_npz):
        print(f'{file_name} has been preprocessed.')
        return
    
    df = pd.read_csv(path_csv_file[file_name], sep='\t', header=None, comment='#')
    sp_adj = from_csv_to_coo(df)
    sp.save_npz(path_npz, sp_adj.tocsr())
    

########## cora, citeseer, pubmed ##########

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_ind(dataset = 'cora'):
    """
    preprocess cora, citeseer, pubmed dataset.
    save the graph and features as sparse matrix.
    """
    
    path_prefix = os.path.join('data', dataset)
    path_graph_csr = os.path.join(path_prefix, '{}_graph_csr.npz'.format(dataset))
    path_features = os.path.join(path_prefix, '{}_features.npz'.format(dataset))
    
    if os.path.exists(path_graph_csr) and os.path.exists(path_features):
        print(f'{dataset} has been preprocessed.')
        return
    
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        file_path = os.path.join(path_prefix, "ind.{}.{}".format(dataset, names[i]))
        load = pkl.load(open(file_path, 'rb'), encoding='latin1')
        objects.append(load)
        
    x, tx, allx, graph = tuple(objects)
    
    test_idx_reorder = parse_index_file(os.path.join(path_prefix, "ind.{}.test.index".format(dataset)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.tocsr()

    G = nx.from_dict_of_lists(graph)
    adj = nx.to_scipy_sparse_array(G, format='csr')
    # save csr matrix to npz file
    sp.save_npz(path_graph_csr, adj)
    sp.save_npz(path_features, features)
    

if __name__ == '__main__':
    # csv file
    for file_name in path_csv_file.keys():
        print(f'Preprocessing {file_name}...')
        preprocess_csv(file_name)
    
    # cora, citeseer, pubmed
    for dataset in ['cora', 'citeseer', 'pubmed']:
        print(f'Preprocessing {dataset}...')
        preprocess_ind(dataset)
        
    print('Done.')