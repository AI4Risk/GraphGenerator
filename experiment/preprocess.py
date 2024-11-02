import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from itertools import count
import pickle as pkl
import re
import os
from os import path
from os.path import join
import zipfile

########## csv file ##########
DATA_PATH = join(path.dirname(__file__), '..', 'data')
DATA_FILE_PATH = {
    'citeseer': join(DATA_PATH, 'citeseer/citeseer.cites'),
    'cora': join(DATA_PATH, 'cora/cora.cites'),
    'Epinions': join(DATA_PATH, 'Epinions/soc-Epinions1.txt'),
    'google': join(DATA_PATH, 'google/web-Google.txt'),
}

FEATURES_FILE_PATH = {
    'citeseer': join(DATA_PATH, 'citeseer/citeseer.content'),
    'cora': join(DATA_PATH, 'cora/cora.content'),
}


########## tool functions ##########
def reorder_index(src: list, tgt: list) -> tuple:
    """
    Reorder the index of the nodes in the graph.
    begin with 0.
    """
    
    all_nodes = np.unique(np.concatenate((src, tgt)))
    
    cnt = count()
    # the nodes are strings
    mapper = { str(node): next(cnt) for node in all_nodes }
    src = [mapper[str(s)] for s in src]
    tgt = [mapper[str(t)] for t in tgt]
    
    return src, tgt, mapper


########## preprocess csv files ##########


def from_csv_to_digraph(df: pd.DataFrame):
    """
    Convert the csv file to a sparse matrix.
    """
    
    src = df[0].values
    tgt = df[1].values
    src, tgt, mapper = reorder_index(src, tgt)
    
    coo = list(zip(src, tgt))
    # deduplicate
    coo_set = set(coo)
    G = nx.DiGraph(coo_set)
    # G.remove_edges_from(nx.selfloop_edges(G))
    
    return G, mapper

def from_csv_to_features(df: pd.DataFrame, mapper: dict, num_nodes=None):
    """
    Convert the csv file to a sparse matrix.
    """
    if num_nodes is None:
        num_nodes = len(mapper)
        
    feat = sp.lil_matrix((num_nodes, df.shape[1] - 2))
    
    for i in range(df.shape[0]):
        node = df.iloc[i, 0]
        assert node in mapper, f'Node {node} is not in the graph.'
        
        feat[mapper[node], :] = df.iloc[i, 1:-1]

    
    feat = feat.tocsr()
    
    return feat

def preprocess_csv(file_name):
    """
    Preprocess the csv file.
    """
    # file path
    path_prefix = path.join(DATA_PATH, file_name)
    directed_path = path.join(path_prefix, f'{file_name}_directed_csr.npz')
    undirected_path = path.join(path_prefix, f'{file_name}_undirected_csr.npz')
    features_path = None
    
    if file_name in FEATURES_FILE_PATH: # if the dataset has features
        features_path = path.join(path_prefix, f'{file_name}_features.npz')
    
    if path.exists(directed_path) and path.exists(undirected_path) and \
        (features_path is None or path.exists(features_path)):
            
        print(f'{file_name} has been preprocessed.')
        return
    
    # read csv file
    df = pd.read_csv(DATA_FILE_PATH[file_name], sep='\t', header=None, comment='#')
    G, mapper = from_csv_to_digraph(df)
    
    # convert to csr matrix
    directed_csr = nx.to_scipy_sparse_array(G, format='csr')
    undirected_csr = nx.to_scipy_sparse_array(G.to_undirected(), format='csr')
    
    # save csr matrix to npz file
    sp.save_npz(directed_path, directed_csr)
    sp.save_npz(undirected_path, undirected_csr)
    
    # save features if exists
    if features_path is not None:
        df_feat = pd.read_csv(FEATURES_FILE_PATH[file_name], sep='\t', header=None, dtype={0: str})
        
        features = from_csv_to_features(df_feat, mapper, G.number_of_nodes())
        sp.save_npz(features_path, features)


########## pubmed ##########
def preprocess_pubmed():
    """
    Preprocess the pubmed dataset.
    """
    # file path
    path_prefix = path.join(DATA_PATH, 'pubmed')
    directed_path = path.join(path_prefix, 'pubmed_directed_csr.npz')
    undirected_path = path.join(path_prefix, 'pubmed_undirected_csr.npz')
    features_path = path.join(path_prefix, 'pubmed_features.npz')
    
    if path.exists(directed_path) and path.exists(undirected_path) and path.exists(features_path):
        print('pubmed has been preprocessed.')
        return
    
    # preprocess the structure
    data_file = join(path_prefix, 'Pubmed-Diabetes.DIRECTED.cites.tab')
    with open(data_file, 'r') as f:
        lines = f.readlines()
        lines = lines[2:] # skip the first two lines, which are comments
        
    src, tgt = [], []
    
    for line in lines:
        num = re.findall(r'\d+', line)
        src.append(num[1])
        tgt.append(num[2])
        
    src, tgt, mapper = reorder_index(src, tgt)

    coo = list(zip(src, tgt))
    
    G = nx.DiGraph(coo)
    
    sp.save_npz(directed_path, nx.to_scipy_sparse_array(G, format='csr'))
    sp.save_npz(undirected_path, nx.to_scipy_sparse_array(G.to_undirected(), format='csr'))
    
    # preprocess the features
    features_file = join(path_prefix, 'Pubmed-Diabetes.NODE.paper.tab')
    with open(features_file, 'r') as f:
        lines = f.readlines()
        cat = re.findall(r'w-\w+', lines[1])
        vec_map = {c: i for i, c in enumerate(cat)}
        lines = lines[2:]
    
    
    feat = sp.lil_matrix((G.number_of_nodes(), len(cat)))
    
    for line in lines:
        line = line.split('\t')
        node = line[0]
        assert node in mapper, f'Node {node} is not in the graph.'
        
        for attr_val in line[1:]:
            attr, val = attr_val.split('=')
            if attr in vec_map:
                feat[mapper[node], vec_map[attr]] = val
    
    sp.save_npz(features_path, feat.tocsr())
    

########## extract all zip files ##########
def extract_all_zip_files():
    """
    Extract all zip files in the data directory.
    """
    dir_names = os.listdir(DATA_PATH)
    
    for dir_name in dir_names:
        dir_path = path.join(DATA_PATH, dir_name)
        zip_file = path.join(dir_path, 'data.zip')
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(dir_path)
            

if __name__ == '__main__':
    # extract all zip files
    extract_all_zip_files()
    
    # csv file
    for file_name in DATA_FILE_PATH.keys():
        print(f'Preprocessing {file_name}...')
        preprocess_csv(file_name)
    
    # pubmed
    print('Preprocessing pubmed...')
    preprocess_pubmed()
    
        
    print('Done.')