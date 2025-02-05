import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io
from itertools import count
import pickle as pkl
import re
import os
from os import path
from os.path import join
import zipfile
from datetime import timedelta
from collections import defaultdict

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

MAT_FILE_PATH = {
    'YelpChi': join(DATA_PATH, 'YelpChi/YelpChi.mat'),
}

TEMPORAL_CONFIG = {
    'email': {'start_win':480, 'end_win':5, 'SLICE_DAYS':1, 'file_name':'email-dnc.edges'},
    'bitcoin': {'start_win':200, 'end_win':600, 'SLICE_DAYS':30, 'file_name':'soc-sign-bitcoinalpha.csv'},
    'vote': {'start_win':200, 'end_win':540, 'SLICE_DAYS':15, 'divide_num':20, 'file_name':'soc-wiki-elec.edges'}
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

########## mat file ##########
def preprocess_mat(file_name):
    """
    Preprocess the mat file.
    """
    # file path
    path_prefix = path.join(DATA_PATH, file_name)
    undirected_path = path.join(path_prefix, '{}_undirected_csr.npz'.format(file_name))
    features_path = path.join(path_prefix, '{}_features.npz'.format(file_name))
    if path.exists(undirected_path) and path.exists(features_path):
        print('{} has been preprocessed.'.format(file_name))
        return
    
    mat_contents = scipy.io.loadmat(MAT_FILE_PATH[file_name])
    
    sp.save_npz(undirected_path, mat_contents['homo'].tocsr())
    sp.save_npz(features_path, mat_contents['features'].tocsr())
    
########## temporal file ##########
def preprocess_temporal_data(data_name):
    conf = TEMPORAL_CONFIG
    dir_path = path.join(DATA_PATH, data_name)
    file_path = path.join(dir_path, conf[data_name]['file_name'])
    save_path = path.join(dir_path, f'{data_name}.pkl')
    if path.exists(save_path):
        print(f'{data_name} has been preprocessed.')
        return
    
    start_win = conf[data_name]['start_win']
    end_win = conf[data_name]['end_win']
    SLICE_DAYS = conf[data_name]['SLICE_DAYS']
    divide_num = conf[data_name].get('divide_num', 1)
    
    if data_name == 'vote':
        data=pd.read_csv(file_path, sep=' ')
    else:
        data=pd.read_csv(file_path)

    data['date']=pd.to_datetime(data['value'],unit='s')

    if data_name == 'email': # email data has no weight
        weights = [1]*len(data)
        links=list(zip(data['src'],data['dst'],weights,data['date'])) 
    else:
        links=list(zip(data['src'],data['dst'],data['weight'],data['date']))

    links.sort(key =lambda x: x[-1])

    if data_name == 'email':
        links=links[1:] # remove the first edge, because its time is far away from the others

    ts=[link[-1] for link in links] 

    START_DATE=min(ts)+timedelta(start_win) 
    END_DATE = max(ts)-timedelta(end_win) 
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    slice_links = defaultdict(lambda: nx.DiGraph())
    links_groups=defaultdict(lambda:[])
    for (a, b, v, time) in links:
        
        datetime_object = time
        
        if datetime_object<=START_DATE or datetime_object>END_DATE:
            continue
        
        slice_id = (datetime_object - START_DATE).days//SLICE_DAYS
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.DiGraph()
            if slice_id > 0:
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))
                # if data_name != 'email': # other data is too large, so it's hard to handle edge addition and deletion
                #     slice_links[slice_id].add_edges_from(slice_links[slice_id-1].edges())
                #     links_groups[slice_id].extend(links_groups[slice_id-1])
        slice_links[slice_id].add_edge(a,b)
        links_groups[slice_id].append([a,b,v])
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]=pd.DataFrame(links_groups[slice_id],columns=['src','dst','value'])

    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))

        for node in slice.nodes():
            if not node in used_nodes:
                used_nodes.append(node)
                
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}
    for id, slice in slice_links.items():
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map) 
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]['src']=links_groups[slice_id]['src'].map(nodes_consistent_map)
        links_groups[slice_id]['dst']=links_groups[slice_id]['dst'].map(nodes_consistent_map)

    snapshots=[] 
    for id, slice in slice_links.items():
        attributes = [] 
        for node in slice:
            if data_name == 'vote':
                sum_votes=links_groups[id].query('dst=={}'.format(node)).value.sum()/divide_num
                attrs=[sum_votes]
            elif data_name == 'bitcoin':
                avg_rates=links_groups[id].query('dst=={}'.format(node)).value.mean() if slice.in_degree(node)!=0 else 0
                attrs=[avg_rates,slice.in_degree(node),slice.out_degree(node)]
            else:
                attrs=[slice.in_degree(node),slice.out_degree(node)]
            attributes.append(attrs)
        slice.graph["feat"]=attributes 
        snapshots.append(slice)
    
    
    with open(save_path, "wb") as f:
        pkl.dump(snapshots, f)
    print("Processed data has been saved at {}".format(save_path))

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
        print(f'\n\nPreprocessing {file_name}...')
        preprocess_csv(file_name)
    
    # pubmed
    print('\n\nPreprocessing pubmed...')
    preprocess_pubmed()
    
    # mat file
    for file_name in MAT_FILE_PATH.keys():
        print(f'\n\nPreprocessing {file_name}...')
        preprocess_mat(file_name)
    
    # temporal data
    for data_name in TEMPORAL_CONFIG.keys():
        print(f'\n\nPreprocessing {data_name}...')
        preprocess_temporal_data(data_name)
    print('Done.')