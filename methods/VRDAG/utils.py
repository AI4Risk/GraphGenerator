import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import warnings
import logging
import os
warnings.filterwarnings('ignore')

def filtered_adj(adj):
    
    adj=sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))
    edges=list(zip(indices[0],indices[1]))
    G=nx.DiGraph(edges)
    return np.array(nx.adjacency_matrix(G).todense())

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)
    
def save_checkpoint(model, optimizer, epoch, save_path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch+1,
    }
    torch.save(checkpoint, save_path)
    
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def save_gen_data(graph_dir, samples):
    # samples是[(A_1,X_1)...(A_T,X_T)]
    save_path = os.path.join(graph_dir, "VRDAG_G_seq.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(samples, f)
    print("Generated data has been saved at {}".format(save_path))

def Padding_G_t(config,graph_seq,t):
    seq_len=config['seq_len']
    if seq_len==None:
        seq_len=len(graph_seq)
    
    max_num_nodes=graph_seq[seq_len-1].number_of_nodes()
    
    Adj=torch.from_numpy(nx.adjacency_matrix(graph_seq[t]).todense()).type(torch.float32)
    pad_adj=nn.ZeroPad2d((0,max_num_nodes-Adj.shape[0],0,max_num_nodes-Adj.shape[1])) 
    Adj=pad_adj(Adj)
    
    # 2. 属性部分
    X=torch.from_numpy(np.array(graph_seq[t].graph['feat'])[:,config['attr_col'][config['data']]]).type(torch.float32)
    pad_attr=nn.ZeroPad2d((0,0,0,max_num_nodes-X.shape[0])) 
    X=pad_attr(X)
    
    return Adj,X

def Padding(graph_seq,
            config):
    
    A_list=[]
    X_list=[]
    
    seq_len=config['seq_len']
    if seq_len==None:
        seq_len=len(graph_seq)
    
    max_num_nodes=graph_seq[seq_len-1].number_of_nodes()
    
    for t in range(seq_len):
        
        Adj=torch.from_numpy(nx.adjacency_matrix(graph_seq[t]).todense()).type(torch.float32)
        pad_adj=nn.ZeroPad2d((0,max_num_nodes-Adj.shape[0],0,max_num_nodes-Adj.shape[1])) 
        Adj=pad_adj(Adj)
        A_list.append(Adj)
        
        X=torch.from_numpy(np.array(graph_seq[t].graph['feat'])[:,config['attr_col'][config['data']]]).type(torch.float32)
        pad_attr=nn.ZeroPad2d((0,0,0,max_num_nodes-X.shape[0]))
        X=pad_attr(X)
        X_list.append(X)
        
    return A_list,X_list
