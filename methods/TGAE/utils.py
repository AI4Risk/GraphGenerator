import copy
import random
import torch
import dgl
import scipy.sparse as sp
import numpy as np
import logging

def coo_to_csp(sp_coo):
    num = sp_coo.shape[0]
    feat_num = sp_coo.shape[1]
    row = sp_coo.row
    col = sp_coo.col
    sp_tensor = torch.sparse.FloatTensor(torch.LongTensor(np.stack([row, col])),
                                         torch.tensor(sp_coo.data),
                                         torch.Size([num, feat_num]))
    return sp_tensor

def from_nx_to_sparse_adj(temporal_graph):
    node_num = temporal_graph[-1].number_of_nodes()
    time_unique = np.array(range(len(temporal_graph)))
    
    temporal_src = np.array([], dtype=int)
    temporal_dst = np.array([], dtype=int)
    target_src = np.array([], dtype=int)
    target_dst = np.array([], dtype=int)

    for timestamp, graph in enumerate(temporal_graph):
        edges = np.array(graph.edges())
        temporal_src = np.append(temporal_src, edges[:, 0] + timestamp * node_num)
        temporal_dst = np.append(temporal_dst, edges[:, 1] + timestamp * node_num)
        
        target_src = np.append(target_src, edges[:, 0] + timestamp * node_num)
        target_dst = np.append(target_dst, edges[:, 1])
        

    inner_src = np.array(range(time_unique.min() * node_num, time_unique.max() * node_num))
    inner_dst = np.array(range((time_unique.min() + 1) * node_num, (time_unique.max() + 1) * node_num))
    self_src = np.array(range(time_unique.min() * node_num, (time_unique.max() + 1) * node_num))
    self_dst = np.array(range(time_unique.min() * node_num, (time_unique.max() + 1) * node_num))

    temporal_edges = (np.concatenate([temporal_src, inner_src, self_src]),
                        np.concatenate([temporal_dst, inner_dst, self_dst]))
    dglg = dgl.graph(temporal_edges)

    train_nids = np.unique(target_src)
    gt_adj = sp.coo_matrix((np.ones(len(target_src)), (target_src, target_dst)),
                            shape=(node_num * (time_unique.max() - time_unique.min() + 1), node_num)).tocsr()
    return dglg, gt_adj, train_nids


def random_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def edge_from_scores(scores_matrix, n_edges):
    degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.
    B = scores_matrix.shape[0]
    N = scores_matrix.shape[1]
    target_g = sp.csr_matrix(scores_matrix.shape)
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

def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)
    
def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch+1,
    }
    torch.save(checkpoint, save_path)
    
def load_checkpoint(model, optimizer, scheduler, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch