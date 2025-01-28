
import torch
import numpy as np
import sys

import os
import os.path as path


import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
import scipy.sparse as sp

from .TGAE import ScalableTGAE
# from ...experiment.eval import compute_statistics
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'experiment'))
from .utils import *
from graph_metrics import compute_statistics, CompEvaluator

def compute_temporal_graph_statistics(A_T):
    seq_len = A_T.shape[0]//A_T.shape[1]
    num_nodes = A_T.shape[1]
    A_seq = [A_T[i*num_nodes:(i+1)*num_nodes, :] for i in range(seq_len)]
    stats = []
    for i in range(seq_len):
        stats.append(compute_statistics(A_seq[i]))
        
    return stats

def cal_avg_median_stats(adj, gen_mat):
    stats_real = compute_temporal_graph_statistics(adj)
    stats_gen_mat = compute_temporal_graph_statistics(gen_mat)
    
    f_avg = {}
    f_med = {}
    for key in stats_real[0].keys():
        if key == 'n_nodes' or key == 'n_edges':
            continue
        stats_real_values = np.array([stat[key] for stat in stats_real])
        stats_gen_mat_values = np.array([stat[key] for stat in stats_gen_mat])
        if 0 in stats_real_values:
            # vals = abs((stats_real_values - stats_gen_mat_values))
            pass
        else:
            vals = abs((stats_real_values - stats_gen_mat_values) / stats_real_values) 
            f_avg[key] = np.mean(vals)
            
            vals = vals[vals != 0]
            f_med[key] = np.median(vals)
    return f_avg, f_med

def main_TGAE(graph_seq, args):
    random_seed(args["seed"])
    evaluator = CompEvaluator()
    dgl_g, adj, nids = from_nx_to_sparse_adj(graph_seq)
    dgl_g = dgl.add_self_loop(dgl_g)
    
    num_nodes = adj.shape[1]
    t = adj.shape[0] // num_nodes
    feat = sp.diags(np.ones(num_nodes * t).astype(np.float32)).tocsr()

    sp.save_npz(os.path.join(args['graph_save_path'], "TGAE_adj.npz"), adj)
    
    train_sampler = MultiLayerFullNeighborSampler(num_layers=args["n_layers"])
    train_dataloader = DataLoader(dgl_g.to(args["device"]),
                                      indices=torch.from_numpy(nids).long().to(args["device"]),
                                      graph_sampler=train_sampler,
                                      device=args["device"],
                                      batch_size=args["batch_size"],
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)
    model = ScalableTGAE(in_dim=num_nodes * t,
                         hid_dim=int(args["H"]/args["n_heads"]),
                         n_heads=args["n_heads"],
                         out_dim=num_nodes).to(args["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"], eta_min=args["eta_min"])
    
    best_f_avg = {}
    best_f_med = {}
    if args["resume"]:
        model, optimizer, scheduler, epoch_begin = load_checkpoint(model, optimizer, scheduler, args["checkpoint_path"])
        log("Resumed from epoch {}".format(epoch_begin))
    else:
        epoch_begin = 0
        
    for epoch in range(epoch_begin, args["epochs"]):
        num_edges_all = 0
        num_loss_all = 0
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            batch_inputs, batch_labels = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args["device"]), \
                                         coo_to_csp(adj[seeds.cpu(), :].tocoo()).to_dense().to(args["device"])
            blocks = [block.to(args["device"]) for block in blocks]
            train_batch_logits = model(blocks, batch_inputs)
            num_edges = batch_labels.sum() / 2
            num_edges_all += num_edges
            loss = -0.5 * torch.sum(batch_labels * torch.log_softmax(train_batch_logits, dim=-1)) / num_edges
            num_loss_all += loss.cpu().data * num_edges
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 50 == 0:
                log("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}".format(epoch+1, step+1, loss.cpu().data))
                
        scheduler.step()
        log("Epoch: {:03d}, overall loss: {:.7f}".format(epoch + 1, num_loss_all/num_edges_all))
        if (epoch+1) % args["eval_per_epochs"] == 0:
            gen_mat = sp.csr_matrix(adj.shape)
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                    test_inputs = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args["device"])
                    blocks = [block for block in blocks]
                    test_batch_logits = torch.softmax(model(blocks, test_inputs), dim=-1)
                    num_edges = adj[seeds.cpu(), :].sum()
                    gen_mat[seeds.cpu(), :] = edge_from_scores(test_batch_logits.cpu().numpy(), num_edges)
                    if (step+1) % 20 == 0:
                        log("Epoch: {:03d}, Generating Step: {:03d}".format(epoch+1, step+1))
                    
            eo = adj.multiply(gen_mat).sum() / adj.sum()
            sp.save_npz(os.path.join(args['graph_save_path'], "TGAE_gen_mat.npz"), gen_mat)
            save_checkpoint(model, optimizer, scheduler, epoch, args["checkpoint_path"])
            
            log("Epoch: {:03d}, Edge Overlap: {:07f}".format(epoch + 1, eo))
            
            log('\n\n' + '='*80)
            # f_avg, f_med= cal_avg_median_stats(adj, gen_mat)
            seq_len = adj.shape[0]//adj.shape[1]
            num_nodes = adj.shape[1]
            A_seq = [adj[i*num_nodes:(i+1)*num_nodes, :] for i in range(seq_len)]
            A_gen_seq = [gen_mat[i*num_nodes:(i+1)*num_nodes, :] for i in range(seq_len)]
            f_avg = evaluator.comp_graph_stats(A_seq, A_gen_seq)
            f_med = evaluator.comp_graph_stats(A_seq, A_gen_seq, eval_method='med')
            for key in f_avg.keys():
                log("f_avg({}): {:.6f}".format(key, f_avg[key]))
                log("f_med({}): {:.6f}".format(key, f_med[key]))
                if key not in best_f_avg or f_avg[key] < best_f_avg[key]:
                    best_f_avg[key] = f_avg[key]
                if key not in best_f_med or f_med[key] < best_f_med[key]:
                    best_f_med[key] = f_med[key]
            log('='*80 + '\n\n')
    log("Finished training.")
    for key in best_f_avg.keys():
        log("Best f_avg({}): {:.6f}".format(key, best_f_avg[key]))
        log("Best f_med({}): {:.6f}".format(key, best_f_med[key]))
            