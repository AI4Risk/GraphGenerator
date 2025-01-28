import time
import numpy as np
import torch
import json
import numpy as np
import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm
from os import path
from .models import *
from .utils import *

from community import best_partition
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
import sys

sys.path.append(path.join(path.dirname(__file__), '..', '..', 'experiment'))
from graph_metrics import compute_statistics, CompEvaluator

def main_CPGAE(A, args):
    stat = compute_statistics(A)
    log('\ndataset statistics: ' + json.dumps(stat, indent=4))
    random_seed(args['seed']) # set random seed
    
    ########## process raw graph ##########
    feat = sp.diags(np.ones(A.shape[0]).astype(np.float32)).tocsr()
    G = nx.Graph(A)
    comm_label = best_partition(G)
    comm_label = torch.LongTensor(list(comm_label.values()))
    num_partitions = len(set(comm_label.tolist()))
    
    dgl_g = dgl.graph(list(G.edges()))
    dgl_g = dgl.add_self_loop(dgl_g)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges() # undirected graph
    
    log('\n==================== Training ====================')
    
    
    ########## dataloader ##########
    train_sampler = MultiLayerFullNeighborSampler(num_layers=1)
    train_dataloader = DataLoader(dgl_g.to(args['device']),
                                    torch.arange(num_nodes).long().to(args['device']),
                                    graph_sampler=train_sampler,
                                    device=args['device'],
                                    batch_size=args['batch_size'],
                                    drop_last=False,
                                    shuffle=True,
                                    num_workers=0)
    
    ########## model ##########
    assert args['hidden_dim'] % args['num_heads'] == 0, 'hidden_dim must be divisible by num_heads'
    dim_head = int(args['hidden_dim'] // args['num_heads'])
    encoder = GATLayer(in_dim=num_nodes, hid_dim=dim_head, n_heads=args['num_heads']).to(args['device'])
    comm_decoder = nn.Linear(args['hidden_dim'], num_partitions).to(args['device'])
    decoder = nn.Linear(args['hidden_dim'], num_nodes).to(args['device'])
    
    ########## optimizer ##########
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    comm_dec_opt = torch.optim.Adam(comm_decoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    
    ########## resume ##########
    
    if args['resume']:
        log('Resuming from checkpoint...')
        ckpt = torch.load(args['checkpoint_path'])
        encoder.load_state_dict(ckpt['encoder'])
        comm_decoder.load_state_dict(ckpt['comm_decoder'])
        decoder.load_state_dict(ckpt['decoder'])
        enc_opt.load_state_dict(ckpt['enc_opt'])
        comm_dec_opt.load_state_dict(ckpt['comm_dec_opt'])
        dec_opt.load_state_dict(ckpt['dec_opt'])
        epoch_begin = ckpt['epoch']
        log('Resumed from epoch {}'.format(epoch_begin))
    else:
        epoch_begin = 0
        
    max_eo = 0
    max_ep = 0
    start_time = time.time()
    total_train_time = 0
    ########## training loop ##########
    
    for epoch in tqdm(range(epoch_begin+1, args['epochs']+1)):
        num_edges_all = 0
        num_loss_all = 0
        encoder.train()
        comm_decoder.train()
        decoder.train()
        train_time = 0
        
        for (input_nodes, seeds, blocks) in train_dataloader:
            batch_feat, batch_labels, batch_comm_labels = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args['device']), \
                                                            coo_to_csp(A[seeds.cpu(), :].tocoo()).to_dense().to(args['device']), \
                                                            comm_label[seeds.cpu()].to(args['device'])
            train_start_time = time.time()
            ########## train comm_decoder ##########
            train_batch_clus_res = comm_decoder(encoder(blocks[0], batch_feat))
            comm_loss = F.cross_entropy(train_batch_clus_res, batch_comm_labels)
            # backprop
            enc_opt.zero_grad()
            comm_dec_opt.zero_grad()
            comm_loss.backward()
            enc_opt.step()
            comm_dec_opt.step()
            
            ########## train decoder ##########
            blocks = [block.to(args['device']) for block in blocks]
            train_batch_logits = decoder(encoder(blocks[0], batch_feat))
            num_edges = batch_labels.sum() / 2
            num_edges_all += num_edges
            loss = -0.5 * torch.sum(batch_labels * torch.log_softmax(train_batch_logits, dim=-1)) / num_edges
            num_loss_all += loss.cpu().data * num_edges
            # backprop
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            dec_opt.step()
            # time consumption
            train_time += time.time() - train_start_time
            total_train_time += time.time() - train_start_time
            
        ########## generate ##########
        if (epoch) % 5 == 0:
            gen_mat = sp.lil_matrix(A.shape)
            encoder.eval()
            comm_decoder.eval()
            decoder.eval()
            gen_time = 0
            
            with torch.no_grad():
                for (input_nodes, seeds, blocks) in train_dataloader:
                    gen_start_time = time.time()
                    test_feat = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args['device'])
                    blocks = [block.to(args['device']) for block in blocks]
                    test_batch_logits = torch.softmax(decoder(encoder(blocks[0], test_feat)), dim=-1)
                    num_edges = A[seeds.cpu(), :].sum() # csr方便横向切片
                    gen_mat[seeds.cpu(), :] = edge_from_scores(test_batch_logits.cpu().numpy(), num_edges)
                    gen_time += time.time() - gen_start_time
            
            eo = edge_overlap(A, gen_mat.tocsr())
            log("Epoch: {}, Overall Loss: {:.5f}, Training Time: {}s".format(epoch, num_loss_all/num_edges_all, train_time))
            log("Edge Overlap: {:.5f}, Total Time: {}s, Generation Time: {}s, Total Time Consumption: {}s"
                .format(eo, int(time.time() - start_time), gen_time, total_train_time))
            if eo > max_eo:
                max_eo = eo
                max_ep = epoch
                log("### New Best Edge Overlap: {:07f} ### graph saved".format(eo))
                
                fname = path.join(args['graph_save_path'], '{}_edge_overlap{:.4f}.npz'.format(args['method'], eo))
                sp.save_npz(fname, gen_mat.tocsr())
                if eo > args['eo_limit']:
                    break
            elif epoch >= max_ep + args['ep_limit']:
                log("!!! Early Stopping after {} Epochs of EO Non-Ascending !!!".format(args['ep_limit']))
                break
        
        ########## save checkpoint ##########
        if (epoch) % 50 == 0:
            checkpoint = {
                'encoder': encoder.state_dict(),
                'comm_decoder': comm_decoder.state_dict(),
                'decoder': decoder.state_dict(),
                'enc_opt': enc_opt.state_dict(),
                'comm_dec_opt': comm_dec_opt.state_dict(),
                'dec_opt': dec_opt.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, args['checkpoint_path'])
            log('Checkpoint saved at epoch {}'.format(epoch))
        
    logPeakGPUMem(args['device'])
    
    gen_mat = sp.load_npz(fname)
    stat = compute_statistics(gen_mat)
    log('eval statistics: ' + json.dumps(stat, indent=4))
    
    evaluator = CompEvaluator()
    res_mean = evaluator.comp_graph_stats(A, gen_mat)
    log("res_mean:" + json.dumps(res_mean, indent=4))
    
    res_med = evaluator.comp_graph_stats(A, gen_mat, eval_method='med')
    log("res_med:" + json.dumps(res_med, indent=4))
    