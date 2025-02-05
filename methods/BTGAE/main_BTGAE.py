from .partition import GraphPartitioner
from .genSubgraph import CGraphVAE, GraphSampler
from .genLink import LinkGenerator
from .aggregate import GraphReconstructor
from .utils import *

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import pandas as pd
import json
import time
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'experiment'))
from graph_metrics import compute_statistics, CompEvaluator

class Trainer:
    def __init__(self, args):
        # config
        self.args = args
        self.device = args['device']
        self.epochs = args['epochs']
        self.lambda_begin = 0.8
        self.lambda_end = 0.2
        self.max_num_nodes = None # max number of nodes in the subgraphs, automatically set
        self.N = None # number of nodes in the global graph
        
        # model
        self.partition = None      # GraphPartitioner, partition the graph, using metis, louvain or other methods
        self.submodel = None       # CGraphVAE, subgraph learner, to generate blocks
        self.BiGAES = None         # list of LinkGenerator, to generate links between blocks
        self.graph_sampler = None  # GraphSampler, to sample subgraphs, using the submodel
        self.reconstructor = None  # GraphReconstructor, to aggregate subgraphs, using the BiGAES
        
        # optimizer
        self.subOptimizer = None
        self.biOptimizers = None

        # data
        self.subgraphs = []
        self.subgraph_list = []     # list of torch.tensor, subgraphs, padded to max_num_nodes
        self.nodes_num = []         # list of int, number of nodes in each subgraph
        self.adj_permute = None     # scipy.sparse.csr_matrix, permuted adjacency matrix
        self.subgraph_list_pyg = [] # list of pyg graph, subgraphs
        self.eigenvalues = []       # list of torch.tensor, eigenvalues of the subgraphs (spectral)
        self.bridges = {}           # dict, {(i,j): bridge between subgraph i and j}
        
        self.loss_fn = nn.BCEWithLogitsLoss() # nn.BCELoss()
 
    def count_edges(self,graphs):
        ret = 0
        for g in graphs:
            ret += g.sum()
        return ret//2

    def check(self,adj):
        # adj is a scipy.sparse.csr_matrix
        # Iterate directly over the non-zero elements in the data array
        for i, value in enumerate(adj.data):
            if value > 1:
                # Get the row index by finding the appropriate interval in indptr
                row_index = np.searchsorted(adj.indptr - 1, i, side='right') - 1
                # Column index is directly given in the indices array
                col_index = adj.indices[i]
                log(f"Edge from {row_index} to {col_index} with weight {value}")
                # return False

        return True
    
    def set_edge_weights_to_one(self,csr_matrix):
        # Set all data values to 1
        csr_matrix.data = np.ones_like(csr_matrix.data, dtype=int)

    def load_model(self, num_sub_graphs):
        max_num_nodes = self.max_num_nodes
        
        self.submodel = CGraphVAE(input_dim=max_num_nodes,
                          hidden_dim=self.args['hidden_dim'],
                          latent_dim=self.args['latent_dim'],
                          max_num_nodes=max_num_nodes,
                          label_size=self.args['label_size'],
                          link_hidden_dim=self.args['link_hidden_dim'],
                          device = self.device).to(self.device)
        
        # a list of biGAE learners
        self.BiGAES = [LinkGenerator(max_num_nodes, self.args['link_hidden_dim'], self.args['device']).to(self.device) \
                       for _ in range(num_sub_graphs)]
        
        lr = self.args['learning_rate']
        wd = self.args['weight_decay']
        lr_ratio = self.args['min_lr_ratio']
        
        self.subOptimizer = optim.AdamW(self.submodel.parameters(), lr=lr, weight_decay=wd)

        self.biOptimizers = [optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) for model in self.BiGAES]
        
        self.scheduler = CosineAnnealingLR(self.subOptimizer, T_max=self.epochs, eta_min=lr/lr_ratio)
        self.scheduler_bi = [CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=lr/lr_ratio) for optimizer in self.biOptimizers]
        
    def prepare_data(self):
        max_num_nodes = self.max_num_nodes
        # convert csr to pyg graph
        self.subgraph_list_pyg = [csr_to_pyg_graph(subgraph, max_num_nodes) for subgraph in self.subgraphs]
        self.nodes_num = [subgraph.shape[0] for subgraph in self.subgraphs]
        
        for subgraph in self.subgraphs:
            node_num = subgraph.shape[0]
            subgraph_np = np.pad(subgraph.toarray(), (0, max_num_nodes - node_num), mode='constant')
            subgraph_tensor = self.to_tensor(subgraph_np, dtype=torch.float, device=self.device)
            self.subgraph_list.append(subgraph_tensor)
            
        # Compute the eigenvalues of the subgraphs
        for subgraph in self.subgraphs:
            eigenval = compute_eigen_csr(A_csr=subgraph, k=self.args['label_size'])
            eigenval = np.expand_dims(eigenval, axis=0)
            eigenval = torch.tensor(eigenval, dtype=torch.float).to(self.device)
            # normalize eigenval
            eigenval = eigenval / eigenval.max()
            self.eigenvalues.append(eigenval)
        
        
        num_subgraphs = len(self.nodes_num)
        row = 0
        for i in range(num_subgraphs - 1):
            ilen = self.nodes_num[i]
            col = row + ilen # start from the next column
            for j in range(i+1,num_subgraphs):
                jlen = self.nodes_num[j]
                sub_ij = self.adj_permute[row:row+ilen, col:col+jlen]
                
                target = pad_sparse_matrix_with_numpy(sub_ij, self.max_num_nodes)
                target = torch.tensor(target, dtype=torch.float)
                self.bridges[(i,j)] = target
                col += jlen # move to the next column
            row += ilen # move to the next row

    def to_tensor(self,array, dtype=torch.float, device='cpu'):
        return torch.from_numpy(array).type(dtype).to(device)
    
    def lambda_scheduler(self, epoch):
        sche_epochs = self.epochs
        lamda = self.lambda_begin + (epoch / sche_epochs) * (self.lambda_end  - self.lambda_begin)
        return lamda
    
    def save_checkpoint(self, epoch):
        ckpt = {
            'submodel': self.submodel.state_dict(),
            'BiGAES': [model.state_dict() for model in self.BiGAES],
            'subOptimizer': self.subOptimizer.state_dict(),
            'biOptimizers': [optimizer.state_dict() for optimizer in self.biOptimizers],
            'scheduler': self.scheduler.state_dict(),
            'scheduler_bi': [scheduler.state_dict() for scheduler in self.scheduler_bi],
            'epoch': epoch + 1,
        }
        torch.save(ckpt, self.args['checkpoint_path'])
    
    def load_checkpoint(self):
        ckpt = torch.load(self.args['checkpoint_path'])
        self.submodel.load_state_dict(ckpt['submodel'])
        for i, model in enumerate(self.BiGAES):
            model.load_state_dict(ckpt['BiGAES'][i])
        self.subOptimizer.load_state_dict(ckpt['subOptimizer'])
        for i, optimizer in enumerate(self.biOptimizers):
            optimizer.load_state_dict(ckpt['biOptimizers'][i])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        for i, scheduler in enumerate(self.scheduler_bi):
            scheduler.load_state_dict(ckpt['scheduler_bi'][i])
        return ckpt['epoch']
    
    def train(self, epochs):
        """
        Train the model using the provided subgraphs and corresponding adjacency matrix parts.

        Args:
        - epochs (int): Number of epochs to train the model.
        """
        for g in self.subgraph_list_pyg:
            g = g.to(self.device)

        # set the model to training mode
        self.submodel.train()     
        for model in self.BiGAES:
            model.train()

        num_subgraphs = len(self.nodes_num)
        
        start_time = time.time()
        
        if self.args['resume']:
            start_epoch = self.load_checkpoint()
            log(f'Resuming training from epoch {start_epoch}')
        else:
            start_epoch = 0
        
        best_eval_res = {}
        # TODO debug
        best_ri_res = {}
        best_el_res = {}
        
        test_time_all = 0
        for epoch in range(start_epoch, epochs):
            tot_loss = 0
            cur_time = time.time()
            loss_all_sub = 0
            bi = []
            
            self.subOptimizer.zero_grad()
            for optimizer in self.biOptimizers:
                optimizer.zero_grad()
                 
            lamda = self.lambda_scheduler(epoch)
            for i in range(num_subgraphs):                
                loss_i, gen_emb_i = self.submodel(self.subgraph_list[i], self.subgraph_list_pyg[i], self.eigenvalues[i])
                
                bi_i = self.BiGAES[i](self.subgraph_list_pyg[i], gen_emb_i, lamda)
                bi.append(bi_i)
                
                loss_all_sub += loss_i
            
            loss_all_sub.backward(retain_graph=True)
            
            for i in range(num_subgraphs - 1):
                for j in range(i+1,num_subgraphs):
                    output = torch.matmul(bi[i],bi[j].t())
                    target = self.bridges[(i,j)].to(self.device)
                    
                    loss_ij = self.loss_fn(output, target)
                    loss_ij.backward(retain_graph=True)
                    
                    tot_loss += loss_ij.item()
            
            self.subOptimizer.step()
            for optimizer in self.biOptimizers:
                optimizer.step()
            
            self.scheduler.step()
            for scheduler in self.scheduler_bi:
                scheduler.step()
            
            tot_loss = tot_loss * 2 / (num_subgraphs*(num_subgraphs-1)) + loss_all_sub.item() / num_subgraphs
            
            # Delete the variables to free up memory
            del bi, loss_all_sub, loss_ij, output, target
            torch.cuda.empty_cache()
            
            log(f'Epoch {epoch+1}, Loss: {tot_loss}, Time: {time.time() - cur_time:.1f}s')
            
            # save checkpoint
            if (epoch+1) % self.args['epochs_save_ckpt'] == 0:
                self.save_checkpoint(epoch)
                log(f'Model saved at epoch {epoch+1}')
            
            # evaluate the model
            if (epoch+1) % self.args['epochs_eval'] == 0 or (epoch+1) == epochs:
                eval_res, test_time = self.test(lamda, is_last = ((epoch+1) == epochs))
                test_time_all += test_time
                for k, v in eval_res.items():
                    if k not in best_eval_res:
                        best_eval_res[k] = v
                    elif v < best_eval_res[k]:
                        best_eval_res[k] = v
                
                              
        log(f'Training time: {time.time() - start_time - test_time_all:.1f}s')
                
        return best_eval_res

    def test(self,lamda, is_last = False):
        test_start_time = time.time()
        
        self.submodel.eval() 
        for i in range(len(self.BiGAES)):
            self.BiGAES[i].eval()
            
        self.graph_sampler = GraphSampler(model = self.submodel)
        newgraphs, gen_embs = self.graph_sampler.sample(self.N, self.subgraphs, self.eigenvalues)
        log("sampling subgraph fininshed")
        log("sampled subgraph number : {}".format(len(newgraphs)))
        log("train_subgraph_edge_number: {}, sampled_subgraph_edge_number: {}".format(self.count_edges(self.subgraphs), self.count_edges(newgraphs)))

        self.reconstructor = GraphReconstructor(method = self.args['recon_method'],BiGAES = self.BiGAES)
        
        redundancy_edge = int(max(len(self.subgraphs),self.count_edges([self.train_graph])- self.count_edges(newgraphs)))
        if self.args['recon_method'] == 'random':
            pred = self.reconstructor.aggregate_subgraphs_random(newgraphs,redundancy_edge)
        elif self.args['recon_method'] == 'cellsbm': # newgraphs , gen_global_graph super_adj
            pred = self.reconstructor.aggregate_subgraphs_cellsbm(self.super_adj,newgraphs)
        elif self.args['recon_method'] == 'synth':
            pred = self.reconstructor.aggregate_subgraphs_synth(self.adj_permute, newgraphs, self.subgraph_list_pyg, gen_embs, lamda, self.max_num_nodes)
        
        pred = csr_delete_diagonal(pred)
        target = csr_delete_diagonal(self.adj_permute)
        
        log("reconstruction finished")
        log("total edge number: {}".format(self.count_edges([pred])))
        log("target edge number: {}".format(self.count_edges([target])))
        log('pred shape: {}'.format(pred.shape))
        log('target shape: {}'.format(target.shape))
        
        # save the graphs
        sp.save_npz(path.join(self.args['graph_save_path'], 'BTGAE_pred.npz'), pred)
        sp.save_npz(path.join(self.args['graph_save_path'], 'BTGAE_target.npz'), target)
        
        
        log("===============================================================")
        log("start eval")
        
        # edge overlap
        eo = edge_overlap(pred, target)
        log("edge overlap: {}".format(eo))

        pred_pruned = prune(self.args['prune_method'], pred)
        evaluator = CompEvaluator()
        eval_res = evaluator.comp_graph_stats(target, pred_pruned)
        log("res:" + json.dumps(eval_res, indent=4))
        
        if is_last and self.args['need_statistics']:
            log("pred graph:")
            stat_pred = compute_statistics(pred_pruned)
            log(json.dumps(stat_pred, indent=4))
        

        self.submodel.train()
        for i in range(len(self.BiGAES)):
            self.BiGAES[i].train()
        
        test_time = time.time() - test_start_time
        return eval_res, test_time

        
    def run(self, train_graph):
        # Step 1: Process the graph
        log("===============================================================")
        log("start running")
        self.train_graph = train_graph
        self.train_graph.astype(np.float64)
        self.train_graph = self.train_graph + self.train_graph.T
        self.set_edge_weights_to_one(self.train_graph)
        self.check(self.train_graph)
        
        log("finish loading")
        # log graph info
        log("data: {}".format(self.args['data']))
        self.N = self.train_graph.shape[0]
        log("node : {}, edges : {}".format(self.N, self.count_edges([self.train_graph])))
        log("partition method: {}".format(self.args['partition']))
        log("subgraph learning epochs : {}".format(self.epochs))
        log("globalgraph learning method: {}".format(self.args['recon_method']))
        log("===============================================================")

        # Step 2: Partition the graph
        self.partition = GraphPartitioner(self.train_graph, partition_type=self.args['partition'])
        self.subgraphs, community_list = self.partition.partition()
    
        self.adj_permute = self.partition.permute_adjacency_matrix(self.train_graph, community_list)
        self.super_adj = self.partition.extract_global_adj_matrix(self.subgraphs,self.adj_permute)
        
        cutting_num = self.count_edges([self.train_graph]) - self.count_edges(self.subgraphs)
        log("finish partitioning")
        log("subgraph number: {}".format(len(self.subgraphs)))
        log("max element: {}, mean size: {}".format(max([g.shape[0] for g in self.subgraphs]), np.mean([g.shape[0] for g in self.subgraphs])))
        log("cutting edge number: {}".format(cutting_num))

        self.max_num_nodes = max([g.shape[0] for g in self.subgraphs])
        num_sub_graphs = len(self.subgraphs)
        
        # Step 3: Train the model
        log("===============================================================")
        log("loading model...")
        self.load_model(num_sub_graphs)
        log("preparing data...")
        self.prepare_data()
        
        partioned_result = {
            "subgraphs": self.subgraphs,
            "adj_permute": self.adj_permute,
            "super_adj": self.super_adj,
            "eigenvalues": self.eigenvalues,
            "cutting_num": cutting_num,
            "max_num_nodes": self.max_num_nodes,
            "num_sub_graphs": num_sub_graphs,
            "N" : self.N
        }
        
        with open(path.join(self.args['graph_save_path'], 'partioned_result.pkl'), 'wb') as f:
            pkl.dump(partioned_result, f)
        log("partioned_result saved")
            
        log("start training")
        log("===============================================================")
        
        best_eval_res = self.train(self.epochs) #! training
        
        
        log("finish training")
        log("===============================================================")        
        
        df_avg = pd.DataFrame([best_eval_res])
        log("best_res:\n" + df_avg.to_csv(sep='\t', index=False, float_format='%.4f'))
        
        logPeakGPUMem(self.args['device'])
        
        log("===============================================================")
        log("finish all")

def main_BTGAE(A, args):
    set_random_seed(args['seed'])
    trainer = Trainer(args)
    trainer.run(A)