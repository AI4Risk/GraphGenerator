from .partition import GraphPartitioner
from .genSubgraph import CGraphVAE, GraphSampler
from .genLink import LinkGenerator
from .aggregate import GraphReconstructor
from .utils import *

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.sparse as sp
import pickle as pkl
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
        self.subgraph_list = []     # list of torch.tensor, subgraphs, padded to max_num_nodes
        self.nodes_num = []         # list of int, number of nodes in each subgraph
        self.adj_permute = None     # scipy.sparse.csr_matrix, permuted adjacency matrix
        self.subgraph_list_pyg = [] # list of pyg graph, subgraphs
        self.eigenvalues = []       # list of torch.tensor, eigenvalues of the subgraphs (spectral)
        
        self.loss_fn = nn.BCEWithLogitsLoss() # nn.BCELoss()
 
    def count_edges(self,graphs):
        ret = 0
        for g in graphs:
            # don't count self loops
            # ret += g.diagonal().sum()
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
        self.BiGAES = [LinkGenerator(max_num_nodes, self.args['link_hidden_dim'], max_num_nodes, self.args['device']).to(self.device) \
                       for _ in range(num_sub_graphs)]
        
        self.subOptimizer = optim.Adam(list(self.submodel.parameters()), lr=self.args['learning_rate'])

        self.biOptimizers = [optim.Adam(list(model.parameters()), lr=self.args['learning_rate']) for model in self.BiGAES]

    def prepare_data(self, subgraphs):
        max_num_nodes = self.max_num_nodes
        # convert csr to pyg graph
        self.subgraph_list_pyg = [csr_to_pyg_graph(subgraph, max_num_nodes) for subgraph in subgraphs]
        self.nodes_num = [subgraph.shape[0] for subgraph in subgraphs]
        
        for subgraph in subgraphs:
            node_num = subgraph.shape[0]
            subgraph_np = np.pad(subgraph.toarray(), (0, max_num_nodes - node_num), mode='constant')
            subgraph_tensor = self.to_tensor(subgraph_np, dtype=torch.float, device=self.device)
            self.subgraph_list.append(subgraph_tensor)
            
        # Compute the eigenvalues of the subgraphs
        for subgraph in subgraphs:
            eigenval = compute_eigen_csr(A_csr=subgraph, k=self.args['label_size'])
            eigenval = np.expand_dims(eigenval, axis=0)
            eigenval = torch.tensor(eigenval, dtype=torch.float).to(self.device)
            self.eigenvalues.append(eigenval)

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
        
        row = 0
        start_time = time.time()
        
        if self.args['resume']:
            start_epoch = self.load_checkpoint()
            log(f'Resuming training from epoch {start_epoch}')
        else:
            start_epoch = 0
        
        
        for epoch in range(start_epoch, epochs):
            tot_loss = 0
            cur_time = time.time()
            
            lamda = self.lambda_scheduler(epoch)
            
            loss_all_sub = 0
            bi = []
            
            self.subOptimizer.zero_grad()
            for optimizer in self.biOptimizers:
                optimizer.zero_grad()
            
            for i in range(num_subgraphs):                
                loss_i, gen_emb_i = self.submodel(self.subgraph_list[i], self.eigenvalues[i])
                emb_i = gen_emb_i.detach()
                emb_i.requires_grad = True
                bi_i = self.BiGAES[i](self.subgraph_list_pyg[i], emb_i, lamda)
                bi.append(bi_i)
                
                loss_all_sub += loss_i
            
            for i in range(num_subgraphs - 1):
                ilen = self.nodes_num[i]
                col = row + ilen # start from the next column
                for j in range(i+1,num_subgraphs):
                    jlen = self.nodes_num[j]
                    sub_ij = self.adj_permute[row:row+ilen, col:col+jlen]
                    # pad sub_ij to max_num_nodes*max_num_nodes with 0
                    target = pad_sparse_matrix_with_numpy(sub_ij, self.max_num_nodes)
                    target = torch.tensor(target, dtype=torch.float).to(self.args['device'])
                    
                    output = torch.matmul(bi[i],bi[j].t())
                    
                    loss_ij = self.loss_fn(output, target)
                    loss_ij.backward(retain_graph=True)
                    
                    tot_loss += loss_ij.item()
                    
                    col += jlen # move to the next column
                row += ilen # move to the next row
            
            loss_all_sub.backward()
            
            
            self.subOptimizer.step()
            for optimizer in self.biOptimizers:
                optimizer.step()
            
            tot_loss = tot_loss * 2 / (num_subgraphs*(num_subgraphs-1)) + loss_all_sub.item() / num_subgraphs
            
            # Delete the variables to free up memory
            del bi, loss_all_sub, loss_ij, output, target
            torch.cuda.empty_cache()
            
            log(f'Epoch {epoch+1}, Loss: {tot_loss}, Time: {time.time() - cur_time:.1f}s')
            
            # save checkpoint
            if (epoch+1) % self.args['epochs_save_ckpt'] == 0:
                self.save_checkpoint(epoch)
                log(f'Model saved at epoch {epoch+1}')
                
        log(f'Training finished in {time.time() - start_time:.1f}s')
        tot_time = time.time() - start_time
        return tot_time

    def run(self, train_graph):
        train_graph.astype(np.float64)
        # Step 1: Process the graph
        log("===============================================================")
        log("start running")
        self.set_edge_weights_to_one(train_graph)
        self.check(train_graph)
        # only remain the upper triangle
        train_graph = sp.triu(train_graph, format='csr')
        
        # fillup the lower triangle to form an undirected graph if it is not a symmetric
        train_graph = train_graph + train_graph.T - sp.diags(train_graph.diagonal())
        
        lcc = largest_connected_components(train_graph)
        train_graph = train_graph[lcc,:][:,lcc]

        log("finish loading")
        # log graph info
        log("data: {}".format(self.args['data']))
        self.N = train_graph.shape[0]
        log("node : {}, edges : {}".format(self.N, self.count_edges([train_graph])))
        log("partition method: {}".format(self.args['partition']))
        log("subgraph learning epochs : {}".format(self.epochs))
        log("globalgraph learning method: {}".format(self.args['recon_method']))
        log("===============================================================")

        # Step 2: Partition the graph
        self.partition = GraphPartitioner(train_graph, partition_type=self.args['partition'])
        subgraphs,community_list = self.partition.partition()
        self.adj_permute = self.partition.permute_adjacency_matrix(train_graph, community_list)
        super_adj = self.partition.extract_global_adj_matrix(subgraphs,self.adj_permute)

        cutting_num = self.count_edges([train_graph]) - self.count_edges(subgraphs)
        log("finish partitioning")
        log("subgraph number: {}".format(len(subgraphs)))
        log("max element: {}, mean size: {}".format(max([g.shape[0] for g in subgraphs]), np.mean([g.shape[0] for g in subgraphs])))
        log("cutting edge number: {}".format(cutting_num))

        self.max_num_nodes = max([g.shape[0] for g in subgraphs])
        num_sub_graphs = len(subgraphs)
        
        # Step 3: Train the model
        log("===============================================================")
        log("loading model...")
        self.load_model(num_sub_graphs)
        log("preparing data...")
        self.prepare_data(subgraphs)
        
        partioned_result = {
            "subgraphs": subgraphs,
            "adj_permute": self.adj_permute,
            "super_adj": super_adj,
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
        
        
        self.train(self.epochs) #! training
        
        
        log("finish training")
        log("===============================================================")
        
        # # Step 4: sample sub graph
        self.graph_sampler = GraphSampler(method = self.args['sample_method'],model = self.submodel)
        newgraphs = self.graph_sampler.sample(self.N, subgraphs, self.eigenvalues)
        log("sampling subgraph fininshed")
        log("sampled subgraph number : {}".format(len(newgraphs)))
        log("train_subgraph_edge_number: {}, sampled_subgraph_edge_number: {}".format(self.count_edges(subgraphs), self.count_edges(newgraphs)))

        # Step 5: reconstruction graph
        log("===============================================================")
        log("start reconstruction")
        self.reconstructor = GraphReconstructor(method = self.args['recon_method'],BiGAES = self.BiGAES)
        
        redundancy_edge = int(max(len(subgraphs),self.count_edges([train_graph])- self.count_edges(newgraphs)))
        if self.args['recon_method'] == 'random':
            pred = self.reconstructor.aggregate_subgraphs_random(newgraphs,redundancy_edge)
        elif self.args['recon_method'] == 'cellsbm': # newgraphs , gen_global_graph super_adj
            pred = self.reconstructor.aggregate_subgraphs_cellsbm(super_adj,newgraphs)
        elif self.args['recon_method'] == 'synth':
            pred = self.reconstructor.aggregate_subgraphs_synth(self.adj_permute, newgraphs, self.max_num_nodes)
        
        log("reconstruction finished")
        log("total edge number: {}".format(self.count_edges([pred])))
        
        pred = sp.triu(pred, format='csr')
        target = sp.triu(self.adj_permute, format='csr')
        pred = pred + pred.T - sp.diags(pred.diagonal())
        target = target + target.T - sp.diags(target.diagonal())
        
        # save the graphs
        sp.save_npz(path.join(self.args['graph_save_path'], 'BTGAE_pred.npz'), pred)
        sp.save_npz(path.join(self.args['graph_save_path'], 'BTGAE_target.npz'), target)
        
        log('pred shape: {}'.format(pred.shape))
        log('target shape: {}'.format(target.shape))
        log("===============================================================")
        log("start eval")
        
        log("pred graph:")
        stat_pred = compute_statistics(pred)
        log(json.dumps(stat_pred, indent=4))
        
        evaluator = CompEvaluator()
        res_mean = evaluator.comp_graph_stats(target, pred)
        log("res_mean:" + json.dumps(res_mean, indent=4))
        
        res_med = evaluator.comp_graph_stats(target, pred, eval_method='med')
        log("res_med:" + json.dumps(res_med, indent=4))
        
        logPeakGPUMem(self.args['device'])
        
        # If we prune the isolated nodes, the nodes of pred will differ from the target.
        # When pruning isolated nodes, do not calculate edge overlap.
        eo = edge_overlap(pred, target) 
        log(f'Edge overlap: {eo}')
        
        log("===============================================================")
        log("finish all")

def main_BTGAE(A, args):
    set_random_seed(args['seed'])
    trainer = Trainer(args)
    trainer.run(A)