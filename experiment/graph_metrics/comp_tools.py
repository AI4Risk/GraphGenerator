# Temporal Graph Evaluator

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from collections import defaultdict
import networkx as nx
from .metrics import *
from .utils import *
    
class CompEvaluator:
    
    def __init__(self, mmd_beta=2.0):
        '''
        eval_method: evaluation method, 'mean' or 'med'
        '''
        self.eval_time_len = None
        self.is_ratio = True
        
        self.mmd_beta=mmd_beta 
        self.stats_map={
                        'deg_dist': deg_dist,
                        'clus_dist': clus_dist,
                        'wedge_count': wedge_count,
                        'triangle_count': triangle_count,
                        'claw_count': claw_count,
                        'n_components': n_component,
                        'lcc_size': LCC,
                        'power_law': power_law_exp,
                        'gini': gini,
                        # 'node_div_dist': node_div_dist,
                        'global_cluster_coef': clustering_coefficient,
                        # 'mean_bc': mean_betweeness_centrality,
                        # 'mean_cc': mean_closeness_centrality
                       }
        
        self.hists_range={'deg_dist_in':(0,100),
                          'deg_dist_out':(0,100),
                          'deg_dist':(0,100),
                          'clus_dist':(0.0,1.0),
                          'node_div_dist':(-1.0,1.0)
                         }
    
    def error_func(self, org_graph, generated_graph, method='mean'):
        
        if self.is_ratio==True:
            metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
        else:
            metric = np.abs(org_graph - generated_graph)

        if method=='med':
            return np.median(metric)
        elif method=='mean':
            return np.mean(metric)
        else:
            raise ValueError('{} method does not exist!'.format(method))
    
    def error_func_dist_mmd(self, org_dist, gen_dist, method='mean'):
        
        metric=np.array([calculate_mmd(org_dist[t].reshape(-1,1),gen_dist[t].reshape(-1,1),beta=self.mmd_beta) for t in range(self.eval_time_len)])
        
        if method=='med':
            return np.median(metric)
        elif method=='mean':
            return np.mean(metric)
        else:
            raise ValueError('{} method does not exist!'.format(method))    
    
    def cal_stat(self, A_seq, stat_func, flow=None):
        if flow is None:
            return np.array([stat_func(A_seq[t]) for t in range(self.eval_time_len)])
        else:
            return np.array([stat_func(A_seq[t], flow) for t in range(self.eval_time_len)])
    
    def cal_stat_dist(self, A_seq, stat_func, hist_range, flow=None):
        
        hists = []
        
        for t in range(self.eval_time_len):
            
            if flow is None:
                dist = stat_func(A_seq[t])
            else:
                dist = stat_func(A_seq[t], flow)
            
            hist, _ = np.histogram(np.array(dist), bins=100, range=hist_range, density=False)
            
            hists.append(hist)
        
        return np.array(hists)
    
    def cal_func(self,stat):
        flow = None
        if stat.endswith('in'):
            flow = 'in'
        elif stat.endswith('out'):
            flow = 'out'

        if stat in self.hists_range.keys():
            return self.error_func_dist_mmd(self.cal_stat_dist(self.A_src, self.stats_map[stat], self.hists_range[stat], flow),
                                                    self.cal_stat_dist(self.A_gen, self.stats_map[stat], self.hists_range[stat], flow),
                                                    self.eval_method)
        else:
            return self.error_func(self.cal_stat(self.A_src, self.stats_map[stat], flow),
                                            self.cal_stat(self.A_gen, self.stats_map[stat], flow),
                                            self.eval_method)
    # output the evaluation results for all statistics
    def comp_graph_stats(self,A_src, A_gen, eval_method='mean', is_ratio=True, stats_list=None, eval_time_len=None):
        # transform the input graph format
        self.A_src = trans_format(A_src)
        self.A_gen = trans_format(A_gen)
        
        if not isinstance(self.A_src, list):
            self.A_src=[self.A_src]
        if not isinstance(self.A_gen, list):
            self.A_gen=[self.A_gen]
        
        # make the graph undirected
        for A in self.A_src:
            A = A + A.T
            A[A > 1] = 1
        for A in self.A_gen:
            A = A + A.T
            A[A > 1] = 1
            
        self.eval_method=eval_method
        
        if eval_time_len is None:
            self.eval_time_len= min([len(self.A_src), len(self.A_gen)])
        else:
            self.eval_time_len=eval_time_len
        self.is_ratio=is_ratio
            
        res_dict=defaultdict(float) # result dictionary
        
        metric_list=stats_list if stats_list is not None else list(self.stats_map.keys())

        time_start = time.time()
        
        with ProcessPoolExecutor() as executor:
            results = executor.map(self.cal_func, metric_list)
            res_dict = {stat: result for stat, result in zip(metric_list, results)}
        time_end = time.time()
        print("Time taken:", time_end - time_start, "seconds")
            
        return res_dict
