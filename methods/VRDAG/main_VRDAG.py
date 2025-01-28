import torch
import torch.nn as nn
import time
from datetime import datetime
import gc
from tqdm import tqdm
import os
from os import path
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

from .var_dist import *
from .generator import VRDAG
from .utils import *

import sys
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'experiment'))
from graph_metrics import CompEvaluator
    

def main_VRDAG(graph_seq, args):
    set_seed(args['seed'])
    args['seq_len'] = len(graph_seq)
    log('Sequence Length is {}'.format(args['seq_len']))
    
    A_list, X_list = Padding(graph_seq, args)
    args['max_num_nodes']=A_list[-1].shape[0]
    log('{} data loaded!'.format(args['data']))
    
    log_dir = os.path.join('log/', args['log_name'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # -----------------------------Training----------------------------
    model = VRDAG(args)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epoch'], eta_min=args['eta_min'])

    judge_s = CompEvaluator(mmd_beta=args['mmd_beta'])
    A_src=[filtered_adj(A_list[j].numpy()) for j in range(args['seq_len'])]
    X_src=[X_list[k].numpy() for k in range(args['seq_len'])] 
    
    if args['resume']:
        log('Resuming from checkpoint...')
        model, optimizer, epoch_begin = load_checkpoint(model, optimizer, args['checkpoint_path'])
        log('Resumed from epoch {}'.format(epoch_begin))
    else:
        epoch_begin = 0
    
    best_mean_res = {}
    best_med_res = {}
        
    for n_epoch in tqdm(range(epoch_begin, args['num_epoch'])):
        if args['ini_method']=='zero':
            h=torch.zeros(args['max_num_nodes'],args['h_dim'],device=args['device'])
        elif args['ini_method']=='embed':
            h = model.id_embedding.weight
        else:
            raise ValueError('Wrong initialization method!')

        avg_kld_loss=0
        avg_struc_loss=0
        avg_attr_loss=0
        
        time_start = time.time() 
        
        for t in range(args['seq_len']):
            
            optimizer.zero_grad(set_to_none=True)

            loss_step = 0
            
            n_nodes=graph_seq[t].number_of_nodes() 
            
            A, X = A_list[t].to(args['device']), X_list[t].to(args['device'])
            
            if args['is_vectorize']:
                t_vec=model.time_to_vec(torch.FloatTensor([t]).to(args['device']))
            else:
                t_vec=None
                
            h, kld_loss, struc_loss, attr_loss = model(A, X, h.data,t_vec,n_nodes)
            
            avg_kld_loss+=kld_loss.data.item()
            avg_struc_loss+=struc_loss.data.item()
            avg_attr_loss+=attr_loss.data.item()
            
            loss_step+=kld_loss+struc_loss+attr_loss
        
            loss_step.backward()
        
            nn.utils.clip_grad_norm_(model.parameters(), args['clip_norm'])
            
            optimizer.step()
            
        
        time_end = time.time()    

        log('One Epoch Training Running Time: {} Sec \n'.format(time_end-time_start))
            
        log('Epoch-{}: Average Attribute Loss is {}'.format(n_epoch+1,avg_attr_loss/args['seq_len']))
        log('Epoch-{}: Average Structure Loss is {}'.format(n_epoch+1,avg_struc_loss/args['seq_len']))
        log('Epoch-{}: Average Latent distribution Loss is {}\n'.format(n_epoch+1,avg_kld_loss/args['seq_len']))
        writer.add_scalar('train/attr_loss', avg_attr_loss/args['seq_len'], n_epoch)
        writer.add_scalar('train/struc_loss', avg_struc_loss/args['seq_len'], n_epoch)
        writer.add_scalar('train/kld_loss', avg_kld_loss/args['seq_len'], n_epoch)
        writer.flush()
        
        time_start = time.time()
        gen_data=model._sampling(args['seq_len'])
        time_end = time.time()    

        log('One Round Testing Running Time: {} Sec \n'.format(time_end-time_start))

        opt_lr = scheduler.get_last_lr()
        log('Current Learning Rate is {}'.format(opt_lr))
        scheduler.step()
        
        if (n_epoch + 1) % args['save_model_interval'] == 0:
            save_checkpoint(model, optimizer, n_epoch, args['checkpoint_path'])
            log('Model Saved!')
            
        if (n_epoch+1) % args['sample_interval'] == 0:
            with torch.no_grad():
                '''Evaluation'''
                gen_data=model._sampling(args['seq_len'])
                
                if 'cuda' in args['device']:
                    A_gen=[filtered_adj(gen_data[i][0].cpu().numpy()) for i in range(args['seq_len'])]
                    X_gen=[gen_data[i][1].cpu().numpy() for i in range(args['seq_len'])]
                else:
                    A_gen=[gen_data[i][0].numpy() for i in range(args['seq_len'])]
                    X_gen=[gen_data[i][1].numpy() for i in range(args['seq_len'])]
                
                samples = [(A_gen[i], X_gen[i]) for i in range(args['seq_len'])]
                save_gen_data(args['graph_save_path'], samples)
                
                mean_res = judge_s.comp_graph_stats(A_src,A_gen)
                med_res = judge_s.comp_graph_stats(A_src,A_gen,eval_method='med')
                

                # attr_entrophy_js,attr_corr=evaluate_attr(src_attrs=X_src,
                #                                     gen_attrs=X_gen,
                #                                     diff='js',
                #                                     bins=20,
                #                                     low_bound=0,
                #                                     upper_bound=1,
                #                                     n_step=args['seq_len'])
                
                # attr_entrophy_emd,attr_corr=evaluate_attr(src_attrs=X_src,
                #                                 gen_attrs=X_gen,
                #                                 diff='emd',
                #                                 bins=20,
                #                                 low_bound=0,
                #                                 upper_bound=1,
                #                                 n_step=args['seq_len'])
                
                
                log('-------------Graph structure statistics are as follows-------------')
                for key in mean_res.keys():
                    log("f_avg({}): {:.4f}".format(key, mean_res[key]))
                    log("f_med({}): {:.4f}".format(key, med_res[key]))
                    if key not in best_mean_res or mean_res[key] < best_mean_res[key]:
                        best_mean_res[key] = mean_res[key]
                    if key not in best_med_res or med_res[key] < best_med_res[key]:
                        best_med_res[key] = med_res[key]
                # log('-------------------------------------------------------------------\n')

                # log('-------------Node attributes evaluation are as follows-------------')
                
                
                # log('Attribute Entrophy- : JSD: {}  EMD: {}'.format(attr_entrophy_js,attr_entrophy_emd))
                # if attr_corr is not None:
                #     log('Attribute Correlation Error: {}'.format(attr_corr))
                
                log('-------------------------------------------------------------------\n')
                
                del gen_data
                del A_gen
                del X_gen
                
                gc.collect()
    
    log('Best Mean Results:')
    for key in best_mean_res.keys():
        log("f_avg({}): {:.4f}".format(key, best_mean_res[key]))
    log('Best Median Results:')
    for key in best_med_res.keys():
        log("f_med({}): {:.4f}".format(key, best_med_res[key]))
    
    save_checkpoint(model, optimizer, args['num_epoch'] - 1, args['checkpoint_path'])
    log('Model Saved!')
