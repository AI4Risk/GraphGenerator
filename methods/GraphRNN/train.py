from tqdm import tqdm
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .utils import *
from .model import *
from .data import *

########## GraphRNN-S ##########
def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args['epochs_log']==0 and batch_idx==0: # only output first batch's statistics
            log('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args['epochs'],loss.data[0], args['data'], args['num_layers'], args['hidden_size_rnn']))

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)

def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args['max_num_node'])
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args['max_prev_node'])).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args['max_prev_node'])).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args['max_prev_node'])).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


########## GraphRNN #########
def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args['num_layers']-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args['epochs_log']==0 and batch_idx==0: # only output first batch's statistics
            log('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args['epochs'],loss.data, args['data'], args['num_layers'], args['hidden_size_rnn']))

        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data*feature_dim
        
    return loss_sum/(batch_idx+1)

def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args['max_num_node'])
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args['max_prev_node'])).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args['max_prev_node'])).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        
        hidden_null = Variable(torch.zeros(args['num_layers'] - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args['max_prev_node'])).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args['max_prev_node'],i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


########### train function for LSTM
def train(args, dataset_train, rnn, output):
    
    # initialize optimizer
    optimizer_rnn = optim.AdamW(rnn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    optimizer_output = optim.AdamW(output.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args['milestones'], gamma=args['lr_rate'])
    scheduler_output = MultiStepLR(optimizer_output, milestones=args['milestones'], gamma=args['lr_rate'])

    # start main loop
    loss_min = 99999999
    resume = args['resume']
    if resume['need']:
        ckpt = torch.load(resume['ckpt_path'])
        rnn.load_state_dict(ckpt['rnn'])
        output.load_state_dict(ckpt['output'])
        optimizer_rnn.load_state_dict(ckpt['optimizer_rnn'])
        optimizer_output.load_state_dict(ckpt['optimizer_output'])
        scheduler_rnn.load_state_dict(ckpt['scheduler_rnn'])
        scheduler_output.load_state_dict(ckpt['scheduler_output'])
        epoch_begin = ckpt['epoch']
    else:
        epoch_begin = 0
        
    for epoch in tqdm(range(epoch_begin+1, args['epochs']+1)): 
        
        # train
        if 'GraphRNN-S' in args['method']:
            loss = train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN' in args['method']:
            loss = train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
         
        if epoch % args['epochs_ckpt']==0:
            ckpt = {
                'rnn': rnn.state_dict(),
                'output': output.state_dict(),
                'optimizer_rnn': optimizer_rnn.state_dict(),
                'optimizer_output': optimizer_output.state_dict(),
                'scheduler_rnn': scheduler_rnn.state_dict(),
                'scheduler_output': scheduler_output.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, args['checkpoint_path'])
            log('checkpoint saved, epoch: {}, loss: {}'.format(epoch, loss))
        
        # if loss is smaller, do test and save graphs
        if loss<loss_min and epoch > args['epochs_graph']:
            log('test start, epoch: {}, training loss: {}'.format(epoch, loss))
            loss_min = loss
            
            G_pred = []
            while len(G_pred)<args['test_total_size']:
                print('test progress: {}/{}'.format(len(G_pred), args['test_total_size']))
                if 'GraphRNN-S' in args['method']:
                    G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args['test_batch_size'],sample_time=args['sample_time'])
                elif 'GraphRNN' in args['method']:
                    G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args['test_batch_size'])
                G_pred.extend(G_pred_step)
            
            # save graphs
            fname = os.path.join(args['graph_save_path'], '{}_loss{}.bin'.format(args['method'], loss_min))
            save_graph_list(G_pred, fname)
            
            log('test done, graphs saved')
        