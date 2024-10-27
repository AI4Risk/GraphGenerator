import os
import shutil
import random
from time import gmtime, strftime
from random import shuffle

from .data import Graph_sequence_sampler_pytorch, split_graph

from .train import *
import yaml


def main_GraphRNN(graph, args):
    # All necessary arguments are defined in args.py
    graphs = split_graph(graph, args)
    
    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_validate = graphs[0:int(0.2*graphs_len)]


    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    log('graph_validate_len:{}'.format(graph_validate_len) )

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    log('graph_test_len: {}'.format(graph_test_len))

    args['max_num_node'] = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args['max_num_node'] = 2000
    # show graphs statistics
    log('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    log('max number node: {}'.format(args['max_num_node']))
    log('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    log('max previous node: {}'.format(args['max_prev_node']))

    #  save ground truth graphs
    for graph in graphs_test:
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
    fname = os.path.join(args['graph_save_path'], '{}_test.bin'.format(args['method']))
    save_graph_list(graphs_test, fname)

    
    dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args['max_prev_node'],max_num_node=args['max_num_node'])
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args['batch_size']*args['batch_ratio'], replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'],
                                               sampler=sample_strategy)

    ### model initialization
    
    if 'GraphRNN-S' in args['method']:
        rnn = GRU_plain(input_size=args['max_prev_node'], embedding_size=args['embedding_size_rnn'],
                        hidden_size=args['hidden_size_rnn'], num_layers=args['num_layers'], has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args['hidden_size_rnn'], embedding_size=args['embedding_size_output'], y_size=args['max_prev_node']).cuda()
    elif 'GraphRNN' in args['method']:
        rnn = GRU_plain(input_size=args['max_prev_node'], embedding_size=args['embedding_size_rnn'],
                        hidden_size=args['hidden_size_rnn'], num_layers=args['num_layers'], has_input=True,
                        has_output=True, output_size=args['hidden_size_rnn_output']).cuda()
        output = GRU_plain(input_size=1, embedding_size=args['embedding_size_rnn_output'],
                           hidden_size=args['hidden_size_rnn_output'], num_layers=args['num_layers'], has_input=True,
                           has_output=True, output_size=1).cuda()

    ### start training
    train(args, dataset_loader, rnn, output)

