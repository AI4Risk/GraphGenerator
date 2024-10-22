from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import json
from os.path import join, abspath
import os
import scipy.sparse as sp

from util import log_folder, logging_conf, log

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    if method == 'GraphRNN' or method == 'GraphRNN-S':
        yaml_file = "config/GraphRNN.yaml"
        
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file) as file:
        config = yaml.safe_load(file)
    
    data = config['data']
    
    args = {
        'method': method,
        'data_path': abspath(join('data', data, f'{data}_graph_csr.npz')),
        'graph_save_path': abspath(join('graphs', data)),
        'checkpoint_path': abspath(join('models', '{}_{}.ckpt'.format(data, method))),
    }
    
    if not os.path.exists(args['graph_save_path']):
        os.makedirs(args['graph_save_path'])
    
    args.update(config)
    
    return args


def main(args):
    # set up logging
    log_folder()
    logging_conf(args['method'])
    formatted_args = json.dumps(args, indent=4)
    log(f"Settings: {formatted_args}")
    
    # load data 
    graph = sp.load_npz(args['data_path'])
    
    # run method
    if args['method'] in ['GraphRNN', 'GraphRNN-S']:    
        from methods.GraphRNN.main_GraphRNN import main_GraphRNN
        main_GraphRNN(graph, args)
        
    

if __name__ == "__main__":
    main(parse_args())
