from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import json
from os.path import join, abspath, dirname
import os
import scipy.sparse as sp
import logging
from datetime import datetime


########## main ##########
def case_insensitive(data):
    dir_names = os.listdir(join(dirname(__file__), 'data'))
    data = next(name for name in dir_names if data.lower() == name.lower()) # case insensitive
    return data
    
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    if method == 'GraphRNN' or method == 'GraphRNN-S':
        yaml_file = "config/GraphRNN.yaml"
    elif method == 'CPGAE':
        yaml_file = "config/CPGAE.yaml"
    elif method == 'BTGAE':
        yaml_file = "config/BTGAE.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file) as file:
        config = yaml.safe_load(file)
    
    config['data'] = case_insensitive(config['data'])
    data = config['data']
    
    args = {
        'method': method,
        'data_path': abspath(join('data', data, f'{data}_undirected_csr.npz')),
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
    elif args['method'] == 'CPGAE':
        from methods.CPGAE.main_CPGAE import main_CPGAE
        main_CPGAE(graph, args)
    elif args['method'] == 'BTGAE':
        from methods.BTGAE.main_BTGAE import main_BTGAE
        main_BTGAE(graph, args)

########## log settings ##########
def log_folder():
    log_folder_name = 'log'
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)
        print(f"folder '{log_folder_name}' is created.")

def logging_conf(method):
    log_file = os.path.join('log/', method + '-' + datetime.now().strftime("%m-%d_%H:%M") + '.log')
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file,
                    filemode='w')

def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)


if __name__ == "__main__":
    main(parse_args())
