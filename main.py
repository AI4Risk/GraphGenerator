from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import json
from os.path import join, abspath, dirname
import os
import scipy.sparse as sp
import logging
from datetime import datetime
import pickle as pkl


########## main ##########
def case_insensitive(data):
    dir_names = os.listdir(join(dirname(__file__), 'data'))
    data = next(name for name in dir_names if data.lower() == name.lower()) # case insensitive
    return data
    
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    parser.add_argument('--update', nargs="*", help='# usage: --update <key1>=<value1> <key2>=<value2> ...')
    cmd_args = parser.parse_args()
    method = cmd_args.method

    if method == 'GraphRNN' or method == 'GraphRNN-S':
        yaml_file = "config/GraphRNN.yaml"
    elif method == 'CPGAE':
        yaml_file = "config/CPGAE.yaml"
    elif method == 'BTGAE':
        yaml_file = "config/BTGAE.yaml"
    elif method == 'VRDAG':
        yaml_file = "config/VRDAG.yaml"
    elif method == 'TGAE':
        yaml_file = "config/TGAE.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file) as file:
        config = yaml.safe_load(file)
    
    if cmd_args.update: # update config
        for item in cmd_args.update:
            key, value = item.split('=')
            val_type = type(config[key])
            if val_type == bool:
                value = True if value.lower() in ['y', 'yes', 'true'] else False
            config[key] = val_type(value)

    config['data'] = case_insensitive(config['data'])
    data = config['data']
    
    args = {
        'method': method,
        'graph_save_path': abspath(join('graphs', data)),
        'checkpoint_path': abspath(join('models', '{}_{}.ckpt'.format(data, method))),
    }
    
    if method in ['VRDAG', 'TGAE']:
        if data not in ['email', 'bitcoin', 'vote']:
            raise ValueError("Unsupported dataset for VRDAG.")
        args['data_path'] = join('data', data, f'{data}.pkl')
    else:
        if data in ['email', 'bitcoin', 'vote']:
            raise ValueError("Unsupported dataset for static graph method.")
        args['data_path'] = join('data', data, f'{data}_undirected_csr.npz')
    if not os.path.exists(args['graph_save_path']):
        os.makedirs(args['graph_save_path'])
    
    args.update(config)
    
    return args


def main(args):
    # set up logging
    log_folder()
    log_name = logging_conf(args['method'], args['data'])
    args['log_name'] = log_name
    formatted_args = json.dumps(args, indent=4)
    log(f"Settings: {formatted_args}")
    
    # load data 
    data_path = args['data_path']
    if data_path.endswith('.npz'):
        graph = sp.load_npz(data_path)
    elif data_path.endswith('.pkl'):
        with open(data_path, "rb") as f:
            graph_seq = pkl.load(f)
    else:
        raise ValueError("Unsupported data format.")
    
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
    elif args['method'] == 'VRDAG':
        from methods.VRDAG.main_VRDAG import main_VRDAG
        main_VRDAG(graph_seq, args)
    elif args['method'] == 'TGAE':
        from methods.TGAE.main_TGAE import main_TGAE
        main_TGAE(graph_seq, args)

########## log settings ##########
def log_folder():
    log_folder_name = 'log'
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)
        print(f"folder '{log_folder_name}' is created.")

def logging_conf(method, data):
    log_name = method + '-' + data + '-' + datetime.now().strftime("%m-%d %H:%M")
    log_file = os.path.join('log/', log_name + '.log')
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file,
                    filemode='w')
    return log_name

def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)


if __name__ == "__main__":
    main(parse_args())
