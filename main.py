from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import json

from util import log_folder, logging_conf, log

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    if method == 'method_name':
        yaml_file = "config/method_name.yaml"
        
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file) as file:
        args = yaml.safe_load(file)
        if args is None:
            args = {}
            
    args['method'] = method
    return args


def main(args):
    log_folder()
    logging_conf(args['method'])
    formatted_args = json.dumps(args, indent=4)
    log(f"Settings: {formatted_args}")
    
    if args['method'] == 'method_name':
        pass
    

if __name__ == "__main__":
    main(parse_args())
