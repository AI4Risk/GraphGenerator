import logging
from datetime import datetime
import os

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