from argparse import Namespace
from header import *
import time
argument_parser = argparse.ArgumentParser()
argument_settings = {
    'epoch': 100,
    'resume': True,
    'data_processed_dir': './data_processed/',
    'data_pretrain_dir': './data_pretrain/',
    'protein': "",
    'ligand': "",
    'matrix': '',
    'train': True,
    'test': True,
    'main': True,
    'short': True,
    'tofile': True,
    'file': 'log.txt',
    'batch_size': 32,
    'l0': 0.01,
    'l1': 0.01,
    'l2': 0.0001,
    'l3': 10,
    'lr': 1e-4,  # learning rate
    'detailed': True,
    'track': True,  # always save loss
    'explore': True,  # don't load loss
    'dropout': 0.5,
    'train_add': True,
    'pretrained': True,
    'newoptim': True,
    'forcepretrain': True,
    'name': 'test' + time.strftime("%m-%d %H:%M:%S", time.localtime())
}
args = Namespace()

# argument format: -xxx=yyy --z


def initialize_parser(settings=argument_settings):
    global args
    for i in settings:
        if settings[i] == True:
            argument_parser.add_argument('-' + i,
                                         action='store_true',
                                         help='')
        else:
            argument_parser.add_argument('-' + i,
                                         type=type(settings[i]),
                                         default=settings[i])
    return argument_parser.parse_args()
