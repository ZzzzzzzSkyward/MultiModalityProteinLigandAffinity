from header import *
argument_parser = argparse.ArgumentParser()
argument_settings = {
    'epoch': 100,
    'resume': 0,
    'data_processed_dir': './data_processed/',
    'data_pretrain_dir': './data_pretrain/',
    'train': 0,
    'batch_size': 32,
    'l0': 0.01,
    'l1': 0.01,
    'l2': 0.0001,
    'l3': 10,
    'lr': 1e-4,  # learning rate
}

# argument format: --xxx=yyy --z=w


def initialize_parser(settings):
    for i in settings:
        argument_parser.add_argument(
            '--' + i,
            type=type(
                settings[i]),
            default=settings[i])


args = argument_parser.parse_args()
