# 1.import libraries
# common headers
from test_single import *
from pretrain_single import *
import pytorchutil as T
from dataset import *
from parameter import *
from logg import *
from cli import *
from constants import *
from header import *
# neural network modules
# command line interface
# log service
# parameters
# dataset loader
# torch util
# trainer
# 2.choose environment
# 3.define super parameters
# currently there is only one model running
params = param()
args = initialize_parser()
params.load_from_cli(args)
params.seed()
params.verify()  # deterministic
log(args)
# 4.define output
toscreen()
if args.tofile:
    tofile(args.file)
# 5.load model
# currently there is only the concatenation model
params.compound_size = 1024
params.protein_size = 2048
params.hidden_size = 128
subdir = "test_1"


def get_trainer(args):
    trainer = Train
    return trainer


seq, smiles, logk, zernike, graph, edge, alpha, lv1 = 'seq', 'smiles', 'logk', 'zernike', 'matrix', '2dfeature', 'alpha', 'lv1'
#choices = [seq, zernike]
choices = [seq, smiles, 'complexinteract', logk]
if smiles == 'smiles':
    params.compound_size = 512
else:
    params.compound_size = 512


def main(model):
    model = model(params)
    global args
    # 6.train model
    if args.train:
        trainer = get_trainer(args)
        # load data
        data_train = dataset("data_train", "train", DATA_DIR, subdir)()
        data_train.choose(choices)
        loader_train = dataloader(data_train, params.bs)
        data_test = dataset(
            "data_test",
            "test_both_present",
            DATA_DIR,
            subdir)()
        data_test.choose(choices)
        loader_test = dataloader(data_test, params.bs, False)
        trainer(model, loader_train, loader_test, args)
    # 7.verify model
    # set environment
    if args.test:
        # load data
        data_train = dataset("data_train", "train", DATA_DIR, subdir)()
        data_train.choose(choices)
        loader_train = dataloader(data_train, params.bs, False)
        data_test = dataset(
            "data_test",
            "test_both_present",
            DATA_DIR,
            subdir)()
        data_test_ligand_only = dataset("data_test_ligand_only",
                                        "test_ligand_only", DATA_DIR, subdir)()
        data_test_protein_only = dataset("data_test_protein_only",
                                         "test_protein_only", DATA_DIR, subdir)()
        data_test_both_none = dataset("data_test_both_none", "test_both_none",
                                      DATA_DIR, subdir)()
        data_test.choose(choices)
        data_test_ligand_only.choose(choices)
        data_test_protein_only.choose(choices)
        data_test_both_none.choose(choices)
        loader_test = dataloader(data_test, params.bs, False)
        loader_test_ligand_only = dataloader(data_test_ligand_only, params.bs,
                                             False)
        loader_test_protein_only = dataloader(data_test_protein_only, params.bs,
                                              False)
        loader_test_both_none = dataloader(
            data_test_both_none, params.bs, False)
        Eval(model, [  # loader_train,
            loader_test, loader_test_ligand_only, loader_test_protein_only,
            loader_test_both_none
        ], args)


def main_test(model):
    model = model(params)
    global args
    # load data
    data_train = dataset("data_train", "train", DATA_DIR, subdir)()
    data_train.choose(choices)
    if args.short:
        data_train.length = 500
    loader_train = dataloader(data_train, params.bs)
    data_test = dataset(
        "data_test",
        "test_both_present",
        DATA_DIR,
        subdir)()
    data_test_ligand_only = dataset("data_test_ligand_only",
                                    "test_ligand_only", DATA_DIR, subdir)()
    data_test_protein_only = dataset("data_test_protein_only",
                                     "test_protein_only", DATA_DIR, subdir)()
    data_test_both_none = dataset("data_test_both_none", "test_both_none",
                                  DATA_DIR, subdir)()
    data_test.choose(choices)
    data_test_ligand_only.choose(choices)
    data_test_protein_only.choose(choices)
    data_test_both_none.choose(choices)
    loader_test = dataloader(data_test, params.bs, False)
    loader_test_ligand_only = dataloader(data_test_ligand_only, params.bs,
                                         False)
    loader_test_protein_only = dataloader(data_test_protein_only, params.bs,
                                          False)
    loader_test_both_none = dataloader(
        data_test_both_none, params.bs, False)
    loader_tests = [
        loader_test,
        loader_test_ligand_only,
        loader_test_protein_only,
        loader_test_both_none]
    # 6.train model
    if args.train:
        trainer = get_trainer(args)
        this_loader_test = loader_tests if args.test else loader_test
        trainer(model, loader_train, this_loader_test, args)
    # 7.verify model
    if args.test and not args.train:
        Eval(model, loader_tests, args)
