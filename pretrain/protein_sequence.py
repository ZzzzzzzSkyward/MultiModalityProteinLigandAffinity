# 1.import libraries
# common headers
from header import *
from constants import *
# neural network modules
from useless.models import *
# command line interface
from cli import *
# log service
from logg import *
# parameters
from parameter import *
# dataset loader
from dataset import *
# torch util
import pytorchutil as T
# trainer
from pretrain_single import *
from test_single import *
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
params.compound_size = 512
params.protein_size = 2048
params.hidden_size = 128
model = ProteinEncoder(params)
subdir = "test_1"
seq, smiles, logk, zernike, graph, edge, alpha, lv1 = 'seq', 'smiles', 'logk', 'zernike', 'matrix', '2dfeature', 'alpha', 'lv1'
choice = [seq,zernike]
# 6.train model
if args.train:
    trainer = Train
    if args.train_add:
        trainer = Train_Add
    # load data
    data_train = dataset("data_train", "train", DATA_DIR, subdir)()
    data_train.choose(choice)
    loader_train = dataloader(data_train, params.bs)
    data_test = dataset("data_test", "test_both_present", DATA_DIR, subdir)()
    data_test.choose(choice)
    loader_test = dataloader(data_test, params.bs, False)
    trainer(model, loader_train, loader_test, args)
# 7.verify model
# set environment
if args.test:
    # load data
    data_train = dataset("data_train", "train", DATA_DIR, subdir)()
    data_train.choose(choice)
    loader_train = dataloader(data_train, params.bs, False)
    data_test = dataset("data_test", "test_both_present", DATA_DIR, subdir)()
    data_test_ligand_only = dataset("data_test_ligand_only",
                                    "test_ligand_only", DATA_DIR, subdir)()
    data_test_protein_only = dataset("data_test_protein_only",
                                     "test_protein_only", DATA_DIR, subdir)()
    data_test_both_none = dataset("data_test_both_none", "test_both_none",
                                  DATA_DIR, subdir)()
    data_test.choose(choice)
    data_test_ligand_only.choose(choice)
    data_test_protein_only.choose(choice)
    data_test_both_none.choose(choice)
    loader_test = dataloader(data_test, params.bs, False)
    loader_test_ligand_only = dataloader(data_test_ligand_only, params.bs,
                                         False)
    loader_test_protein_only = dataloader(data_test_protein_only, params.bs,
                                          False)
    loader_test_both_none = dataloader(data_test_both_none, params.bs, False)
    Eval(model, [loader_train,
                 loader_test, loader_test_ligand_only, loader_test_protein_only,
                 loader_test_both_none
                 ], args)
