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
params.compound_size = 1024
params.protein_size = 1024
params.hidden_size = 128
model = CompoundPretrainTopologyEncoder(params)
subdir = "pretrain_ligand_full"
smiles = "selfies"
# 6.train model
if args.train:
    trainer = Train
    # load data
    data_train = dataset("data_train", "train", DATA_DIR, subdir)()
    data_train.choose([args.ligand,'fingerprint'])
    loader_train = dataloader(data_train, params.bs)
    data_test = dataset("data_test", "test_protein_only", DATA_DIR, subdir)()
    data_test.choose([smiles,'fingerprint'])
    loader_test = dataloader(data_test, params.bs, False)
    trainer(model, loader_train, loader_test, args)
if args.test:
    # load data
    data_test = dataset("data_test", "test_protein_only", DATA_DIR, subdir)()
    data_test.choose([smiles])
    loader_test = dataloader(data_test, params.bs, False)
    tester = Eval_display
    while input() == '':
        tester(model, loader_test, args)
