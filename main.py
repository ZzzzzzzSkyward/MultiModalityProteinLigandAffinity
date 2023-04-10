# 1.import libraries
# common headers
from header import *
from constants import *
# neural network modules
from models import *
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
from train_single import *
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
    tofile()
# 5.load model
# currently there is only the concatenation model
params.input_size = 1024
params.hidden_size = 128
model = OneDimensionalAffinityModel(params)
# 6.train model
if args.train:
    # load data
    data_train = dataset("data_train", "train", DATA_DIR, "test_1")()
    data_test = dataset("data_test", "test_both_none", DATA_DIR, "test_1")()
    data_train.choose(["seq", "smiles", "logk"])
    data_test.choose(["seq", "smiles", "logk"])
    loader_train = dataloader(data_train, params.bs)
    loader_test = dataloader(data_test, params.bs, False)
    Train(model, loader_train, loader_test, args)
# 7.verify model
# set environment
# model.eval()
# Eval(model)
