# 1.import libraries
# common headers
from unittest import loader
from header import *
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
params.load_from_cli(args)
params.seed()
params.verify()  # deterministic
# 4.define output
logg.toscreen()
# 5.load model
# currently there is only the concatenation model
model = ConcatenationModel(params)
# 6.train model
# load data
data_train = dataset("data_train", "train", DATA_DIR, "test_1")
data_test = dataset("data_test", "validate", DATA_DIR, "test_1")
data_train.select(["seq", "smiles", "logk"])
data_test.select(["seq", "smiles", "logk"])
loader_train = dataloader(data_train, args.batch_size)
loader_test = dataloader(data_test, args.batch_size)
Train(model, loader_train, loader_test, args.epoch, args.batch_size, args.lr)
del loader_train, loader_test
# 7.verify model
# set environment
# model.eval()
# Eval(model)
