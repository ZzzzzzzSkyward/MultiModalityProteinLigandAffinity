# 1.import libraries
# common headers
from header import *
# neural network modules
from models import *
# command line interface
from cli import *
# log service
import logg
# parameters
from parameter import *
# dataset loader
from dataset import *
# 2.choose environment
# 3.define super parameters
# currently there is only one model running
params = param()
params.load_from_cli()
params.seed()
params.verify()
# 4.define output
logg.toscreen()
# 5.load model
# currently there is only the concatenation model
model = ConcatenationModel(params)
# 6.train model
# set environment
model.train()
# load data
data_train = dataset("TrainDataset", "train", DATA_DIR)
data_validate = dataset("ValidateDataset", "validate", DATA_DIR)
Train(model, data_train, data_validate)
del data_train, data_validate
# 7.verify model
# set environment
model.eval()
Eval(model)
