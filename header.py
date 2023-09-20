import math
import numpy as np
import colorlog
import os
# must do this  before torch
BLOCKCUDA = 0#make sure consistency with getgpu.py


def blockgpu0():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


blockgpu0()
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data
import torch.backends
import torch.backends.cudnn
import random
import argparse
import scipy.stats
from sklearn.metrics import roc_curve, auc, average_precision_score
import multiprocessing
import torch.multiprocessing as mp
import torch.optim as optim
from tqdm import tqdm
import zzz  # my own library
import datetime
try:
    from torch_geometric.nn import *
except ImportError:
    pass
