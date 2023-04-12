from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def Eval(model, loaders, args):
    model.eval()
    for i in loaders:
        p, r, t, h = evaluate_affinity(model, i)
        log(f'pearson={p},rmse={r},tau={t},rho={h}')
