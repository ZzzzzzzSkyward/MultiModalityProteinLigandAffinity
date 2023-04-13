from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def Eval(model, loaders, args):
    model.eval()
    # load data
    checkpoint_pth = "test_pth"
    if True:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    for i in loaders:
        p, r, t, h = evaluate_affinity(model, i)
        log(f'pearson={p},rmse={r},tau={t},rho={h}')
        # c,c2=evaluate_contact(model,,i,args.batch_size,)
