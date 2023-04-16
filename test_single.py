from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def Eval(model, loaders, args, load_checkpoint=True):
    model.eval()
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load_checkpoint:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    for i in loaders:
        p, r, t, h = evaluate_affinity(model, i)
        addloss4(p,r,t,h)
        # c,c2=evaluate_contact(model,,i,args.batch_size,)
