from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def Eval(model, loaders, args, load=True):
    model.eval()
    DEVICE = get_device()
    log('Eval start', 'Using device:', DEVICE.type, DEVICE.index)
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    for i in loaders:
        p, r, t, h = evaluate_affinity(model, i)
        addloss4(p, r, t, h)
        # c,c2=evaluate_contact(model,,i,args.batch_size,)
