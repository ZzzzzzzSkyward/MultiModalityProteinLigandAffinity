from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def Eval(model, loaders, args, load=True):
    model.eval()
    DEVICE = get_device()
    log('Eval start.', 'Device:', DEVICE.type, DEVICE.index)
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        log(f"loss={checkpoint['min_test_loss']:.2f}")
    for i in loaders:
        p, r, t, h = evaluate_affinity(model, i)
        addloss4(p, r, t, h)
        # c,c2=evaluate_contact(model,,i,args.batch_size,)


SELFIES_CHARS = [
    ' ', '#', '(', ')', '+', '-', '.', '/', ':', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '=', '_', '$', '%', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '|', '\\', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


def Eval_display(model, loader, args, load=True):
    model.eval()
    DEVICE = get_device()
    model.to(DEVICE)
    log('Eval start.', 'Device:', DEVICE.type, DEVICE.index)
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        log(f"loss={checkpoint['min_test_loss']:.2f}")
    input = None
    output = None
    with torch.no_grad():
        for batch in loader:
            batch = batch[0]
            input = batch[0].detach().cpu().numpy()
            batch = batch.to(DEVICE)
            outputs = model(batch)
            output = outputs.detach().cpu().numpy()[0]
            break
    l = len(SELFIES_CHARS)
    input = [SELFIES_CHARS[i] for i in input if i > 0]
    input = "".join(input)
    output = [SELFIES_CHARS[min(i.argmax(), l - 1)]
              for i in output if i.argmax() > 0]
    output = "".join(output[:len(input)])
    log(f"\nreal:{input}\npred:{output}")


def Eval_count(model, loader, args, load=True):
    model.eval()
    DEVICE = get_device()
    model.to(DEVICE)
    log('Eval start.', 'Device:', DEVICE.type, DEVICE.index)
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        log(f"loss={checkpoint['min_test_loss']:.2f}")
    input = None
    output = None
    with torch.no_grad():
        for batch in loader:
            batch = batch[0]
            input = batch[0].detach().cpu().numpy()
            batch = batch.to(DEVICE)
            outputs = model(batch)
            output = outputs.detach().cpu().numpy()[0]
            break
    l = len(SELFIES_CHARS)
    # count word freq

    input = "".join(input)
    output = [SELFIES_CHARS[min(i.argmax(), l - 1)]
              for i in output if i.argmax() > 0]
    output = "".join(output[:len(input)])
    log(f"\nreal:{input}\npred:{output}")


def Eval_show(model, loader, args, load=True):
    model.eval()
    DEVICE = get_device()
    model.to(DEVICE)
    log('Eval start.', 'Device:', DEVICE.type, DEVICE.index)
    # load data
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if load:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        log(f"loss={checkpoint['min_test_loss']:.2f}")
    input = None
    output = None
    with torch.no_grad():
        for batch in loader:
            output0 = batch[-1][0].detach().cpu().numpy()
            input = batch
            input.pop()
            input = move_to(*input, device=DEVICE)
            outputs = model(*input)
            output = outputs[0].detach().cpu().numpy()
            break
    l = len(SELFIES_CHARS)
    # count word freq

    input = output0
    output = [np.argmax(i) for i in output]
    log(f"\nreal:{input}\npred:{output}")
