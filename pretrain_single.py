from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *
try:
    import adai_optim
except BaseException:
    pass


DEVICE = None


def train(model, loader, optimizer, criterion):
    global DEVICE
    total_loss = 0
    length = len(loader)
    for batch, inputs in enumerate(loader):
        inputs = move_to(*inputs, device=DEVICE)
        outputs = model(*inputs[:-1])
        label = inputs[-1]
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, label)
        total_loss += np.sum(getloss(loss))
        optimizer.zero_grad()
        loss.backward()
        # very important for gru like structure, use 1e-5 for initial
        nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()

    return total_loss / length


def test(model, loader, criterion):
    total_loss = 0
    length = len(loader)
    with torch.no_grad():
        for batch, inputs in enumerate(loader):
            inputs = move_to(*inputs, device=DEVICE)
            outputs = model(*inputs[:-1])
            label = inputs[-1]
            #print(outputs[0],label[0])
            loss = criterion(outputs, label)
            total_loss += np.sum(getloss(loss))
    return total_loss / length


def Train(model, loader_train, loader_test, args):
    # set up device
    global DEVICE
    DEVICE = get_device()
    log('Pretrain Start.', 'Device:', DEVICE.type, DEVICE.index)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr  # actually useless
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(reduction='mean')
    optimizer = model.optimizer() if hasattr(
        model, 'optimizer') else optim.AdamW(model.parameters(), lr=lr)
    # adai_optim.AdaiV2
    optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    min_test_loss = 1e10
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if not args.newoptim:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if not args.explore:
            min_test_loss = checkpoint['min_test_loss']
        if hasattr(model, 'resume'):
            model.resume(args)
    else:
        winit(model)
    if args.pretrained:
        if hasattr(model, 'pretrained'):
            model.pretrained(args)
    epoch = 0
    train_loss = 0
    progress = tqdm(range(num_epochs))
    progress.update(max(0, start_epoch - 1))
    for epoch in range(start_epoch, num_epochs):
        # train model
        model.train()
        train_loss = train(model, loader_train, optimizer, criterion)
        addloss(train_loss)
        progress.update(1)
        if epoch % 5 == 0 or args.detailed:
            # test model
            model.eval()
            test_loss = test(model, loader_test, criterion)
            epochloss2(epoch, train_loss, test_loss)
            if min_test_loss > test_loss or args.track:
                min_test_loss = test_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    min_test_loss,
                    checkpoint_pth)
