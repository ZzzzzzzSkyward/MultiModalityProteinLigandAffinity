from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *

DEVICE = get_device()


def train(model, loader, optimizer, criterion):
    total_loss = 0
    length = len(loader)
    for batch, input1 in enumerate(loader):
        input1 = move_to(input1[0], device=DEVICE)[0]
        outputs = model(input1)
        label = input1.float()
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, label)
        total_loss += getloss(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss / length


def test(model, loader, criterion):
    total_loss = 0
    length = len(loader)
    with torch.no_grad():
        for batch, input1 in enumerate(loader):
            input1 = move_to(input1[0],
                             device=DEVICE)[0]
            label = input1.float()
            outputs = model(input1)
            loss = criterion(outputs, label)
            total_loss += getloss(loss)
    return total_loss / length


def Train(model, loader_train, loader_test, args):
    # set up device
    log('Training Start.', 'Using device:', DEVICE.type, DEVICE.index)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(reduction='mean')
    optimizer = model.optimizer() if hasattr(
        model, 'optimizer') else optim.Adam(model.parameters(), lr=lr)
    progress = tqdm(range(num_epochs))
    start_epoch = 0
    min_test_loss = 1e4
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_test_loss = checkpoint['min_test_loss']
    epoch = 0
    train_loss = 0
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

    if epoch % 10 == 0:
        # test model
        model.eval()
        test_loss = test(model, loader_test, criterion)
        epochloss2(epoch, train_loss, test_loss)
