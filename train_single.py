from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def train(model, loader_train, optimizer, criterion):
    total_loss = 0
    for batch, (input1, input2, labels) in enumerate(loader_train):
        input1, input2, labels = move_to(input1, input2, labels, device=DEVICE)
        outputs = model(input1, input2)
        loss = criterion(outputs, labels)
        total_loss += getloss(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss


def test(model, loader_test, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch, (input1, input2, labels) in enumerate(loader_test):
            input1, input2, labels = move_to(
                input1, input2, labels, device=DEVICE)
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            total_loss += getloss(loss)
    return total_loss


def Train(model, loader_train, loader_test, args):
    # set up device
    log('Training Start.', 'Using device:', DEVICE.type)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(
        reduction='mean')
    optimizer = model.optimizer() if hasattr(model, 'optimizer') else optim.Adam(
        model.parameters(), lr=lr)
    progress = tqdm(range(num_epochs))
    start_epoch = 0
    min_test_loss = 1e4
    checkpoint_pth = "test_pth"
    if args.resume:
        checkpoint = torch.load('./weights/' + checkpoint_pth)
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
        if epoch % 100 == 0:
            # test model
            model.eval()
            test_loss = test(model, loader_test, criterion)
            epochloss2(epoch, train_loss, test_loss)
            if min_test_loss > test_loss:
                min_test_loss = test_loss
                torch.save({'epoch': epoch,
                            'min_test_loss': min_test_loss,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           './weights/' + checkpoint_pth + ".pth")

    if epoch % 100 == 0:
        # test model
        model.eval()
        test_loss = test(model, loader_test, criterion)
        epochloss2(epoch, train_loss, test_loss)
