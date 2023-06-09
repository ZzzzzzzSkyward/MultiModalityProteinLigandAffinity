from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *


def train(model, loader, optimizer, criterion):
    total_loss = 0
    length = len(loader)
    DEVICE = get_device()
    for batch, (input1, input2, labels) in enumerate(loader):
        # print(input1.shape, input2.shape, labels.shape)
        input1, input2, labels = move_to(input1, input2, labels, device=DEVICE)
        outputs = model(input1, input2)
        # print(outputs.shape, labels.shape)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        total_loss += getloss(loss)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss / length


def train_add(model, loader, optimizer, criterion, leng):
    total_loss = 0
    length = len(loader)
    DEVICE = get_device()
    for batch, (input1, input2, labels) in enumerate(loader):
        if leng < batch:
            return total_loss / batch
        # print(input1.shape, input2.shape, labels.shape)
        input1, input2, labels = move_to(input1, input2, labels, device=DEVICE)
        outputs = model(input1, input2)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        total_loss += getloss(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss / length


def test(model, loader, criterion):
    DEVICE = get_device()
    total_loss = 0
    length = len(loader)
    with torch.no_grad():
        for batch, (input1, input2, labels) in enumerate(loader):
            input1, input2, labels = move_to(input1,
                                             input2,
                                             labels,
                                             device=DEVICE)
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            total_loss += getloss(loss)
    return total_loss / length


def Train(model, loader_train, loader_test, args):
    DEVICE = get_device()
    # set up device
    log('Training Start.', 'Device:', DEVICE.type)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(reduction='mean')
    optimizer = model.optimizer() if hasattr(
        model, 'optimizer') else optim.AdamW(model.parameters(), lr=lr)
    progress = tqdm(range(num_epochs))
    start_epoch = 0
    min_test_loss = 1e4
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        progress.update(start_epoch - 1)
        if not args.explore:
            min_test_loss = checkpoint['min_test_loss']
        if hasattr(model, 'resume'):
            model.resume(args)
    if args.pretrained:
        if hasattr(model, 'pretrained'):
            model.pretrained(args)
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

    if epoch % 5 == 0 or args.detailed:
        # test model
        model.eval()
        test_loss = test(model, loader_test, criterion)
        epochloss2(epoch, train_loss, test_loss)


def Train_Add(model, loader_train, loader_test, args):
    DEVICE = get_device()
    # set up device
    log('Training Start.', 'Device:', DEVICE.type)

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
        if not args.explore:
            min_test_loss = checkpoint['min_test_loss']
    epoch = 0
    train_loss = 0
    for epoch in range(start_epoch, num_epochs):
        # train model
        model.train()
        # calculate loader length
        leng = int(max(((epoch + 1) / num_epochs)**3 * len(loader_train), 1))
        loop_length = min(100, max(int((num_epochs / (epoch + 1))
                                       ** 0.8), int(epoch**0.2))) - 1
        log(f'length={leng},extraloop={loop_length}')
        for i in range(loop_length):
            train_loss = train_add(
                model, loader_train, optimizer, criterion, leng)
        train_loss = train_add(model, loader_train, optimizer, criterion, leng)
        addloss(train_loss)
        progress.update(1)
        if epoch % 10 == 0 or args.detailed:
            # test model
            model.eval()
            test_loss = test(model, loader_test, criterion)
            epochloss2(epoch, train_loss, test_loss)
            if min_test_loss > test_loss:
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
