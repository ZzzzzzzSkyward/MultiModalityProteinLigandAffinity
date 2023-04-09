from logg import *


def train(model, optimizer, criterion, loader_train):
    total_loss = 0
    for batch, (inputs, labels) in enumerate(loader_train):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += _getloss(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss


def test(model, criterion, loader_test):
    total_loss = 0
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(loader_test):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += _getloss(loss)
    return total_loss


def Train(model, loader_train, loader_test,
          num_epochs=100, lr=0.001):
    # set up device
    log('Training Start.', 'Using device:' + DEVICE)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    criterion = model.criterion() if model.criterion else nn.CrossEntropyLoss()
    optimizer = model.optimizer() if model.optimizer else optim.Adam(
        model.parameters(), lr=lr)
    progress = tqdm(range(num_epochs))
    start_epoch = 0
    checkpoint_pth = "test_pth"
    if args.resume:
        checkpoint = torch.load('./weights/' + checkpoint_pth)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

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
