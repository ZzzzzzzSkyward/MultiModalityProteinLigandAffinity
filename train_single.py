from logg import *
from header import *
from constants import *
from measurement import *
from pytorchutil import *
try:
    import adai_optim
except BaseException:
    pass

input_len = 2

def train(model, loader, optimizer, criterion):
    total_loss = 0
    length = len(loader)
    DEVICE = get_device()
        # print(input1.shape, input2.shape, labels.shape)
    for batch, il in enumerate(loader):
        il = move_to(*il, device=DEVICE)
        inputs = il[:input_len]
        labels = il[input_len:]
        outputs = model(*inputs)
        # print(outputs.shape, labels.shape)
        optimizer.zero_grad()
        # loss = criterion(outputs, labels, input=[input1, input2])
        loss = criterion(outputs, *labels)
        total_loss += np.sum(getloss(loss))
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss / length


def train_add(model, loader, optimizer, criterion, leng):
    total_loss = 0
    length = len(loader)
    DEVICE = get_device()
    for batch, il in enumerate(loader):
        il = move_to(*il, device=DEVICE)
        inputs = il[:input_len]
        labels = il[input_len:]
        if leng < batch:
            return total_loss / batch
        # print(input1.shape, input2.shape, labels.shape)
        outputs = model(*inputs)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        total_loss += np.sum(getloss(loss))
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
        for batch, il in enumerate(loader):
            il = move_to(*il, device=DEVICE)
            inputs = il[:input_len]
            labels = il[input_len:]
            outputs = model(*inputs)
            loss = criterion(outputs, *labels)
            total_loss += np.sum(getloss(loss))
    return total_loss / length


def calc(args):
    addloss4(*args)


def test_test(model, loaders, criterion):
    DEVICE = get_device()
    total_loss = 0
    length = len(loaders[0])
    loader=loaders[0]
    with torch.no_grad():
        for batch, il in enumerate(loader):
            il = move_to(*il, device=DEVICE)
            inputs = il[:input_len]
            labels = il[input_len:]
            outputs = model(*inputs)
            loss = criterion(outputs, *labels)
            total_loss += np.sum(getloss(loss))
    tasks = [evaluate_affinity_val(model, i, calc)for i in loaders]
    pool = multiprocessing.Pool(processes=1)  # sequential
    # run each fn in fns and gather the results to addloss4, no need to wait
    # for all to finish
    pool.starmap(evaluate_affinity_calc, tasks)
    pool.close()
    return total_loss / length


def Train(model, loader_train, loader_test, args):
    # set up device
    DEVICE = get_device()
    log('Train Start.', 'Device:', DEVICE.type, DEVICE.index)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(reduction='mean')
    optimizer = model.optimizer() if hasattr(
        model, 'optimizer') else optim.AdamW(model.parameters(), lr=lr)
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
            test_loss = test(
                model,
                loader_test,
                criterion) if not isinstance(
                loader_test,
                list) else test_test(
                model,
                loader_test,
                criterion)
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
    log('Train_Add Start.', 'Device:', DEVICE.type, DEVICE.index)

    # move model to device
    model.to(DEVICE)

    # set up loss function and optimizer
    num_epochs = args.epoch
    lr = args.lr
    criterion = model.criterion() if hasattr(
        model, 'criterion') else nn.MSELoss(reduction='mean')
    optimizer = model.optimizer() if hasattr(
        model, 'optimizer') else optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    min_test_loss = 1e10
    checkpoint_pth = args.name if hasattr(args, "name") else "test_pth"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_pth, DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if not args.explore:
            min_test_loss = checkpoint['min_test_loss']
    epoch = 0
    train_loss = 0
    progress = tqdm(range(num_epochs))
    progress.update(max(0, start_epoch - 1))

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
        if epoch % 5 == 0 or args.detailed:
            # test model
            model.eval()
            test_loss = test(
                model,
                loader_test,
                criterion) if not isinstance(
                loader_test,
                list) else test_test(
                model,
                loader_test,
                criterion)
            epochloss2(epoch, train_loss, test_loss)
            if min_test_loss > test_loss:
                min_test_loss = test_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    min_test_loss,
                    checkpoint_pth)

    if epoch % 5 == 0:
        # test model
        model.eval()
        test_loss = test(model, loader_test, criterion)
        epochloss2(epoch, train_loss, test_loss)
