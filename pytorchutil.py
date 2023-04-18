from header import *
from getgpu import *


def activate(activation_name="Mish"):
    '''
    ReLU
    Sigmoid
    Tanh
    SiLU
    LeakyReLU
    ELU
    Softmax
    Mish
    '''
    try:
        return getattr(nn, activation_name)()
    except BaseException:
        print(activation_name, "do not exist")
        return nn.Mish()


def dropout(rate=0.1):
    return nn.Dropout(rate)


def initialize_weight(model):
    if isinstance(model, nn.Conv1d):
        nn.init.kaiming_uniform_(model.weight.data, nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)

    elif isinstance(model, nn.BatchNorm1d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)

    elif isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)
    else:
        if model.weight and model.weight.data:
            nn.init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            nn.init.kaiming_uniform_(model.bias.data)


def move_to(*tensors, device=None):
    if not device:
        device = torch.device('cpu')
    device = torch.device(device)
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return tuple(moved_tensors)


def load_checkpoint(path, device=None):
    full_path = './weights/' + path + ".pth"
    if not os.path.exists(full_path):
        raise Exception("Checkpoint not found: {}".format(path))
    return torch.load(full_path, map_location=device)


def save_checkpoint(model, optimizer, epoch, min_test_loss, path):
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    torch.save(
        {
            'epoch': epoch,
            'min_test_loss': min_test_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, './weights/' + path + ".pth")


device = None


def get_device():
    global device
    """
    返回显存占用最少的可用 GPU 的设备号
    """
    if device:
        return device
    try:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            return device
    except BaseException:
        device = torch.device("cpu")
        return device
    device_count = torch.cuda.device_count()
    if device_count == 0:
        # 如果没有可用的 GPU，则返回 CPU 设备
        device = torch.device("cpu")
    elif device_count == 1:
        # 如果只有一个 GPU，则返回该 GPU 设备
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{get_best_gpu()}")
    return device
