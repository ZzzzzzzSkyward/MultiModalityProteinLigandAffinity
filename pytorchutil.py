from header import *


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
