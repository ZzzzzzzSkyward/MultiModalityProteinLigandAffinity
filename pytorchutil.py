from header import *


class torchutil:
    @staticmethod
    def activate(activation_name="Mish"):
        try:
            return getattr(nn, activation_name)()
        except BaseException:
            print(activation_name, "do not exist")
            return nn.Mish()
    __doc__ = '''
    ReLU
    Sigmoid
    Tanh
    SiLU
    LeakyReLU
    ELU
    Softmax
    Mish
    '''

    @staticmethod
    def dropout(rate=0.1):
        return nn.Dropout(rate)
