'''
从拓扑数据出发
'''
if __name__ == '__main__':
    from main import *
else:
    from header import *
import torch.nn.functional as F
interact_size = [200, 36]


class CNN_Complex(nn.Module):
    def __init__(self, params):
        super().__init__()
        channel_size = 32
        hidden_size = 32
        next_size = hidden_size // 2
        self.cnn_large = nn.Conv2d(1, channel_size, (8, 8), 4, 2)
        self.cnn1 = nn.Conv2d(channel_size, channel_size, (2, 2), 1, 2)
        self.pool1 = nn.Sequential(
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=params.dropout),
            nn.MaxPool2d((4, 4), 4, 2)
        )
        self.shrink = nn.Linear(3072, 128)
        self.cnn_small = nn.Conv2d(1, channel_size, (2, 2), 2, 1)
        self.cnn2 = nn.Conv2d(channel_size, channel_size, (2, 2), 2, 1)
        self.pool2 = nn.Sequential(
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=params.dropout),
            nn.MaxPool2d((16, 16), 8)
        )
        self.shrink2 = nn.Linear(192, 128)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=params.dropout),
            nn.Linear(hidden_size, next_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=params.dropout),
            nn.Linear(next_size, 1),
        )

    def forward(self, complex):
        complex = complex.unsqueeze(1)
        large = self.cnn_large(complex)
        large = self.pool1(large)
        large = self.cnn1(large)
        large = self.shrink(large.reshape(large.shape[0], -1))
        small = self.cnn_small(complex)
        small = self.pool2(small)
        small = self.cnn2(small)
        small = self.shrink2(small.reshape(small.shape[0], -1))
        output = large + small
        output = self.fc(output)
        return output.squeeze()


if __name__ == '__main__':
    main(CNN_Complex)
