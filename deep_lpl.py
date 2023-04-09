from header import *


class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_conv1=False, strides=1, dropout=0.3):
        super().__init__()

        self.process = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=strides,
                padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        if use_conv1:
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=strides)
        else:
            self.conv1 = None

    def forward(self, x):
        left = self.process(x)
        right = x if self.conv1 is None else self.conv1(x)

        return F.relu(left + right)

# %%


class cnnModule(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=32, dropout=0.3):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(
                in_channel,
                hidden_channel,
                7,
                stride=2,
                padding=3,
                bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2)
        )

        self.cnn = nn.Sequential(
            resBlock(
                hidden_channel,
                out_channel,
                use_conv1=True,
                strides=1,
                dropout=dropout),
            resBlock(out_channel, out_channel, strides=1, dropout=dropout),
            resBlock(out_channel, out_channel, strides=1, dropout=dropout),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.cnn(x)

        return x

# %%


class DeepLPI(nn.Module):
    def __init__(self, molshape, seqshape, dropout=0.3):
        super().__init__()

        self.molshape = molshape
        self.seqshape = seqshape

        self.molcnn = cnnModule(1, 16)
        self.seqcnn = cnnModule(1, 16)

        self.pool = nn.AvgPool1d(5, stride=3)
        self.lstm = nn.LSTM(
            16,
            16,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Linear(round(((300 + 6165) / 4 - 2) * 2 / 3) * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),

            nn.Linear(256, 1),
        )

    def forward(self, mol, seq):
        mol = self.molcnn(mol.reshape(-1, 1, self.molshape))
        seq = self.seqcnn(seq.reshape(-1, 1, self.seqshape))

        # put data into lstm
        x = torch.cat((mol, seq), 2)
        x = self.pool(x)
        # print(x.shape)
        x = x.reshape(-1,
                      round(((self.molshape + self.seqshape) / 4 - 2) / 3),
                      16)

        x, _ = self.lstm(x)
        # fully connect layer
        x = self.mlp(x.flatten(1))

        x = x.flatten()

        return x
