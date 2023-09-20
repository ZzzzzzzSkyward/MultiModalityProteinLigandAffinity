'''
输入为泽尼克描述符与SMILES
'''
from main import *
zernike_size = 176
channel_size = 64


class GRU_Zernike(nn.Module):
    def __init__(self, params):
        super().__init__()
        hidden_size, dropout_prob = params.hidden_size, params.dropout
        output_size = 1
        next_size = max(output_size, int(hidden_size / 4))
        hidden_size = 64
        self.protein_encoder = ProteinEncoder(params)
        params.output_size = hidden_size * 2
        self.ligand_encoder = CompoundEncoder(params)
        self.fc = nn.Sequential(
            nn.Linear(160, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )
        self.zernike_encoder = ZernikeEncoder(params)

    def forward(self, protein_seq, compound_seq, zernike, *args):
        protein_out = self.zernike_encoder(zernike)
        compound_out = self.ligand_encoder(compound_seq)
        output = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output


class ProteinEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        embedding_size = 512
        hidden_size = 64
        self.hidden_size = hidden_size
        input_size = params.protein_size
        num_layers = 2
        kernel_size = 3
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        iconv = self.conv(embedded.contiguous().transpose(1, 2))
        iconv = self.pool(iconv).contiguous().transpose(1, 2)
        iconv, _ = self.gru(iconv)
        return iconv[:, -1, :].squeeze()


class CompoundEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        vocab_size = 128
        embedding_size = 64
        hidden_size = 64
        num_layers = 2
        kernel_size = 3
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = x.to(torch.long)
        embedded = self.embedding(x)
        iconv = self.conv(embedded.contiguous().transpose(1, 2))
        iconv = self.pool(iconv).contiguous().transpose(1, 2)
        iconv, _ = self.gru(iconv)
        iconv = iconv[:, -1, :].squeeze()
        return iconv


class ZernikeEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channel_size, 4, 2, 1),
            nn.MaxPool1d(4, 2, 1),
            nn.Conv1d(channel_size, channel_size // 2, 2, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(45, zernike_size // 4),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(zernike_size // 4, zernike_size // 8),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(zernike_size // 8, 1)
        )

    def forward(self, zernike):
        zernike = self.conv(zernike.unsqueeze(1))
        zernike = self.fc(zernike)
        return zernike.squeeze()


if __name__ == '__main__':
    main(GRU_Zernike)
