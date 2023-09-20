'''
拼接蛋白质的氨基酸序列编码、泽尼克编码、化合物编码
'''
from main import *
from gru import ProteinEncoder, CompoundEncoder

zernike_size = 176
channel_size = 64


class GRU_Zernike_Concat(nn.Module):
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
            nn.Linear(288, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )
        self.zernike_encoder = ZernikeEncoder(params)

    def forward(self, protein_seq, compound_seq, zernike):
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        zernike_out = self.zernike_encoder(zernike)
        output = torch.cat((protein_out, zernike_out, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output


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
    main(GRU_Zernike_Concat)
