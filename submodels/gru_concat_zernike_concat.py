'''
拼接两次
使用1dCNN编码泽尼克描述符，使用1dCNN+GRU编码氨基酸序列，然后拼接起来
得到的蛋白质编码再与化合物编码拼接
'''
from main import *
from gru import ProteinEncoder, CompoundEncoder
from gru_zernike import ZernikeEncoder

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
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )
        self.zernike_encoder = ZernikeEncoder(params)
        self.fuse = ProteinFuser(params)

    def forward(self, protein_seq, compound_seq, zernike):
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        zernike_out = self.zernike_encoder(zernike)
        output = self.fuse(protein_out, zernike_out)
        output = torch.cat((output, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output


class ProteinFuser(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_size = 160
        self.fc = nn.Sequential(
            nn.Linear(input_size, 144),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(144, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, seq, zernike):
        output = torch.concat([seq, zernike], dim=1)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    main(GRU_Zernike_Concat)
