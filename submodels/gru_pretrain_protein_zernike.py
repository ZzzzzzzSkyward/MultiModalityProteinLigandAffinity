'''
预训练蛋白质编码器
预测泽尼克描述符
'''
from main_pretrain import *
from protein_encoder import ProteinEncoder
zernike_size = 176


class ProteinEncoderPretrainZernike(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = ProteinEncoder(params)
        self.fc = nn.Sequential(
            nn.Linear(128, zernike_size // 4),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(zernike_size // 4, zernike_size // 2),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(zernike_size // 2, zernike_size)
        )

    def forward(self, protein):
        encoded = self.encoder(protein)
        decoded = self.fc(encoded)
        return decoded

    def criterion(self):
        return self.loss

    def loss(self, output, label):
        return F.mse_loss(output, label)

if __name__ == '__main__':
    main(ProteinEncoderPretrainZernike)
