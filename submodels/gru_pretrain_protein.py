'''
通过自编码器预训练蛋白质编码器
（这个自编码器并不能复原蛋白质序列）
'''
from main_pretrain import *
from protein_encoder import ProteinEncoder
import torch.nn.functional as F


class ProteinEncoderPretrain(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = ProteinEncoder(params)
        self.pool = nn.Sequential(
            nn.Conv1d(1, 32, 2, 2, 2),
            nn.AdaptiveMaxPool1d(128)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 8),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, params.protein_size // 4),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear( params.protein_size // 4,params.protein_size // 2),
            nn.Dropout(p=params.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(params.protein_size // 2, params.protein_size)
        )

    def forward(self, protein):
        encoded = self.encoder(protein).unsqueeze(1)
        decoded = self.pool(encoded).contiguous().transpose(1, 2)
        decoded = self.fc(decoded).squeeze()
        decoded = self.fc2(decoded)
        return decoded

    def criterion(self):
        return self.loss

    def loss(self, output, label):
        label=label.to(torch.float32)
        return F.cross_entropy(output, label)

if __name__ == '__main__':
    main(ProteinEncoderPretrain)
