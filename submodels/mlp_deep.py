'''
更多层感知机（基线模型）
五层，每层分别是Linear,LeakyReLU(,BatchNorm),Dropout
'''
from main import *


class MLP(nn.Module):
    def __init__(self, params, cross=False):
        super().__init__()
        self.protein_encoder = nn.Sequential(nn.Linear(params.protein_size, 512),
                                             nn.LeakyReLU(),
                                             nn.BatchNorm1d(512),
                                             nn.Dropout(),
                                             nn.Linear(512, 256),
                                             nn.LeakyReLU(),
                                             nn.Dropout(),
                                             nn.Linear(256, 192),
                                             nn.LeakyReLU(),
                                             nn.Dropout(),
                                             nn.Linear(192, 144),
                                             nn.LeakyReLU(),
                                             nn.Dropout(),
                                             nn.Linear(144, 96),
                                             nn.LeakyReLU(),
                                             nn.Dropout(),
                                             nn.Linear(96, 64))
        self.ligand_encoder = nn.Sequential(nn.Linear(params.compound_size, 256),
                                            nn.LeakyReLU(),
                                            nn.BatchNorm1d(256),
                                            nn.Dropout(),
                                            nn.Linear(256, 192),
                                            nn.LeakyReLU(),
                                            nn.Dropout(),
                                            nn.Linear(192, 144),
                                            nn.LeakyReLU(),
                                            nn.Dropout(),
                                            nn.Linear(144, 96),
                                            nn.LeakyReLU(),
                                            nn.Dropout(),
                                            nn.Linear(96, 64))
        self.decoder = nn.Sequential(nn.Linear(128, 96),
                                     nn.BatchNorm1d(96),
                                     nn.Dropout(),
                                     nn.LeakyReLU(),
                                     nn.Linear(96, 64),
                                     nn.Dropout(),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 8),
                                     nn.LeakyReLU(),
                                     nn.Linear(8, 1))
        self.concator = self.concat if not cross else self.interact
        self.tanh = nn.Tanh()
        self.interactionfc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(64, 64)
        )
        self.interactionfc2 = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128)
        )

    def concat(self, protein, compound):
        return torch.cat((protein, compound), dim=-1)

    def interact(self, protein, compound):
        fused = torch.cat((protein, compound), dim=-1)
        fused = self.interactionfc(fused)
        fused = torch.cat((fused, protein, compound), dim=-1)
        fused = self.interactionfc2(fused)
        return fused

    def forward(self, protein_seq, compound_seq):
        protein_seq = protein_seq.to(torch.float)
        compound_seq = compound_seq.to(torch.float)
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        output = self.concator(protein_out, compound_out)
        output = self.decoder(output)
        output = output.squeeze()
        return output

if __name__ == '__main__':
    main(MLP)
