from header import *


class ConcatenationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1.protein 1d sequences
        self.protein_1d = nn.Embedding(AMINO_ACID_AMOUNT, AMINO_EMBEDDING_SIZE)
        # Gated Recurrent Unit
        self.protein_1d_rnn = nn.GRU(
            AMINO_EMBEDDING_SIZE, GRU_SIZE, batch_first=True)

        #Graph Attention Network
        self.protein_2d_gan=GAN(in_channels=16, out_channels=32, heads=4, dropout=0.5)

    def forward(self, protein_1d_data):
        protein_1d_embedding = self.protein_1d(protein_1d_data)