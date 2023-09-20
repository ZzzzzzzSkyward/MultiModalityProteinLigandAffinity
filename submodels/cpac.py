'''
CPAC
https://github.com/Shen-Lab/CPAC/
交叉作用
GRU编码器，MLP解码器。编码器内运用32-mer分割
由于改变了原始模型的结构，因此无法直接与原论文的结果比较
'''
from main import *


class crossInteraction(nn.Module):
    def __init__(self, params):
        super().__init__()
        embedding_size = 128
        hidden_size = 64
        protein_size = params.protein_size
        self.embedding = nn.Embedding(params.protein_size, embedding_size)
        self.protein_encoder1 = nn.GRU(
            embedding_size, hidden_size, batch_first=True)
        self.protein_encoder2 = nn.GRU(
            hidden_size, hidden_size, batch_first=True)
        self.compound_encoder1 = nn.GRU(
            embedding_size, hidden_size, batch_first=True)
        self.compound_encoder2 = nn.GRU(
            hidden_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.shrink_size = 128
        self.shrink_protein = nn.Linear(protein_size, self.shrink_size)
        self.shrink_ligand = nn.Linear(512, self.shrink_size)
        self.fuse = nn.Linear(2 * embedding_size, embedding_size)
        self.fc0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.1),
                                 nn.Conv2d(
            64, 1, kernel_size=2, stride=2, padding=0),
        )
        self.fc1 = nn.Sequential(nn.Linear(1024, 128),
                                 nn.LeakyReLU(0.1),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 32),
                                 nn.LeakyReLU(0.1),
                                 nn.Dropout(0.5),
                                 nn.Linear(32, 1))

    def forward(self, protein_input, ligand_input):
        protein_embedding = self.embedding(protein_input)
        batch, seq_len, embedding_size = protein_embedding.size()
        bag = seq_len // 32
        division = seq_len // bag
        protein_embedding = protein_embedding.reshape(
            batch * division, bag, embedding_size)
        encoded, hidden = self.protein_encoder1(protein_embedding)
        encoded = encoded.reshape(batch * bag, division, -1)
        encoded_protein, hidden = self.protein_encoder2(encoded)
        encoded_protein = encoded_protein.reshape(
            batch, seq_len, -1).transpose(1, 2)
        encoded_protein = self.shrink_protein(encoded_protein).transpose(1, 2)
        compound_embedding = self.embedding(ligand_input)
        batch, seq_len, embedding_size = compound_embedding.size()
        bag = seq_len // 32
        division = seq_len // bag
        compound_embedding = compound_embedding.reshape(
            batch * division, bag, embedding_size)
        encoded, hidden = self.compound_encoder1(compound_embedding)
        encoded = encoded.reshape(batch * bag, division, -1)
        encoded_compound, hidden = self.compound_encoder2(encoded)
        encoded_compound = encoded_compound.reshape(
            batch, seq_len, -1).transpose(1, 2)
        encoded_compound = self.shrink_ligand(encoded_compound).transpose(1, 2)
        fused = self.tanh(
            torch.einsum(
                'bij,bkj->bikj',
                encoded_protein,
                encoded_compound))
        fused_output = self.fc0(fused.transpose(1, 3))
        fused_output = fused_output.transpose(1, 3)
        fused_output = fused_output.reshape(fused_output.size(0), -1)
        fused_output = self.fc1(fused_output)
        fused_output = fused_output.squeeze()
        return fused_output

if __name__ == '__main__':
    main(crossInteraction)
