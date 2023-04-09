from header import *


class ConcatenationModel(nn.Module):
    def __init__(self, params={}):
        super().__init__()
        # 1.protein 1d sequences
        self.protein_1d = nn.Embedding(AMINO_ACID_AMOUNT, AMINO_EMBEDDING_SIZE)
        # Gated Recurrent Unit
        self.protein_1d_rnn = nn.GRU(
            AMINO_EMBEDDING_SIZE, GRU_SIZE, batch_first=True)

        # Graph Attention Network
        self.protein_2d_gan = GAN(
            in_channels=16,
            out_channels=32,
            heads=4,
            dropout=0.5)

    def forward(self, protein_1d_data):
        protein_1d_embedding = self.protein_1d(protein_1d_data)

# 1 dimensional model for protein sequences and compound smiles


class ProteinRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_seq):
        input_seq = input_seq.to(torch.int32)
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return hidden


class CompoundRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_seq):
        input_seq = input_seq.to(torch.int32)
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return hidden


class OneDimensionalAffinityModel(nn.Module):
    def __init__(self, params):
        protein_input_size, hidden_size, dropout_prob = params.input_size, params.hidden_size, params.dropout
        compound_input_size = protein_input_size
        output_size = 1
        super().__init__()
        self.protein_rnn = ProteinRNN(protein_input_size, hidden_size)
        self.compound_rnn = CompoundRNN(compound_input_size, hidden_size)
        self.fc = nn.Sequential(
            #nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob),
            nn.Mish(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, protein_seq, compound_seq):
        protein_hidden = self.protein_rnn(protein_seq)
        compound_hidden = self.compound_rnn(compound_seq)
        concat_hidden = torch.cat((protein_hidden, compound_hidden), dim=1)
        output = self.fc(concat_hidden)
        return output
