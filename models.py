from torch.nn.modules.batchnorm import BatchNorm1d
from header import *


class ConcatenationModel(nn.Module):

    def __init__(self, params={}):
        super().__init__()
        # 1.protein 1d sequences
        self.protein_1d = nn.Embedding(AMINO_ACID_AMOUNT, AMINO_EMBEDDING_SIZE)
        # Gated Recurrent Unit
        self.protein_1d_rnn = nn.GRU(AMINO_EMBEDDING_SIZE,
                                     GRU_SIZE,
                                     batch_first=True)

        # Graph Attention Network
        self.protein_2d_gan = GAN(in_channels=16,
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
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, input_seq):
        input_seq = input_seq.to(torch.int32)
        embedded = self.embedding(input_seq)
        output, _ = self.gru(embedded)
        output = output[:, -1, :]
        return output


class CompoundRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_seq):
        input_seq = input_seq.to(torch.int32)
        embedded = self.embedding(input_seq)
        output, _ = self.gru(embedded)
        output = output[:, -1, :]
        return output


class OneDimensionalAffinityModel(nn.Module):

    def __init__(self, params):
        protein_input_size, hidden_size, dropout_prob = params.input_size, params.hidden_size, params.dropout
        compound_input_size = protein_input_size
        output_size = 1
        next_size = max(output_size, int(hidden_size / 4))
        super().__init__()
        self.protein_rnn = ProteinRNN(protein_input_size, hidden_size)
        self.compound_rnn = CompoundRNN(compound_input_size, hidden_size)
        self.fc = nn.Sequential(
            #nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            # nn.MaxPool1d(2),
            nn.BatchNorm1d(next_size),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )

    def forward(self, protein_seq, compound_seq):
        protein_out = self.protein_rnn(protein_seq)
        compound_out = self.compound_rnn(compound_seq)
        concat_hidden = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc(concat_hidden)
        output = output.flatten()
        return output


class ProteinAutoEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.gru_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_seq):
        # Embedding
        embedded = self.embedding(input_seq)

        # Encoding
        encoder_output, hidden = self.gru_encoder(embedded)

        # Decoding with self-attention and residual connection
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)
        decoder_output, _ = self.gru_decoder(decoder_input, hidden)
        decoder_output = F.relu(decoder_output)
        decoder_output = F.dropout(
            decoder_output, p=0.5, training=self.training)
        decoder_output = decoder_output + decoder_input  # Residual connection
        query = decoder_output.transpose(0, 1)
        key = encoder_output.transpose(0, 1)
        value = encoder_output.transpose(0, 1)
        decoder_output, _ = self.attention(query, key, value)
        decoder_output = decoder_output.transpose(0, 1)
        decoder_output = F.relu(decoder_output)
        decoder_output = F.dropout(
            decoder_output, p=0.5, training=self.training)
        decoder_output = decoder_output + decoder_input  # Residual connection

        # Decoding with linear layer
        decoded = self.linear(decoder_output)

        return decoded
