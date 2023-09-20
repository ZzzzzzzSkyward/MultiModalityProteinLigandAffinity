'''
Contains useless functions
'''
from header import *
from pytorchutil import *
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerXL(nn.Module):
    def __init__(self, n_layer=8, n_head=8, d_model=1024, d_ff=8, dropout=0.1):
        super(TransformerXL, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.embed = nn.Embedding(512, 1)
        self.layers = [nn.ModuleList([
            MultiHeadAttention(d_model, n_head),
            FeedForward(d_model, d_ff, dropout)
        ]) for _ in range(n_layer)]
        for i in self.layers:
            i.to(get_device())

    def forward(self, x):
        x = self.embed(x).squeeze()
        for i in range(self.n_layer):
            attn, ff = self.layers[i]
            x = attn(x)
            x = ff(x)
        return x, self.layers[-1]


class CompoundAutoEncoder(nn.Module):

    def character_loss(self, y_pred, y_true):
        cut = 1024
        # mask = (y_true != 0).float()
        rate = 1
        y_true = int8_to_onehot(y_true, device=get_device())
        ce_loss = F.cross_entropy(
            y_pred[:, :cut], y_true[:, :cut], reduction='sum')
        mse_loss = F.mse_loss(y_pred[:, :cut], y_true[:, :cut])
        # masked_loss = (mse_loss * mask).mean()
        loss = mse_loss * (1 - rate) + ce_loss * rate
        return loss

    def criterion(self):
        return self.character_loss

    def __init__(self, params):
        embedding_size = 64
        hidden_size = 64
        input_size = params.compound_size
        self.input_size = input_size
        num_layers = 4
        num_heads = 8
        self.vocab = 128
        dropout = params.dropout
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = 8
        self.map_size = hidden_size * 2
        self.input_size2 = hidden_size
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        # self.t_encoder = TransformerXL()
        self.attention_encoder = MultiHeadAttention(
            self.map_size, self.num_heads)
        self.linear_map = nn.Linear(input_size,
                                    self.map_size)
        self.gru_encoder = nn.GRU(
            input_size=embedding_size,
            bidirectional=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.gru_decoder = nn.GRU(
            input_size=self.input_size2,  # bidirectional
            hidden_size=self.input_size2 // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_layers * 2,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=input_size,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU()
        )
        self.inter_linear = nn.Linear(hidden_size * num_heads, input_size)
        self.linear = nn.Linear(self.input_size2, self.vocab)
        self.softmax = nn.Softmax(dim=2)

    def forward_encode(self, input_seq):
        # Embedding
        # print(input_seq.shape, self.input_size)
        # F.pad(input_seq, (0, pad_size), mode='constant')
        input_seq = input_seq.to(torch.int32)
        input_seq = move_to(input_seq, device=get_device())[0]
        '''
        input_seq = [torch.zeros(1, self.input_size), *input_seq]
        # print(input_seq.shape)
        pad_size = self.input_size
        rnn_utils.pad_sequence(
            input_seq,
            batch_first=True,
            padding_value=0)
        input_seq = input_seq[1:]
        batch_size, seq_len = input_seq.size()
        '''
        # Encoding
        input_seq = self.embedding(input_seq)
        output, hidden = self.gru_encoder(input_seq)
        # print(output.shape)
        # (batch_size, seq_len, num_heads*(hidden_size*2//num_heads))
        # linear_out = self.linear_map(output.permute(0,2,1))
        linear_out = output
        # print(linear_out.shape)
        # attention_output, _ = self.attention_encoder(
        #    linear_out, linear_out, linear_out)  # (batch_size, num_heads*head_dim, seq_len)
        # head_vectors = attention_output  # .permute(0, 2, 1)
        # print("ohh", output.shape, head_vectors.shape, hidden.shape)
        # output = torch.cat([output, head_vectors], dim=-1)
        return output, hidden, None

    def encode(self, input_seq):

        output, hidden, attention = self.forward_encode(input_seq)
        return hidden

    def forward(self, input_seq):
        encoder_output, hidden, attention = self.forward_encode(input_seq)
        decoder_input = hidden.permute(1, 0, 2).view(len(input_seq), -1)
        # decoder_input = torch.concat([decoder_input, attention], dim=-1)
        # cnn_output = self.cnn(decoder_input)
        cnn_output = self.inter_linear(decoder_input)
        # print(decoder_input.shape, cnn_output.shape)
        decoder_output, _ = self.gru_decoder(cnn_output)
        decoder_output = F.leaky_relu(decoder_output)
        decoder_output = F.dropout(
            decoder_output, p=0.5, training=self.training, inplace=True)
        decoder_output = self.linear(decoder_output)
        '''
        attention_output, _ = self.attention(
            decoder_output, encoder_output, encoder_output)
        decoder_output = decoder_output + attention_output
        '''
        # print(decoder_output.shape, decoder_input.shape, encoder_output.shape)
        decoder_output = decoder_output.squeeze(dim=-1)
        decoder_output = self.softmax(decoder_output)
        # print(decoder_output.shape)
        return decoder_output
