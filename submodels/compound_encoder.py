from header import *
class CompoundEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        vocab_size = 128
        embedding_size = 64
        hidden_size = 64
        num_layers = 2
        kernel_size = 3
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = x.to(torch.long)
        embedded = self.embedding(x)
        iconv = self.conv(embedded.contiguous().transpose(1, 2))
        iconv = self.pool(iconv).contiguous().transpose(1, 2)
        iconv,_ = self.gru(iconv)
        iconv=iconv[:, -1, :].squeeze()
        return iconv