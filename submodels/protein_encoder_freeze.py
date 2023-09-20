from header import *
class ProteinEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        embedding_size = 512
        hidden_size = 64
        self.hidden_size = hidden_size
        input_size = params.protein_size
        num_layers = 2
        kernel_size = 3
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_seq):
        with torch.no_grad():
            embedded = self.embedding(input_seq)
        iconv = self.conv(embedded.contiguous().transpose(1, 2))
        iconv = self.pool(iconv).contiguous().transpose(1, 2)
        iconv, _ = self.gru(iconv)
        return iconv[:, -1, :].squeeze()

