from header import *
from pytorchutil import *

# 1 dimensional model for protein sequences and compound smiles


class OneDimensionalAffinityModel(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.redirect = OneDimensionalEncoderAffinityModel(params)

    def forward(self, protein_seq, compound_seq):
        return self.redirect(protein_seq, compound_seq)


class ProteinEncoder(nn.Module):
    def decoder(self, x):
        if self.zernike:
            decoded = self.decoder_zernike(x)
        else:
            decoded = self.decoder_count(x)
        return decoded

    def decoder_count(self, x):
        decoded = self.linear(x)
        return decoded

    def decoder_zernike(self, x):
        decoded = self.fc2(x)
        return decoded

    def __init__(self, params):
        embedding_size = 512
        hidden_size = 64
        self.hidden_size = hidden_size
        input_size = params.protein_size
        num_layers = 2
        next_size2 = int(hidden_size * 2.2)
        output_size = 176
        self.zernike = True
        num_heads = 8
        kernel_size = 3
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru_encoder = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.gru_decoder = nn.GRU(
            input_size=hidden_size * 2,  # bidirectional
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = MultiHeadAttention(
            hidden_size * 2,
            num_heads
        )
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru_encoder2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, next_size2),
            nn.LeakyReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(next_size2, next_size2),
            nn.LeakyReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(next_size2, output_size)
        )

    def zernike_encode(self, input):
        pass

    def forward_encode(self, input_seq):
        # Embedding
        embedded = self.embedding(input_seq)

        # Encoding
        with torch.no_grad():
            iconv = self.conv(embedded.contiguous().transpose(1, 2))
            iconv = self.pool(iconv).contiguous().transpose(1, 2)
            hidden = None
            encoder_output, hidden = self.gru_encoder(embedded)
        encoder_output = self.pool(
            encoder_output.contiguous().transpose(
                1, 2)).contiguous().transpose(
            1, 2)
        gru_out, _ = self.gru_encoder2(iconv)
        # print(encoder_output.shape, gru_out.shape)
        encoder_output = gru_out + encoder_output
        return encoder_output, hidden

    def encode(self, input_seq):
        # print(input_seq)
        # print(input_seq.shape)
        output, hidden = self.forward_encode(input_seq)
        encoded = output[:, -1, :].squeeze()

        return encoded

    def forward(self, input_seq):

        encoder_output, hidden = self.forward_encode(input_seq)
        # Decoding with self-attention and residual connection
        decoder_input = encoder_output[:, -1, :].squeeze()
        # decoder_output, _ = self.gru_decoder(decoder_input)
        # attention_output = self.attention(decoder_output)
        # decoder_input = attention_output + decoder_input  # Residual
        # connection

        # Decoding with linear layer
        decoded = self.decoder(decoder_input)
        # print(decoded.shape)
        decoded = decoded.squeeze()

        return decoded


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.dropout_prob = 0.3
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_heads, self.dropout_prob, batch_first=True)
        self.relu = nn.Mish()

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        # pos_enc: (seq_len, batch_size, d_model)
        # attn_mask: (seq_len, seq_len) or None
        output, _ = self.multihead_attn(x, x, x)
        output = self.relu(output)
        return output


class LigandEncoder(nn.Module):
    fingerprint = True

    def __init__(self, params):
        input_size = params.compound_size
        vocab_size = 128
        embedding_size = 64
        hidden_size = 64
        num_layers = 2
        num_heads = 4
        dropout = 0.1
        output_size = 176
        kernel_size = 3
        fingerprint_size = 128
        self.hidden_size = hidden_size
        super().__init__()
        self.mse = nn.MSELoss()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.attention = MultiHeadAttention(
            hidden_size * 2, num_heads)

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(
                in_channels=128,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        self.map = nn.Linear(hidden_size * 6, hidden_size * 2)
        self.map2 = nn.Linear(embedding_size * 2, embedding_size)
        self.map3 = nn.Linear(embedding_size * 2, embedding_size)
        next_size = int(hidden_size * 1.4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, next_size,
                      bias=True),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(next_size, vocab_size)
        )
        self.sumfc = nn.Linear(input_size, 1)
        self.attention = MultiHeadAttention(
            hidden_size * 2,
            num_heads
        )
        self.conv = nn.Conv1d(
            embedding_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.fingerfc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, fingerprint_size)
        )

    def encode(self, x):
        # Encoder
        x = x.to(torch.long)
        embedded = self.embedding(x)
        # multi head attention
        # encoded1, h = self.encoder(embedded)
        # attention = self.attention(encoded1)
        # print(encoded1.shape, attention.shape)
        # encoded = encoded1[:, -1, :].squeeze()
        # attention = attention[:, -1, :].squeeze()
        # print(encoded1.shape, attention.shape)
        # encoded = attention + encoded
        # encoded=[batch_size,hidden_size*2]

        # conv
        iconv = self.conv(embedded.contiguous().transpose(1, 2))
        iconv = self.pool(iconv).contiguous().transpose(1, 2)
        encoder_output, hidden = self.encoder(embedded)
        encoder_output = self.pool(
            encoder_output.contiguous().transpose(
                1, 2)).contiguous().transpose(
            1, 2)
        gru_out, _ = self.gru2(iconv, hidden[:, :, :self.hidden_size])
        # print(encoder_output.shape, gru_out.shape)
        encoder_output = encoder_output + gru_out
        encoder_output = encoder_output[:, -1, :].squeeze()
        return encoder_output

    def decoder(self, x):
        if self.fingerprint:
            decoded = self.decoder_fingerprint(x)
        else:
            decoded = self.decoder_count(x)
        return decoded

    def decoder_count(self, x):
        decoded = self.fc(x)
        return decoded

    def decoder_fingerprint(self, x):
        decoded = self.fingerfc(x)
        return decoded

    def forward(self, x):
        encoded = self.encode(x)
        # encoded=encoded.unsqueeze(1).repeat(1, 1024, 1)
        # print(encoded.shape)
        # [batch_size, seq_len, hidden_size*2]
        # encoded = encoded.contiguous().transpose(0, 1)
        # encoded, _ = self.attention(encoded, encoded, encoded)
        # [batch_size, seq_len, hidden_size*2]
        # encoded = encoded.contiguous().transpose(0, 1)

        # Decoder
        decoded = self.decoder(encoded)
        # decoded = self.decoder(encoded)
        # print(decoded.shape)
        decoded = decoded.squeeze()

        # print(decoded.shape)
        # decoded = self.fc(decoded)
        # print(decoded.shape)

        return decoded
    printrnd = 100

    def character_freq_loss(self, output, target):
        # 将模型输出转换为词频张量
        batch_size, seq_len = target.size()
        vocab_size = output.size(-1)
        target = target.to(torch.long)
        freq = torch.zeros(
            batch_size,
            vocab_size,
            dtype=torch.float,
            device=output.device)
        freq.scatter_add_(1, target, torch.ones_like(target, dtype=torch.float))
        if self.printrnd == 100:
            self.printrnd = 0
            print(freq[0].detach().to(torch.int).tolist()[:90])
            print(output[0].detach().to(torch.int).tolist()[:90])
        # 计算损失
        loss = F.smooth_l1_loss(output[:, 1:], freq[:, 1:], reduction='sum')
        self.printrnd += 1
        return loss

    def criterion(self):
        if self.fingerprint:
            return self.mse
        else:
            return self.character_freq_loss


class ParallelModel(nn.Module):
    def __init__(self, module_a, module_b):
        super().__init__()
        device = get_device()
        if device.type == 'cuda':
            device = [get_best_gpu()]
        else:
            device = [device]
        self.device_ids = device
        self.module_a = module_a
        self.module_b = module_b

    def forward(self, x, y):
        # Create a shared memory tensor to store the results
        results = mp.Manager().list([None, None])

        # Define a function to run module_a's forward on a separate process
        def run_module_a():
            result_a = self.module_a(x)
            results[0] = result_a

        # Define a function to run module_b's forward on a separate process
        def run_module_b():
            result_b = self.module_b(y)
            results[1] = result_b

        # Create two processes
        p1 = mp.Process(target=run_module_a)
        p2 = mp.Process(target=run_module_b)

        # Start the processes
        p1.start()
        p2.start()

        # Wait for the processes to finish
        p1.join()
        p2.join()

        # Return the results
        return results[0], results[1]


class AffinityDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_size, hidden_size, dropout_prob = params.input_size, params.hidden_size, params.dropout
        next_size = hidden_size // 2
        output_size = 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            # nn.BatchNorm1d(next_size),
            nn.Mish(inplace=True),
            # nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )

    def forward(self, x):
        return self.fc(x)


class OneDimensionalEncoderAffinityModel(nn.Module):
    def __init__(self, params):
        protein_input_size, hidden_size, dropout_prob = params.input_size, params.hidden_size, params.dropout
        compound_input_size = params.compound_size
        output_size = 1
        next_size = max(output_size, int(hidden_size / 4))
        hidden_size = 64
        super().__init__()
        self.protein_encoder = ProteinEncoder(params)
        params.output_size = hidden_size * 2
        #self.protein_encoder = ContactGraphEncoder(params)
        # LE
        self.ligand_encoder = LigandEncoder(params)
        # self.ligand_encoder = C_Encoder(1024, 128, 64, 2, 0.3)
        # self.protein_encoder.forward = self.protein_encoder.encode  # hack
        # LE
        # self.ligand_encoder.forward = self.ligand_encoder.encode  # hack
        # self.parallel_encoder = ParallelModel(self.protein_encoder, self.ligand_encoder)
        # convolution on data as [batch_size,seq_len,embedding_size]
        self.fc = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=(3, 3),
                stride=1,
                padding=1),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size * 4, 1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
        )
        self.fc1 = nn.Linear(hidden_size * 4, output_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            # nn.BatchNorm1d(next_size),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )

    def forward(self, protein_seq, compound_seq):
        # batch_size, seq_len=2048,dim
        # with torch.no_grad():
        protein_out = self.protein_encoder.encode(protein_seq)
        # print(protein_out.shape)
        # batch_size,seq_len,hidden_size
        compound_out = self.ligand_encoder.encode(compound_seq)
        # protein_out, compound_out = self.parallel_encoder(
        #    protein_seq, compound_seq)
        # print(protein_out.shape, compound_out.shape)
        output = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc2(output)
        output = output.squeeze()
        return output

    def pretrained(self, args):
        if args.resume and not args.forcepretrain:
            return
        DEVICE = get_device()
        if args.protein != '':
            checkpoint_pretrained = args.protein
            checkpoint = load_checkpoint(checkpoint_pretrained, DEVICE)
            self.protein_encoder.load_state_dict(
                checkpoint['model_state_dict'], strict=False)
        if args.ligand != '':
            checkpoint_pretrained = args.ligand
            checkpoint = load_checkpoint(checkpoint_pretrained, DEVICE)
            self.ligand_encoder.load_state_dict(
                checkpoint['model_state_dict'], strict=False)


class MLPModel(nn.Module):
    def __init__(self, params, cross=False):
        params.compound_size = 512
        super().__init__()
        self.protein_encoder = nn.Sequential(nn.Linear(params.protein_size, 256),
                                             nn.LeakyReLU(),
                                             nn.BatchNorm1d(256),
                                             nn.Dropout(),
                                             nn.Linear(256, 128),
                                             nn.LeakyReLU(),
                                             nn.Dropout(),
                                             nn.Linear(128, 64))
        self.ligand_encoder = nn.Sequential(nn.Linear(params.compound_size, 256),
                                            nn.LeakyReLU(),
                                            nn.BatchNorm1d(256),
                                            nn.Dropout(),
                                            nn.Linear(256, 128),
                                            nn.LeakyReLU(),
                                            nn.Dropout(),
                                            nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(128, 64),
                                     nn.Dropout(),
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


class C_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        return hidden

    def encode(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.gru(embedded)
        encoded = output[:, -1, :].squeeze()
        return encoded


class C_Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class CompoundSeq2Seq(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_size = params.compound_size
        embed_size = 128
        layer = 2
        dropout = 0.1
        hidden_size = 64
        output_size = 1024  # selfies
        self.vocab_size = 128
        self.encoder = C_Encoder(
            input_size,
            embed_size,
            hidden_size,
            layer,
            dropout)
        self.decoder = C_Decoder(
            output_size,
            embed_size,
            hidden_size,
            layer,
            dropout)
        self.fc = nn.Linear(hidden_size * 2, self.vocab_size)
        self.device = get_device()

    def forward(self, inp, outp, teacher_forcing_ratio=0.5):
        batch_size = outp.shape[0]
        max_len = outp.shape[1]

        outputs = torch.zeros(
            batch_size,
            max_len,
            self.vocab_size).to(
            self.device)
        encoder_outputs = self.encoder(inp)
  #      hidden = encoder_outputs.view(-1, 1, encoder_outputs.shape[-1])

        input = outp[:, 0]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, encoder_outputs)
            output = output.squeeze()
            output = self.fc(output)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = outp[:, t] if teacher_force else top1
        return outputs
    printrnd = 100

    def criterion(self):
        return self.loss

    def loss(self, output, label):
        # 将目标序列转换为tensor
        target_tensor = label.long().unsqueeze(0)
        target_tensor = target_tensor.to(output.device)

        # 计算交叉熵损失
        _criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = _criterion(
            output.view(-1, output.shape[-1]), target_tensor.view(-1))

        return loss


class OneDimensionalToAffinityAndZernike(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = get_device()
        self.protein_encoder = ProteinEncoder(params)
        self.ligand_encoder = LigandEncoder(params)
        params.input_size = params.hidden_size * 2
        self.affinity_decoder = AffinityDecoder(params)
        protein_size = 176
        self.protein_decoder = nn.Sequential(
            nn.Linear(protein_size, protein_size),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(protein_size, params.zernike_size),
        )
        self.ligand_decoder = nn.Sequential(
            nn.Linear(128, params.compound_size),
            nn.GRU(params.compound_size, params.hidden_size, batch_first=True),
        )
        self.toaffinity = nn.Linear(304, params.hidden_size * 2)
        self.protein_param = 0.00001
        self.compound_param = 0.1

    def criterion(self):
        return self.loss

    def loss(self, output, label, input=None):
        # input[0] = input[0].float()
        # input[1] = input[1].float()
        affinity = label[0]
        affinity_loss = F.mse_loss(output, affinity)
        zernike = label[1]
        protein_loss = 0
        # compound_loss = 0
        # for i in range(len(input[0])):
        protein_loss += F.mse_loss(zernike, self._protein_output)
        # compound_loss += F.cross_entropy(input[1][i], self._compound_output[i])
        # Auto fit ratio
        while protein_loss > affinity_loss:
            protein_loss /= 10
        # fixed ratio
        # protein_loss = protein_loss * self.protein_param
        # compound_loss = compound_loss * self.compound_param
        return affinity_loss + protein_loss  # + compound_loss

    def cat(self, protein, compound):
        return torch.cat((protein, compound), dim=-1)

    def forward(self, protein, compound):
        # encode
        protein = self.protein_encoder.encode(protein)
        compound = self.ligand_encoder.encode(compound)
        # concat
        output = self.cat(protein, compound)
        output = self.toaffinity(output)
        # decode to affinity
        output = self.affinity_decoder(output)
        # decode to protein
        self._protein_output = self.protein_decoder(protein)
        # decode to compound
        # self._compound_output, _2 = self.ligand_decoder(compound)
        return output.squeeze()


class ZernikeEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = get_device()
        self.protein_encoder = nn.Sequential(
            nn.Linear(params.zernike_size, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(128, params.output_size)
        )

    def forward(self, protein):
        protein = self.protein_encoder(protein)
        return protein

    def encode(self, protein):
        return self.forward(protein)


class OneDimentionalCrossZernike(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = get_device()
        self.protein_encoder = ProteinEncoder(params)
        self.ligand_encoder = LigandEncoder(params)

        params.output_size = params.hidden_size * 2
        self.affinity_decoder = AffinityDecoder(params)
        self.affinity_decoder_zernike = AffinityDecoder(params)
        protein_size = 176
        self.protein_decoder = nn.Sequential(
            nn.Linear(protein_size, protein_size),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(protein_size, params.zernike_size),
        )
        self.ligand_decoder = nn.Sequential(
            nn.Linear(128, params.compound_size),
            nn.GRU(params.compound_size, params.hidden_size, batch_first=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(384, params.hidden_size * 2),
            nn.BatchNorm1d(params.hidden_size * 2),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(params.hidden_size * 2, params.hidden_size),
            nn.LeakyReLU()
        )

        self.zernike_encoder = ZernikeEncoder(params)
        self.zernike_decoder = nn.Sequential(
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1))

        self.toaffinity = nn.Linear(256, params.output_size)
        self.threetoaffinity = nn.Sequential(nn.Linear(512, 256),
                                             nn.BatchNorm1d(256),
                                             nn.Dropout(),
                                             nn.LeakyReLU(),
                                             nn.Linear(256, 32),
                                             nn.LeakyReLU(),
                                             nn.Linear(32, 1))
        self.protein_param = 0.00001
        self.compound_param = 0.1

    def cat(self, *a):
        return torch.cat(a, dim=-1)

    def combine(self, *a):
        return torch.add(*a)

    def forward(self, protein, compound, zernike):
        protein_encoded = self.protein_encoder.encode(protein)
        ligand_encoded = self.ligand_encoder.encode(compound)
        zernike_encoded = 0
        affinity = 0
        #zernike_encoded = self.zernike_encoder(zernike)
        # fuse protein and zernike
        #protein_encoded = self.cat(protein_encoded, zernike_encoded)
        #protein_encoded = self.fuse(protein_encoded,zernike_encoded)

        # calculate protein+compound->affinity
        affinity = self.cat(protein_encoded, ligand_encoded)
        affinity = self.toaffinity(affinity)
        affinity = self.affinity_decoder(affinity)

        # calculate protein+zernike+compound->affinity
        '''
        affinity = self.cat(
            protein_encoded,
            zernike_encoded,
            ligand_encoded)
        affinity = self.threetoaffinity(affinity)
        '''
        # calculate zernike+compound->affinity
        #zernike_encoded = self.cat(zernike_encoded, ligand_encoded)
        #zernike_encoded = self.zernike_decoder(zernike_encoded)
        # combine
        affinity = self.combine(affinity, zernike_encoded)
        return affinity.squeeze()


class ContactGraphEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim, hidden_dim, output_dim = params.input_size, params.hidden_size, params.output_size
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # params.input_size = hidden_dim
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)   # 通过平均池化将所有节点特征压缩为一个向量
        x = self.graph_encoder(x)
        return x

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)


class CompoundPretrainTopologyEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        params.output_size = 128
        self.ligand_encoder = LigandEncoder(params)
        self.topology_decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(512, params.alpha_size)
        )

    def forward(self, x):
        x = self.ligand_encoder.encode(x)
        x = self.topology_decoder(x)
        return x
