'''
1dCNN+GRU
先CNN后GRU
'''
if __name__ == '__main__':
    from main import *
else:
    from header import *
import torch.nn.functional as F
from protein_encoder import ProteinEncoder
from compound_encoder import CompoundEncoder
zernike_size = 176


class GRU(nn.Module):
    def __init__(self, params):
        super().__init__()
        hidden_size, dropout_prob = params.hidden_size, params.dropout
        output_size = 1
        next_size = max(output_size, int(hidden_size / 4))
        hidden_size = 64
        self.protein_encoder = ProteinEncoder(params)
        params.output_size = hidden_size * 2
        self.ligand_encoder = CompoundEncoder(params)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, next_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(next_size, output_size),
        )

    def forward(self, protein_seq, compound_seq, *args):
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        output = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output

    def pretrained(self, args):
        if args.resume:
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

'''
冻结
'''
class GRU_Freeze(GRU):
    def __init__(self, params):
        super().__init__(params)

    def forward(self, protein_seq, compound_seq, *args):
        with torch.no_grad():
            protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        output = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output

'''
部分冻结（冻结嵌入层）
'''
class GRU_PartialFreeze(GRU):
    def __init__(self, params):
        super().__init__(params)
        from protein_encoder_freeze import ProteinEncoder
        self.protein_encoder = ProteinEncoder(params)

    def forward(self, protein_seq, compound_seq, *args):
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        output = torch.cat((protein_out, compound_out), dim=1)
        output = self.fc(output)
        output = output.squeeze()
        return output


'''
带有泽尼克描述符任务的模型
权重1e-5
'''


class GRU_Aware_Zernike(GRU):
    def __init__(self, params):
        super().__init__(params)
        self.zernike_decoder = nn.Sequential(
            nn.Linear(256, 192),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=params.dropout),
            nn.Linear(192, 192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(192, zernike_size)
        )
        self.lambda_ = 1e-5

    def forward(self, protein_seq, compound_seq, *args):
        protein_out = self.protein_encoder(protein_seq)
        compound_out = self.ligand_encoder(compound_seq)
        output = torch.cat((protein_out, compound_out), dim=1)
        zernike = self.zernike_decoder(output)
        output = self.fc(output)
        output = output.squeeze()
        return [output, zernike]

    def criterion(self):
        return self.loss

    def loss(self, output, label, zernike):
        return F.mse_loss(output[0], label) + self.lambda_ * \
            F.mse_loss(output[1], zernike)


if __name__ == '__main__':
    main(GRU_PartialFreeze)
