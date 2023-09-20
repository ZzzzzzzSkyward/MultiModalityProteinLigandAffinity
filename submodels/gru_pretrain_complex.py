'''
使用complex_interact的数据预训练
'''
from main_pretrain import *
import torch.nn.functional as F
from cnn_complex_interact import CNN_Complex
from gru import GRU


class GRU_Complex_Pretrain(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = GRU(params)
        self.labeler = CNN_Complex(params)

    def forward(self, protein, compound, interact):
        encoded = self.encoder(protein, compound)
        with torch.no_grad():
            labeled = self.labeler(interact)
        return [encoded, labeled]

    def criterion(self):
        return self.loss

    def loss(self, output, label):
        real_label = output[1]
        output = output[0]
        return F.mse_loss(output, real_label)

    def pretrained(self,args):
        if args.resume:
            return
        DEVICE = get_device()
        if args.protein!='':
            checkpoint_pretrained = args.protein
            checkpoint = load_checkpoint(checkpoint_pretrained, DEVICE)
            self.labeler.load_state_dict(
                checkpoint['model_state_dict'], strict=False)


if __name__ == '__main__':
    main(GRU_Complex_Pretrain)
