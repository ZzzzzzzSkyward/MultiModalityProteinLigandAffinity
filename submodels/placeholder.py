'''
测试用
'''
from main import *


class PlaceHolder(nn.Module):
    def __init__(self, params):
        log("init")
        self.linear = nn.Linear(params.protein_size, 256)
        super().__init__()

    def forward(self, protein_seq, compound_seq, *args):
        log("forward")
        output = np.zeros(protein_seq.shape[0])
        return output


if __name__ == '__main__':
    main(PlaceHolder)
