from header import *

# TODO add more lambda, fix wrongly named lambda


class param:
    l0 = 0
    l1 = 0
    l2 = 0
    l3 = 0
    lr = 1e-4  # learning_rate
    bs = 32  # batch_size
    ep = 200  # epoch
    sd = 0  # seed
    input_size = 0
    output_size = 0
    hidden_size = 0
    dropout = 0

    def load_from_cli(self, _args={}):
        self.l0 = _args.l0
        self.l1 = _args.l1
        self.l2 = _args.l2
        self.l3 = _args.l3
        self.lr = _args.lr
        self.bs = _args.batch_size
        self.ep = _args.epoch

    def seed(self):
        torch.manual_seed(self.sd)
        torch.cuda.manual_seed_all(self.sd)
        np.random.seed(self.sd)
        random.seed(self.sd)

    @staticmethod
    def optim():
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def verify():
        torch.backends.cudnn.deterministic = True
