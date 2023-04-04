from header import *

#TODO add more lambda, fix wrongly named lambda
class param:
    l0 = 0
    l1 = 0
    l2 = 0
    l3 = 0
    lr = 1e-4  # learning_rate
    bs = 32  # batch_size
    ep = 200  # epoch
    sd = 0  # seed

    def load_from_cli(self):
        self.l0=args.l0
        self.l1=args.l1
        self.l2=args.l2
        self.l3=args.l3
        self.lr=args.lr
        self.bs=args.batch_size
        self.ep=args.epoch


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
