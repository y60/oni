import torch.nn as nn
from oni_module import Linear_ONI


class MLP_ONI(nn.Sequential):
    def __init__(self, in_c, out_c, f_c=256, depth=10, oni_itr=5,
                 orthinit=True, scaling=True):
        assert depth >= 2
        layers = []
        layers.append(Linear_ONI(
            in_c, f_c, oni_itr=oni_itr, orthinit=orthinit, scaling=scaling))
        layers.append(nn.ReLU())
        for i in range(1, depth-1):
            layers.append(Linear_ONI(
                f_c, f_c, oni_itr=oni_itr, orthinit=orthinit, scaling=scaling))
            layers.append(nn.ReLU())
        layers.append(Linear_ONI(
            f_c, out_c, oni_itr=oni_itr, orthinit=orthinit))
        layers.append(nn.LogSoftmax(dim=1))
        super().__init__(*layers)
