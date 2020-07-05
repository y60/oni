import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def oni(zc, oni_itr=5):
    # Bounding Z’s singular values
    v = zc * torch.mm(zc, zc.t()).norm().rsqrt()
    # Calculate covariance matrix
    s = torch.mm(v, v.t())
    # Newton’s iteration
    b = (torch.eye(s.shape[0], device=s.device) * 3 - s) * 0.5
    for i in range(1, oni_itr):
        b = 1.5 * b - 0.5 * torch.mm(torch.mm(b, b), torch.mm(b, s))
    weight = torch.mm(b.t(), v)
    return weight


class Linear_ONI(nn.Module):
    def __init__(self, in_c, out_c, oni_itr=5, orthinit=True, scaling=True):
        super().__init__()
        z = torch.randn([out_c, in_c])
        if orthinit:
            nn.init.orthogonal_(z)
        else:
            nn.init.kaiming_normal_(z)
        self.z = nn.Parameter(z)
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.g = nn.Parameter(torch.ones([out_c, 1]))
        self.in_c = in_c
        self.oni_itr = oni_itr
        self.scaling = scaling

        self.register_buffer("j", torch.eye(
            in_c) - torch.ones([in_c, in_c]) * (1. / in_c))

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        if self.oni_itr >= 1:
            # Centering
            zc = torch.mm(self.z, self.j)
            # oni
            self.weight = oni(zc, oni_itr=self.oni_itr)
            # Learnable scalar to relax the constraint
            #  of orthonormal to orthogonal
            weight = self.weight * self.g
            if self.scaling:
                weight *= np.sqrt(2)
        else:
            self.weight = self.z
            weight = self.weight
        x = F.linear(x, weight, self.bias)
        return x


class Conv2d_ONI(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0,
                 oni_itr=5, orthinit=True, scaling=True):
        super().__init__()
        in_dim = in_c * kernel_size * kernel_size
        z = torch.randn([out_c, in_dim])
        if orthinit:
            nn.init.orthogonal_(z)
        else:
            nn.init.kaiming_normal_(z)
        self.z = nn.Parameter(z)
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.g = nn.Parameter(torch.ones([out_c, 1]))
        self.in_c = in_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.oni_itr = oni_itr
        self.scaling = scaling

        self.register_buffer("j", torch.eye(
            in_dim) - torch.ones([in_dim, in_dim]) * (1. / in_dim))

    def forward(self, x):
        if self.oni_itr >= 1:
            # Centering
            zc = torch.mm(self.z, self.j)
            # oni
            self.weight = oni(zc, oni_itr=self.oni_itr)
            # Learnable scalar to relax the constraint
            #  of orthonormal to orthogonal
            weight = self.weight * self.g
            if self.scaling:
                weight *= np.sqrt(2)
        else:
            self.weight = self.z
            weight = self.weight
        x = F.conv2d(x,
                     weight.view(-1, self.in_c,
                                 self.kernel_size, self.kernel_size),
                     self.bias, self.stride, self.padding)
        return x
