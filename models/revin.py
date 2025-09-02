# models/revin.py
import torch, torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode='norm', stats=None):
        # x: [B, L, C]
        if mode == 'norm':
            mean = x.mean(dim=1, keepdim=True)
            var  = x.var(dim=1, keepdim=True, unbiased=False)
            x_n = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n, (mean, var)
        else:  # 'denorm'
            mean, var = stats
            x_d = x
            if self.affine:
                x_d = (x_d - self.beta) / (self.gamma + 1e-8)
            x_d = x_d * torch.sqrt(var + self.eps) + mean
            return x_d
