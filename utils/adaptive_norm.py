# utils/adaptive_norm.py
import torch
import torch.nn as nn


class EMATrend(nn.Module):
    """简单 EMA 去趋势（逐样本逐通道）"""
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        trend = torch.zeros_like(x)
        trend[:, 0] = x[:, 0]
        g = self.gamma
        for t in range(1, L):
            trend[:, t] = g * trend[:, t-1] + (1 - g) * x[:, t]
        return x - trend, trend


class AdaRevIN(nn.Module):
    """
    自适应 RevIN：
    - 输入侧：x_in = α*x_norm + (1-α)*x_raw
    - 输出侧：y_out = β*inv_norm(y_hat) + (1-β)*y_hat
    """
    def __init__(self, nvars, eps=1e-5, affine=True,
                 learnable_alpha='per_channel',   # 'none' | 'scalar' | 'per_channel' | 'mlp'
                 learnable_beta='scalar',
                 use_ema_detrend=False, ema_gamma=0.99):
        super().__init__()
        self.nvars = nvars
        self.eps = eps
        self.affine = affine
        self.use_ema = use_ema_detrend
        if self.use_ema:
            self.ema = EMATrend(ema_gamma)

        # α 门控
        if learnable_alpha == 'scalar':
            self.alpha = nn.Parameter(torch.tensor(0.7))
            self.alpha_proj = None
        elif learnable_alpha == 'per_channel':
            self.alpha = nn.Parameter(torch.full((nvars,), 0.7))
            self.alpha_proj = None
        elif learnable_alpha == 'mlp':
            self.alpha = None
            self.alpha_proj = nn.Sequential(
                nn.Linear(4*nvars, 256), nn.ReLU(),
                nn.Linear(256, nvars)
            )
        else:
            self.alpha = None
            self.alpha_proj = None

        # β 门控
        if learnable_beta == 'scalar':
            self.beta = nn.Parameter(torch.tensor(1.0))
        elif learnable_beta == 'per_channel':
            self.beta = nn.Parameter(torch.ones(nvars))
        else:
            self.beta = None

        # 可选仿 RevIN 的仿射参数
        if self.affine:
            self.affine_scale = nn.Parameter(torch.ones(nvars))
            self.affine_bias  = nn.Parameter(torch.zeros(nvars))

        self._cache = {}

    def _moments(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        return mean, std

    def normalize(self, x):
        if self.use_ema:
            x_detr, trend = self.ema(x)
        else:
            x_detr, trend = x, None
        mean, std = self._moments(x_detr)
        x_norm = (x_detr - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_scale + self.affine_bias
        return x_norm, (mean, std, trend)

    def denormalize(self, y, stats):
        mean, std, trend = stats
        if self.affine:
            y = (y - self.affine_bias) / (self.affine_scale + self.eps)
        y = y * std + mean
        if trend is not None:
            y = y + trend[:, -y.shape[1]:]
        return y

    def forward_in(self, x):
        x_norm, stats = self.normalize(x)
        B, L, C = x.shape

        if self.alpha_proj is not None:
            mean, std, trend = stats
            slope = (x[:, -1] - x[:, 0]) / (x.shape[1]-1)
            feats = torch.cat([
                mean.squeeze(1), std.squeeze(1),
                x.amax(dim=1), x.amin(dim=1), slope
            ], dim=-1)  # [B, 5C]
            a = torch.sigmoid(self.alpha_proj(feats)).unsqueeze(1)  # [B,1,C]
        elif self.alpha is not None:
            a = torch.sigmoid(self.alpha).view(1,1,C)
        else:
            a = torch.ones(1,1,C, device=x.device)

        x_in = a * x_norm + (1 - a) * x
        self._cache['stats'] = stats
        self._cache['alpha_used'] = a
        return x_in, stats, a

    def forward_out(self, y):
        stats = self._cache.get('stats', None)
        if stats is None:
            return y
        y_inv = self.denormalize(y, stats)
        if self.beta is None:
            return y_inv
        b = torch.sigmoid(self.beta)
        if b.ndim == 0:
            return b * y_inv + (1 - b) * y
        else:
            b = b.view(1,1,-1)
            return b * y_inv + (1 - b) * y
