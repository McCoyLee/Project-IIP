# layers/de_stationary_attention.py
import torch
import torch.nn as nn

class DeStationaryAdapter(nn.Module):
    """
    反平稳注意力：由样本统计量生成每头缩放/偏置，调制注意力 logits。
    用法：
        logits = q @ k^T / sqrt(d) (+ bias)
        logits = adapter(logits, stats)
    """
    def __init__(self, nvars, nheads, hidden=128):
        super().__init__()
        self.nvars = nvars
        self.nheads = nheads
        self.mlp = nn.Sequential(
            nn.Linear(4*nvars, hidden), nn.ReLU(),
            nn.Linear(hidden, 2*nheads)  # 每头输出 (scale, bias)
        )

    @torch.no_grad()
    def _flatten_stats(self, stats):
        mean, std, trend = stats     # mean/std: [B,1,C]
        # 可以扩展更多统计量（分位数/斜率等），此处保持简洁
        zeros = torch.zeros_like(mean.squeeze(1))
        return torch.cat([mean.squeeze(1), std.squeeze(1), zeros, zeros], dim=-1)  # [B,4C]

    def forward(self, logits, stats):
        """
        logits: [B, H, Lq, Lk]
        stats: (mean,std,trend)
        """
        B, H, Lq, Lk = logits.shape
        feats = self._flatten_stats(stats)          # [B, 4C]
        ab = self.mlp(feats)                        # [B, 2H]
        a, b = ab.chunk(2, dim=-1)                  # [B,H], [B,H]
        a = torch.tanh(a).view(B, H, 1, 1)          # (-1,1)
        b = 0.5 * torch.tanh(b).view(B, H, 1, 1)    # 缩小偏置幅度
        return logits * (1.0 + a) + b
