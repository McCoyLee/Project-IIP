# layers/adanorm.py
import torch
import torch.nn as nn

class AdaNorm(nn.Module):
    """
    轻量 AdaNorm：沿 B,L 统计，推理期用 EMA；alpha/beta 支持 scalar/per_channel。
    期望输入 [B, L, D]，可与 LayerNorm 互换。
    """
    def __init__(self, d_model, alpha_mode="per_channel", beta_mode="scalar", ema_gamma=0.995, eps=1e-5):
        super().__init__()
        assert alpha_mode in ("per_channel", "scalar")
        assert beta_mode  in ("per_channel", "scalar")
        self.eps = eps
        self.ema_gamma = float(ema_gamma)

        def mkparam(mode, init):
            if mode == "per_channel":
                return nn.Parameter(torch.full((d_model,), init))
            return nn.Parameter(torch.tensor(init, dtype=torch.float32))

        self.alpha = mkparam(alpha_mode, 1.0)
        self.beta  = mkparam(beta_mode, 0.0)

        self.register_buffer("ema_mean", torch.zeros(1, 1, d_model))
        self.register_buffer("ema_var",  torch.ones(1, 1, d_model))

    def forward(self, x):
        # x: [B, L, D]
        if self.training:
            m = x.mean(dim=(0, 1), keepdim=True)
            v = x.var(dim=(0, 1), unbiased=False, keepdim=True)
            with torch.no_grad():
                self.ema_mean.mul_(self.ema_gamma).add_(m * (1 - self.ema_gamma))
                self.ema_var.mul_(self.ema_gamma).add_(v * (1 - self.ema_gamma))
        else:
            m, v = self.ema_mean, self.ema_var

        xhat = (x - m) / (v + self.eps).sqrt()

        def view(p):
            return p if p.dim() == 0 else p.view(1, 1, -1)

        return view(self.alpha) * xhat + view(self.beta)
