# utils/adaptive_norm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ======================================================================
# TokenAdaptiveNorm (TAN) v2 — 本论文补充贡献
# ======================================================================
class TokenAdaptiveNorm(nn.Module):
    """
    Token 级自适应归一化 v2（残差门控 + 统计量 detach）。

    设计原则：
      1. **残差恒等**：gate=0 时 TAN 退化为恒等变换，不伤害 baseline。
      2. **统计量 detach**：per-patch mean/std 不参与反传，避免 gradient-through-stats
         导致的长训练后 NaN（v1 在 epoch 42 因此崩溃）。
      3. **无 affine**：LayerNorm 在 backbone 每层已有 γ/β，TAN 再加 affine 冗余且
         引入可自由漂移的 scale 参数（v1 的 affine_scale→0 是又一个 NaN 源）。
      4. **简洁反归一化**：无除法，不需要 "safe_scale" 之类的数值护栏。

    数学形式：
      forward_in:
        x_local_norm_i = (x_i − μ_i) / σ_i            (detached μ, σ)
        g_i = sigmoid(gate(freq_i))                     (频率条件门控)
        x_out_i = (1 − g_i) · x_i + g_i · x_local_norm_i

      forward_out:
        y_out_j = (1 − g_last) · y_j + g_last · (y_j · σ_last + μ_last)

      g→0 时两端均为恒等。
    """

    def __init__(
            self,
            n_vars: int,
            patch_len: int,
            stride: int,
            freq_dim: int = 0,
            use_freq_cond: bool = True,
            std_floor: float = 1e-2,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.freq_dim = int(freq_dim)
        self.std_floor = float(std_floor)
        self.use_freq_cond = bool(use_freq_cond) and self.freq_dim > 0

        if self.use_freq_cond:
            self.gate = nn.Linear(self.freq_dim, 1)
            # 零初始化 → sigmoid(0)=0.5；设偏置为 −2 → sigmoid(−2)≈0.12，
            # 训练初期 TAN 几乎恒等，让 baseline 收敛后再逐步打开
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)
        else:
            self.gate_param = nn.Parameter(torch.tensor(-2.0))

        self._cache = {}

    # -----------------------------------------------------------------
    @staticmethod
    def _local_stats(x: torch.Tensor, patch_len: int, stride: int,
                     n_tokens: int, std_floor: float):
        """Per-patch mean/std，全程 float32，输出 detach。"""
        B, L, C = x.shape
        P, S = int(patch_len), int(stride)
        x_bc = x.detach().float().permute(0, 2, 1).contiguous()      # [B,C,L]
        if x_bc.shape[-1] < P:
            x_bc = F.pad(x_bc, (0, P - x_bc.shape[-1]), mode='replicate')
        patches = x_bc.unfold(dimension=-1, size=P, step=S)          # [B,C,N',P]
        Np = patches.shape[2]
        if Np < n_tokens:
            last = patches[:, :, -1:, :].expand(-1, -1, n_tokens - Np, -1)
            patches = torch.cat([patches, last], dim=2)
        elif Np > n_tokens:
            patches = patches[:, :, :n_tokens, :]
        mu = patches.mean(dim=-1)                                     # [B,C,N]
        var = patches.var(dim=-1, unbiased=False)
        std = torch.sqrt(var + std_floor * std_floor)                 # [B,C,N]
        return mu, std                                                # already detached

    # -----------------------------------------------------------------
    def forward_in(self, x: torch.Tensor, freq_features: torch.Tensor,
                   n_tokens: int):
        """
        x:             [B, L, C]
        freq_features: [B*C, N, K] or None
        n_tokens:      N
        Returns:       (x_out [B,L,C], ctx dict)
        """
        B, L, C = x.shape
        N = int(n_tokens)

        # --- detached per-patch statistics ---
        mu_l, std_l = self._local_stats(
            x, self.patch_len, self.stride, N, self.std_floor)        # [B,C,N]

        # --- 映射 per-patch → per-timestep（nearest-token）---
        device = x.device
        token_centers = (torch.arange(N, device=device).float()
                         * self.stride + self.patch_len / 2.0)
        time_idx = torch.arange(L, device=device).float()
        tok_per_time = (time_idx.unsqueeze(1) - token_centers.unsqueeze(0)
                        ).abs().argmin(dim=1)                         # [L]

        mu_t = mu_l[:, :, tok_per_time].permute(0, 2, 1)             # [B,L,C] detached
        std_t = std_l[:, :, tok_per_time].permute(0, 2, 1)           # [B,L,C] detached

        # --- 频率条件门控 ---
        _has_freq = self.use_freq_cond and freq_features is not None
        if _has_freq:
            # [B*C,N,K] → [B,C,N,K]
            ff = freq_features.view(B, C, N, -1)
            g_tok = torch.sigmoid(self.gate(ff)).squeeze(-1)          # [B,C,N]
            g_t = g_tok[:, :, tok_per_time].permute(0, 2, 1)         # [B,L,C]
            g_last = g_tok[:, :, -1]                                  # [B,C]
        else:
            g_val = torch.sigmoid(self.gate_param)
            g_t = g_val.view(1, 1, 1).expand(B, L, C)
            g_last = g_val.view(1, 1).expand(B, C)

        # --- 残差归一化 ---
        x_f = x.float()
        x_local_norm = (x_f - mu_t) / std_t                          # detach stats → safe
        g32 = g_t.float()
        x_out = (1.0 - g32) * x_f + g32 * x_local_norm
        x_out = x_out.to(x.dtype)

        # --- 存储反归一化所需信息 ---
        mu_last = mu_l[:, :, -1]                                      # [B,C]
        std_last = std_l[:, :, -1]                                    # [B,C]
        ctx = {
            "mu_last": mu_last.detach(),    # [B,C]
            "std_last": std_last.detach(),  # [B,C]
            "g_last": g_last.detach(),      # [B,C]
        }
        self._cache = ctx
        return x_out, ctx

    # -----------------------------------------------------------------
    def forward_out(self, y: torch.Tensor, ctx: dict = None,
                    freq_features_last: torch.Tensor = None):
        """
        y:    [B, H, C]  prediction in normalized space
        ctx:  from forward_in
        freq_features_last: [B, C, K] or None
        Returns: [B, H, C]
        """
        if ctx is None:
            ctx = self._cache
        if not ctx:
            return y

        out_dtype = y.dtype
        y_f = y.float()

        mu = ctx["mu_last"].unsqueeze(1).float()                      # [B,1,C]
        std = ctx["std_last"].unsqueeze(1).float()                    # [B,1,C]

        # gate for denormalization
        if self.use_freq_cond and freq_features_last is not None:
            g = torch.sigmoid(self.gate(freq_features_last))          # [B,C,1]
            g = g.squeeze(-1).unsqueeze(1)                            # [B,1,C]
        else:
            g = ctx["g_last"].unsqueeze(1).float()                    # [B,1,C]

        # 反归一化：g→0 恒等，g→1 完全 local denorm
        y_denorm = (1.0 - g) * y_f + g * (y_f * std + mu)

        return y_denorm.to(out_dtype)
