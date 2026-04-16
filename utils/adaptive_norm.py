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
# TokenAdaptiveNorm (TAN) — 本论文补充贡献
# ======================================================================
class TokenAdaptiveNorm(nn.Module):
    """
    Token 级自适应归一化。

    核心思想：
      - 对每个 patch (stride/length 对齐到主干 tokenization) 计算局部统计量 (μ_i, σ_i)
      - 同时保留全局统计量 (μ_g, σ_g)
      - 根据局部频率特征 f_i ∈ R^K，学习混合系数 α_i ∈ [0,1]：
            x_norm_i = α_i · LocalNorm(x_i) + (1-α_i) · GlobalNorm(x_i)
      - 反归一化阶段类似：β_j 控制局部/全局反归一化混合

    使用方式：
      tan = TokenAdaptiveNorm(n_vars=C, patch_len=P, stride=S, freq_dim=K)
      x_norm, ctx = tan.forward_in(x_raw, freq_features)   # freq_features: [B*C,N,K]
      ...
      y_denorm = tan.forward_out(y_pred, ctx, freq_features_last)

    当 freq_features 为 None 时退化为纯 per-token 归一化（固定 α=0.7）。
    """

    def __init__(
            self,
            n_vars: int,
            patch_len: int,
            stride: int,
            freq_dim: int = 0,
            eps: float = 1e-5,
            use_freq_cond: bool = True,
            alpha_init_bias: float = 0.0,    # sigmoid(0) = 0.5
            affine: bool = True,
            std_floor: float = 1e-2,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.freq_dim = int(freq_dim)
        self.eps = eps
        # std_floor：防止 fp16 下 1/std 溢出（数据已 StandardScaled 时，std~1，0.01 是合理下限）
        self.std_floor = float(std_floor)
        self.use_freq_cond = bool(use_freq_cond) and self.freq_dim > 0
        self.affine = affine

        if self.use_freq_cond:
            # 频率→归一化强度的条件门控 MLP
            hidden = max(16, self.freq_dim * 2)
            self.alpha_mlp = nn.Sequential(
                nn.Linear(self.freq_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.beta_mlp = nn.Sequential(
                nn.Linear(self.freq_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # 初始化：输出层零初始化 + 偏置 → 初始 α ≈ sigmoid(alpha_init_bias)
            nn.init.zeros_(self.alpha_mlp[-1].weight)
            nn.init.constant_(self.alpha_mlp[-1].bias, float(alpha_init_bias))
            nn.init.zeros_(self.beta_mlp[-1].weight)
            nn.init.constant_(self.beta_mlp[-1].bias, float(alpha_init_bias))
        else:
            self.alpha_mlp = None
            self.beta_mlp = None
            # 无频率条件：使用固定标量 α
            self.alpha = nn.Parameter(torch.tensor(0.7))
            self.beta = nn.Parameter(torch.tensor(0.7))

        if self.affine:
            self.affine_scale = nn.Parameter(torch.ones(n_vars))
            self.affine_bias = nn.Parameter(torch.zeros(n_vars))

        self._cache = {}

    @staticmethod
    def _global_stats(x: torch.Tensor, std_floor: float):
        # x: [B, L, C]；全程 float32；clamp 放在 sqrt 内部避免 autograd sqrt(0) → NaN
        x_f = x.float()
        mean = x_f.mean(dim=1, keepdim=True)                              # [B,1,C]
        var = x_f.var(dim=1, unbiased=False, keepdim=True)
        std = torch.sqrt(var.clamp_min(std_floor * std_floor))            # [B,1,C]
        return mean, std

    @staticmethod
    def _local_stats(x: torch.Tensor, patch_len: int, stride: int, n_tokens: int, std_floor: float):
        """
        x: [B, L, C]
        Returns:
            mu_local  : [B, C, N]
            std_local : [B, C, N]
        全程 float32；clamp 放在 sqrt 内部：sqrt(var.clamp_min(floor^2)) 而非
        sqrt(var).clamp_min(floor)，后者在 var=0 时 autograd 产生 0/0=NaN。
        """
        B, L, C = x.shape
        P, S = int(patch_len), int(stride)
        x_bc = x.float().permute(0, 2, 1).contiguous()                  # [B, C, L]
        if x_bc.shape[-1] < P:
            x_bc = F.pad(x_bc, (0, P - x_bc.shape[-1]), mode='replicate')
        patches = x_bc.unfold(dimension=-1, size=P, step=S)             # [B, C, N', P]
        Np = patches.shape[2]
        if Np < n_tokens:
            last = patches[:, :, -1:, :].expand(-1, -1, n_tokens - Np, -1)
            patches = torch.cat([patches, last], dim=2)
        elif Np > n_tokens:
            patches = patches[:, :, :n_tokens, :]
        mu = patches.mean(dim=-1)                                       # [B,C,N]
        var = patches.var(dim=-1, unbiased=False)
        # 关键：clamp 在 sqrt 内，避免 sqrt(0).backward() → grad=0/(2*0)=NaN
        std = torch.sqrt(var.clamp_min(std_floor * std_floor))           # [B,C,N]
        return mu, std

    def forward_in(self, x: torch.Tensor, freq_features: torch.Tensor, n_tokens: int):
        """
        Input:
            x              : [B, L, C]
            freq_features  : [B*C, N, K] or None
            n_tokens       : N (must match patch tokenization of main model)
        Output:
            x_norm         : [B, L, C] — token-level adaptive normalized input
            ctx            : dict holding global/local stats for denormalization
        """
        B, L, C = x.shape
        N = int(n_tokens)

        # Global stats（float32，下限 std_floor）
        mu_g, std_g = self._global_stats(x, self.std_floor)             # [B,1,C]

        # Local per-patch stats（float32，下限 std_floor）
        mu_l, std_l = self._local_stats(x, self.patch_len, self.stride, N, self.std_floor)  # [B,C,N]

        # 将 per-patch 统计扩展回 per-timestep 统计，沿 [B, L, C]
        # 简化：对 token i 的局部 stat 广播到该 token 覆盖的时间段（优先对每个时间步用"最近 token"的统计）
        # 为了实现高效，我们构建 per-timestep 的索引：time_to_token[t] → token_idx
        device = x.device
        time_idx = torch.arange(L, device=device)                      # [L]
        # 对重叠 patch，使用 patch 中心距离最近的 token
        token_centers = (torch.arange(N, device=device).float() * self.stride + self.patch_len / 2.0)
        # 每个时间步 → 最近的 token
        dist = (time_idx.float().unsqueeze(1) - token_centers.unsqueeze(0)).abs()  # [L, N]
        tok_per_time = dist.argmin(dim=1)                              # [L]

        # mu_l: [B, C, N] → [B, C, L] via index
        mu_l_t = mu_l[:, :, tok_per_time]                              # [B, C, L]
        std_l_t = std_l[:, :, tok_per_time]                            # [B, C, L]
        mu_l_t = mu_l_t.permute(0, 2, 1).contiguous()                  # [B, L, C]
        std_l_t = std_l_t.permute(0, 2, 1).contiguous()                # [B, L, C]

        # Compute mixing coefficient α per (batch, token, channel)
        if self.use_freq_cond and freq_features is not None:
            # freq_features: [B*C, N, K] → [B, C, N, K]
            ff = freq_features.view(B, C, N, -1)
            alpha_tok = torch.sigmoid(self.alpha_mlp(ff)).squeeze(-1)  # [B, C, N]
            beta_tok = torch.sigmoid(self.beta_mlp(ff)).squeeze(-1)    # [B, C, N]
            # Broadcast to per-timestep via same nearest-token mapping
            alpha_t = alpha_tok[:, :, tok_per_time].permute(0, 2, 1).contiguous()  # [B, L, C]
            beta_t = beta_tok[:, :, tok_per_time].permute(0, 2, 1).contiguous()
        else:
            alpha_t = torch.sigmoid(self.alpha).expand(B, L, C)
            beta_t = torch.sigmoid(self.beta).expand(B, L, C)

        # 混合归一化（在 float32 下做除法以避免 fp16 溢出）
        x_f = x.float()
        x_norm_local = (x_f - mu_l_t.float()) / std_l_t.float().clamp_min(self.std_floor)
        x_norm_global = (x_f - mu_g.float()) / std_g.float().clamp_min(self.std_floor)
        # 防御性裁剪：极端样本 (如常数段) 可能仍接近下限，直接截断到合理范围
        x_norm_local = x_norm_local.clamp(-30.0, 30.0)
        x_norm_global = x_norm_global.clamp(-30.0, 30.0)
        a32 = alpha_t.float()
        x_norm = a32 * x_norm_local + (1.0 - a32) * x_norm_global
        x_norm = x_norm.to(x.dtype)

        if self.affine:
            x_norm = x_norm * self.affine_scale.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)

        ctx = {
            "mu_g": mu_g, "std_g": std_g,
            "mu_l": mu_l, "std_l": std_l,
            "tok_per_time": tok_per_time,
            "alpha_t": alpha_t.detach(),
            "beta_t": beta_t.detach(),
            # 反归一化用的混合系数由输入端决定，方便训练阶段梯度稳定
            "_use_freq_cond": self.use_freq_cond,
        }
        self._cache = ctx
        return x_norm, ctx

    def forward_out(self, y: torch.Tensor, ctx: dict = None, freq_features_last: torch.Tensor = None):
        """
        Input:
            y                   : [B, H, C] prediction in normalized space
            ctx                 : stats dict from forward_in
            freq_features_last  : optional [B, C, K] — 最后几个输入 token 的平均频率特征，
                                  用于外推预测阶段的归一化策略；若为 None 则沿用输入端的 β
        Returns:
            y_denorm            : [B, H, C]
        """
        if ctx is None:
            ctx = self._cache
        if ctx is None or "mu_g" not in ctx:
            return y

        B, H, C = y.shape
        mu_g = ctx["mu_g"]          # [B, 1, C]
        std_g = ctx["std_g"]        # [B, 1, C]
        mu_l = ctx["mu_l"]          # [B, C, N]
        std_l = ctx["std_l"]        # [B, C, N]

        out_dtype = y.dtype
        y_f = y.float()
        if self.affine:
            scale = self.affine_scale.view(1, 1, -1).float()
            # 若 |scale| 过小则替换为 std_floor，避免 fp16 下 1/scale 溢出
            safe_scale = torch.where(
                scale.abs() < self.std_floor,
                torch.full_like(scale, self.std_floor),
                scale,
            )
            y_f = (y_f - self.affine_bias.view(1, 1, -1).float()) / safe_scale

        # 使用输入序列最后一个 token 的统计作为预测阶段的局部统计量
        mu_l_last = mu_l[:, :, -1].unsqueeze(1).float()                 # [B, 1, C]
        std_l_last = std_l[:, :, -1].unsqueeze(1).float().clamp_min(self.std_floor)  # [B, 1, C]

        # 自适应反归一化混合系数
        if self.use_freq_cond and freq_features_last is not None:
            # freq_features_last: [B, C, K]
            beta = torch.sigmoid(self.beta_mlp(freq_features_last)).squeeze(-1)  # [B, C]
            beta = beta.unsqueeze(1)                                    # [B, 1, C]
        elif self.use_freq_cond:
            # 退化：使用 ctx 中的平均 β
            beta = ctx["beta_t"].mean(dim=1, keepdim=True)              # [B, 1, C]
        else:
            beta = torch.sigmoid(self.beta).expand(B, 1, C)

        beta = beta.float()
        y_local = y_f * std_l_last + mu_l_last
        y_global = y_f * std_g.float() + mu_g.float()
        y_out = beta * y_local + (1.0 - beta) * y_global
        return y_out.to(out_dtype)
