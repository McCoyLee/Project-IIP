# layers/unica_cov.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniCAFiLM(nn.Module):
    """
    协变量同质化 + FiLM/残差 融合 + 稳定门控
    作用对象：Timer-XL 的 token 嵌入 [B, C*N, D]
    协变量：原始多变量 x 与时间标记 x_mark（可选）
    关键：恒等起步、弱门控、全局混合系数、沿 token 的轻量平滑
    """
    def __init__(
            self,
            d_model: int,
            bottleneck: int = 128,
            input_token_len: int = 24,
            exclude_target: bool = False,
            fusion: str = "film_gate",          # "film_gate" 或 "res_add"
            gamma_scale: float = 0.1,           # γ 扰动幅度
            beta_scale: float  = 0.05,          # β 位移幅度（film_gate）
            res_scale: float   = 0.1,           # Δ 残差幅度（res_add）
            dropout: float     = 0.0,           # 条件分支正则
            init_gate_bias: float  = -2.0,      # gate 初始小
            init_alpha_bias: float = -2.0,      # α 初始小 (sigmoid 后 ~0.12)
            smooth_gate_ks: int = 3,            # gate 的 token 平滑核(奇数)
            smooth_beta_ks: int = 3             # β 的 token 平滑核(奇数)
    ):
        super().__init__()
        assert fusion in ["film_gate", "res_add"]
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.input_token_len = input_token_len
        self.exclude_target = exclude_target
        self.fusion = fusion
        self.gamma_scale = float(gamma_scale)
        self.beta_scale  = float(beta_scale)
        self.res_scale   = float(res_scale)
        self.smooth_gate_ks = int(smooth_gate_ks)
        self.smooth_beta_ks = int(smooth_beta_ks)

        # 条件分支
        self.proj_in  = nn.LazyLinear(bottleneck)   # [*, F] -> [*, B]
        self.ln       = nn.LayerNorm(bottleneck)
        self.drop     = nn.Dropout(dropout)

        if fusion == "film_gate":
            self.proj_out = nn.Linear(bottleneck, 2 * d_model)  # -> [γ_pred, β_pred]
            self.gate     = nn.Linear(bottleneck, 1)
        else:
            self.proj_out = nn.Linear(bottleneck, d_model)       # -> Δ

        # 全局混合系数（通过 sigmoid 到 0~1）
        self.alpha = nn.Parameter(torch.tensor(init_alpha_bias))

        # === 关键初始化：恒等起步 & 弱门控 ===
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        if fusion == "film_gate":
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, init_gate_bias)

    @torch.no_grad()
    def _window_pool(self, x: torch.Tensor, x_mark: torch.Tensor, target_channel: int):
        """
        将 x/x_mark 对齐到 token 级：每变量每 token 给一个摘要
        x: [B,L,C], x_mark: [B,L,M] or None
        return: [B, C*N, F_cov]
        """
        B, L, C = x.shape
        P = self.input_token_len
        assert L % P == 0, f"L={L} must be multiple of input_token_len={P}"
        N = L // P

        # 每变量每 token 的均值/标准差
        x_var = x.permute(0, 2, 1).unfold(dimension=-1, size=P, step=P)     # [B,C,N,P]
        mu  = x_var.mean(dim=-1, keepdim=False)                              # [B,C,N]
        std = x_var.var(dim=-1, keepdim=False, unbiased=False).add(1e-6).sqrt()
        stat = torch.stack([mu, std], dim=-1).view(B, C * N, 2)              # [B,C*N,2]

        # 可选：排除目标通道
        if self.exclude_target and (0 <= target_channel < C):
            stat_ = stat.view(B, C, N, 2)
            others = torch.cat([stat_[:, :target_channel], stat_[:, target_channel+1:]], dim=1)  # [B,C-1,N,2]
            repl = others.mean(dim=1, keepdim=True).expand(-1, 1, -1, -1)                        # [B,1,N,2]
            stat_[:, target_channel:target_channel+1] = repl
            stat = stat_.view(B, C * N, 2)

        # 时间标记池化并拼接
        if x_mark is not None:
            xm = x_mark.unfold(dimension=1, size=P, step=P).mean(dim=2)      # [B,N,M]
            xm = xm.unsqueeze(1).expand(B, C, N, -1).contiguous().view(B, C * N, -1)
            cov_feat = torch.cat([stat, xm], dim=-1)                         # [B,C*N,2+M]
        else:
            cov_feat = stat                                                  # [B,C*N,2]
        return cov_feat

    def _smooth_1d(self, t: torch.Tensor, k: int) -> torch.Tensor:
        """对 [B*C, D, N] 或 [B*C, 1, N] 做 1D 均值平滑（k 为奇数）"""
        if k <= 1: return t
        pad = k // 2
        return F.avg_pool1d(t, kernel_size=k, stride=1, padding=pad)

    def forward(self, embed_tokens: torch.Tensor, x: torch.Tensor, x_mark: torch.Tensor,
                target_channel: int):
        """
        embed_tokens : [B, C*N, D]  —— 主分支 token 嵌入（Timer-XL 的 Linear 后）
        x            : [B, L, C]    —— 建议传“未归一化”的原始序列
        x_mark       : [B, L, M] 或 None
        return       : [B, C*N, D]  —— 融合后的 token 嵌入
        """
        B, CN, D = embed_tokens.shape
        Bx, L, C = x.shape
        P = self.input_token_len
        N = L // P
        assert CN == C * N, f"shape mismatch: CN({CN}) != C*N({C}*{N})"

        cov_feat = self._window_pool(x, x_mark, target_channel)              # [B, C*N, F]
        h = self.proj_in(cov_feat)                                           # [B, C*N, Bk]
        h = self.ln(F.gelu(h))
        h = self.drop(h)

        mix = torch.sigmoid(self.alpha)                                       # 全局混合系数 ∈ (0,1)

        if self.fusion == "film_gate":
            film = self.proj_out(h)                                           # [B, C*N, 2D]
            gamma_pred, beta = film.chunk(2, dim=-1)                          # [B,C*N,D], [B,C*N,D]
            gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_pred)
            beta  = self.beta_scale * torch.tanh(beta)                        # 限制位移幅度

            # --- 沿 token 维平滑 beta ---
            if self.smooth_beta_ks > 1:
                beta_bcnd = beta.view(B * C, N, D).transpose(1, 2)            # [B*C, D, N]
                beta_bcnd = self._smooth_1d(beta_bcnd, self.smooth_beta_ks)   # [B*C, D, N]
                beta = beta_bcnd.transpose(1, 2).contiguous().view(B, C * N, D)

            y = embed_tokens * gamma + beta

            gate = torch.sigmoid(self.gate(h))                                # [B, C*N, 1]
            # --- 沿 token 维平滑 gate ---
            if self.smooth_gate_ks > 1:
                g = gate.view(B * C, N, 1).transpose(1, 2)                    # [B*C,1,N]
                g = self._smooth_1d(g, self.smooth_gate_ks)                   # [B*C,1,N]
                gate = g.transpose(1, 2).contiguous().view(B, C * N, 1)

            out = embed_tokens + (mix * gate) * (y - embed_tokens)            # 残差门控 + 全局混合
            return out
        else:  # "res_add"
            delta = self.proj_out(h)                                          # [B, C*N, D]
            out = embed_tokens + mix * (self.res_scale * delta)
            return out
