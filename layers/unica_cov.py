# layers/unica_cov.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniCAFiLM(nn.Module):
    """
    协变量同质化 + FiLM 融合 + 残差门控
    作用对象：Timer-XL 的 token 嵌入 [B, C*N, D]
    协变量：原始多变量 x 与时间标记 x_mark（可选）
    """
    def __init__(self, d_model: int, bottleneck: int = 128, input_token_len: int = 24,
                 exclude_target: bool = False, fusion: str = "film_gate"):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.input_token_len = input_token_len
        self.exclude_target = exclude_target
        assert fusion in ["film_gate", "res_add"]
        self.fusion = fusion

        # LazyLinear 让我们不必提前知道协变量拼接后的 in_dim
        self.proj_in  = nn.LazyLinear(bottleneck)      # [*, F] -> [*, B]
        self.ln       = nn.LayerNorm(bottleneck)
        self.proj_out = nn.Linear(bottleneck, 2*d_model) if fusion == "film_gate" \
            else nn.Linear(bottleneck, d_model)
        # 残差门控（只在 film_gate 用到）
        self.gate = nn.Linear(bottleneck, 1)

    @torch.no_grad()
    def _window_pool(self, x: torch.Tensor, x_mark: torch.Tensor,
                     target_channel: int, use_normed_x: bool = True):
        """
        把协变量（多变量序列 + 时间标记）对齐到 token 级别，并做轻量池化
        输入:
            x       : [B, L, C]
            x_mark  : [B, L, M] 或 None
        返回:
            cov_feat: [B, C*N, F_cov]  (与 [B, C*N, D] 对齐)
        """
        B, L, C = x.shape
        P = self.input_token_len
        assert L % P == 0, f"L={L} must be multiple of input_token_len={P}"
        N = L // P

        # 1) 多变量窗口池化（每个变量、每个 token 一个摘要）
        #    x -> [B, C, L] -> unfold: [B, C, N, P]
        x_var = x.permute(0, 2, 1).unfold(dimension=-1, size=P, step=P)  # [B,C,N,P]
        # 统计量：均值 & 标准差
        mu  = x_var.mean(dim=-1, keepdim=False)                           # [B,C,N]
        std = x_var.std(dim=-1, keepdim=False).clamp_min(1e-6)            # [B,C,N]
        stat = torch.stack([mu, std], dim=-1)                              # [B,C,N,2]
        stat = stat.view(B, C*N, 2)                                        # [B, C*N, 2]

        # 可选：排除 target 通道（只用“其它协变量”的统计）
        if self.exclude_target and (0 <= target_channel < C):
            # 用所有通道的均值替代目标通道的统计，避免“泄漏式 shortcut”
            # 这里为了简单，直接把该通道的统计替换为其它通道均值
            stat_ = stat.view(B, C, N, 2)
            others = torch.cat([stat_[:, :target_channel], stat_[:, target_channel+1:]], dim=1)  # [B, C-1, N, 2]
            repl = others.mean(dim=1, keepdim=True).expand(-1, 1, -1, -1)                        # [B,1,N,2]
            stat_[:, target_channel:target_channel+1] = repl
            stat = stat_.view(B, C*N, 2)

        # 2) 时间标记窗口池化：x_mark -> [B,N,P,M] -> [B,N,M] -> Repeat 到每个通道
        if x_mark is not None:
            xm = x_mark.unfold(dimension=1, size=P, step=P)                # [B,N,P,M]
            xm = xm.mean(dim=2)                                            # [B,N,M]
            xm = xm.unsqueeze(1).expand(B, C, N, -1).contiguous()          # [B,C,N,M]
            xm = xm.view(B, C*N, xm.shape[-1])                              # [B,C*N,M]
            cov_feat = torch.cat([stat, xm], dim=-1)                        # [B,C*N, 2+M]
        else:
            cov_feat = stat                                                 # [B,C*N, 2]
        return cov_feat  # [B, C*N, F_cov]

    def forward(self, embed_tokens: torch.Tensor, x: torch.Tensor, x_mark: torch.Tensor,
                target_channel: int):
        """
        embed_tokens : [B, C*N, D]  — 主分支 token 嵌入（Timer-XL 的 Linear 后）
        x            : [B, L, C]
        x_mark       : [B, L, M] 或 None
        return       : [B, C*N, D] 融合后的 token 嵌入
        """
        B, CN, D = embed_tokens.shape
        cov_feat = self._window_pool(x, x_mark, target_channel)      # [B, C*N, F_cov]

        h = self.proj_in(cov_feat)                                   # [B, C*N, Bk]
        h = self.ln(F.gelu(h))                                       # [B, C*N, Bk]

        if self.fusion == "film_gate":
            film = self.proj_out(h)                                  # [B, C*N, 2D]
            gamma, beta = film.chunk(2, dim=-1)                      # [B,C*N,D], [B,C*N,D]
            # 让 gamma 近 1，避免破坏原预训练分布
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
            y = embed_tokens * gamma + beta                          # FiLM
            gate = torch.sigmoid(self.gate(h))                       # [B,C*N,1]
            out = embed_tokens + gate * (y - embed_tokens)           # 残差门控融合
            return out
        else:  # "res_add"：直接给主分支加一个小残差
            delta = self.proj_out(h)                                 # [B, C*N, D]
            return embed_tokens + 0.1 * delta
