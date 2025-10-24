# layers/patch.py
import math
import torch
import torch.nn as nn

class PatchEmbed1D(nn.Module):
    """
    PatchTST 风格的 1D patch + 线性投影
    输入:  x [B, L, C]
    输出:  tokens [B*C, N, d_model], meta=(B, C, N)
    说明:  使用 unfold 滑窗，支持 stride < patch_len（重叠窗口）
    """
    def __init__(
            self,
            patch_len: int = 24,
            stride: int = 24,
            d_model: int = 256,
            add_pos: bool = True,
            pos_learnable: bool = False,
            pos_max_len: int = 4096,   # 只在 learnable 位置编码时使用
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model, bias=True)

        self.add_pos = add_pos
        self.pos_learnable = pos_learnable
        if add_pos and pos_learnable:
            # 预分配一个最大长度的可学习位置编码，前 N 切片即可
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_max_len, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", None, persistent=False)

    @staticmethod
    def _pos_sincos(N: int, d_model: int, device):
        pe = torch.zeros(N, d_model, device=device)
        position = torch.arange(0, N, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1,N,d]

    def forward(self, x: torch.Tensor):
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2).contiguous()
        B, C, L = x.shape
        if L < self.patch_len:
            # 右侧 pad 到至少一个窗口
            pad = self.patch_len - L
            x = nn.functional.pad(x, (0, pad), mode="replicate")
            L = self.patch_len

        # unfold: [B, C, N, P]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape
        patches = patches.contiguous().view(B * C, N, P)

        tokens = self.proj(patches)  # [B*C, N, d_model]

        if self.add_pos:
            if self.pos_learnable and self.pos_embed is not None:
                if self.pos_embed.size(1) < N:
                    # 动态扩容（极少发生）
                    extra = N - self.pos_embed.size(1)
                    add = torch.zeros(1, extra, self.pos_embed.size(-1), device=tokens.device)
                    nn.init.trunc_normal_(add, std=0.02)
                    self.pos_embed = nn.Parameter(torch.cat([self.pos_embed, add], dim=1))
                pos = self.pos_embed[:, :N, :]
            else:
                pos = self._pos_sincos(N, tokens.size(-1), tokens.device)
            tokens = tokens + pos  # 广播到 [B*C,N,d]
        return tokens, (B, C, N)

class PatchUnembedHead(nn.Module):
    """
    将 [B*C, N, d] 聚合为预测:
      - head_type='last'  : 取最后 token -> Linear(d, H)
      - head_type='mean'  : 平均池化    -> Linear(d, H)
      - head_type='tokenwise': 每个 token -> Linear(d, P)，要求 stride==patch_len，可还原回 [B,L,C]
    """
    def __init__(
            self,
            d_model: int,
            pred_len: int,
            head_type: str = "last",    # 'last' | 'mean' | 'tokenwise'
            token_out_len: int = None,  # 当 head_type='tokenwise' 时需给定 = output_token_len
            stride_equals_patch: bool = True,
    ):
        super().__init__()
        self.head_type = head_type
        self.pred_len = pred_len
        self.token_out_len = token_out_len
        self.stride_equals_patch = stride_equals_patch

        if head_type in ("last", "mean"):
            self.head = nn.Linear(d_model, pred_len)
        elif head_type == "tokenwise":
            assert token_out_len is not None, "tokenwise 需要设置 token_out_len=output_token_len"
            assert stride_equals_patch, "tokenwise 仅支持 stride == patch_len（否则需要 overlap-add）"
            self.head = nn.Linear(d_model, token_out_len)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, h: torch.Tensor, meta):
        # h: [B*C, N, d]
        B, C, N = meta
        if self.head_type == "last":
            z = h[:, -1, :]                     # (B*C, d)
            out = self.head(z).view(B, C, self.pred_len)   # (B,C,H)
            return out                           # (B,C,H)
        elif self.head_type == "mean":
            z = h.mean(dim=1)                    # (B*C, d)
            out = self.head(z).view(B, C, self.pred_len)   # (B,C,H)
            return out
        else:  # tokenwise
            tok = self.head(h)                   # (B*C, N, P_out)
            tok = tok.view(B, C, N, self.token_out_len)    # (B,C,N,P)
            seq = tok.reshape(B, C, N * self.token_out_len)  # (B,C,L) 仅在 stride==patch 有效
            return seq.transpose(1, 2)           # (B,L,C)
