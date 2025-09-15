# layers/patch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed1D(nn.Module):
    """
    1D Patch 切片 + 线性投影到 d_model
    输入: x [B, L, C]
    输出: tokens [B*C, N, d_model], meta=(B, C, N)
    """
    def __init__(self, patch_len: int = 24, stride: int = 24,
                 d_model: int = 256, add_pos: bool = True, pos_learnable: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.add_pos = add_pos
        self.pos_learnable = pos_learnable
        self.register_buffer("_pos_cache", None, persistent=False)

    def _pos(self, N: int, d_model: int, device):
        if not self.add_pos:
            return None
        if self.pos_learnable:
            # 每个 forward 的 N 可能不同，按需生成
            return nn.Parameter(torch.zeros(1, N, d_model, device=device))
        # 正弦位置编码（无需参数）
        pos = torch.arange(N, device=device).float()[:, None]
        dim = torch.arange(d_model, device=device).float()[None, :]
        div = torch.exp((torch.arange(0, d_model, 2, device=device).float()) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(N, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, N, d_model]

    def forward(self, x: torch.Tensor):
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2).contiguous()
        B, C, L = x.shape
        # unfold 1D: [B, C, N, P]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape
        patches = patches.contiguous().view(B * C, N, P)
        tokens = self.proj(patches)  # [B*C, N, d_model]
        pos = self._pos(N, tokens.size(-1), tokens.device)
        if self.add_pos and pos is not None:
            if isinstance(pos, nn.Parameter) and pos.shape[1] != N:
                # 重新生成匹配 N 的参数
                pos = nn.Parameter(torch.zeros(1, N, tokens.size(-1), device=tokens.device))
            tokens = tokens + pos
        return tokens, (B, C, N)

class PatchUnembedHead(nn.Module):
    """
    将 [B*C, N, d_model] 聚合为预测 [B, H, 1] 或 [B, H, C]
    简单做法：Flatten N 维后线性映射到 H；也可做 mean pool。
    """
    def __init__(self, d_model: int, N_max: int, pred_len: int, out_channels: int = 1,
                 head_type: str = "flatten"):  # "flatten" | "meanpool"
        super().__init__()
        self.head_type = head_type
        self.pred_len = pred_len
        self.out_channels = out_channels
        if head_type == "flatten":
            self.head = nn.Linear(d_model * N_max, pred_len * out_channels)
        else:
            self.head = nn.Linear(d_model, pred_len * out_channels)

    def forward(self, h: torch.Tensor, meta):
        # h: [B*C, N, d_model]
        B, C, N = meta
        if self.head_type == "flatten":
            # 若实际 N < N_max，可在构造时把 N_max 设为训练/评测使用的固定 N（seq_len 固定时常见）
            out = self.head(h.reshape(B * C, -1))
        else:
            out = self.head(h.mean(dim=1))  # mean over N
        out = out.view(B, C, self.pred_len, self.out_channels)  # [B, C, H, out_ch]
        return out  # 后面按 target_channel 取用
