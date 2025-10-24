# layers/afs_gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TriangularBank(nn.Module):
    """K个可学习三角带通滤波窗（定义在[0,1]归一化频率上）"""
    def __init__(self, K: int, init_centers=None, init_width=0.25):
        super().__init__()
        self.K = K
        if init_centers is None:
            init_centers = torch.linspace(0.05, 0.95, K)  # 均匀初始化
        self.centers = nn.Parameter(init_centers.clone())          # 直接学习c_k∈R，前向用sigmoid约束到[0,1]
        self.widths  = nn.Parameter(torch.full((K,), float(init_width)))  # 学习w_k，前向用softplus>0

    def forward(self, F_bins: int, device=None, dtype=None):
        # 生成 [K, F] 的滤波窗矩阵
        f = torch.linspace(0, 1, F_bins, device=device, dtype=dtype)  # 频率轴
        c = torch.sigmoid(self.centers).unsqueeze(1)                  # [K,1]
        w = F.softplus(self.widths).unsqueeze(1) + 1e-6               # [K,1], 避免0
        # 三角窗：w_k(f) = relu(1 - |f-c_k|/w_k)
        W = torch.relu(1.0 - torch.abs(f.unsqueeze(0) - c) / w)       # [K, F]
        # 归一化每个带宽窗以免能量随w漂移
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-9)
        return W                                                       # [K, F]


class AdaptiveFreqGate(nn.Module):
    """
    输入: x [B, L, D] (L=token长度/时间步)
    步骤: rFFT(L) → K个带宽窗 → 频带能量 → α=softmax → 频谱掩码M=αW → irFFT → FiLM融合
    输出: x_out [B, L, D], freq_prior [B, K]
    """
    def __init__(self, d_model: int, K: int = 8, use_film: bool = True,
                 res_scale: float = 0.2, init_gamma: float = 0.0):
        super().__init__()
        self.K = K
        self.use_film = use_film
        self.res_scale = res_scale

        self.bank = TriangularBank(K)
        # 由频带能量生成权重 α
        self.alpha_mlp = nn.Sequential(
            nn.Linear(K, K), nn.ReLU(inplace=True),
            nn.Linear(K, K)
        )
        # 频谱掩码强度 γ（可学，初值0→恒等映射）
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

        if use_film:
            self.film = nn.Linear(K, 2 * d_model)  # 生成scale和bias
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xt = x.transpose(1, 2)  # [B, D, L] 便于沿时间FFT

        # 实数FFT（频点数 F = floor(L/2)+1）
        Xf = torch.fft.rfft(xt, dim=-1)                 # [B, D, F]
        Fbins = Xf.shape[-1]
        W = self.bank(Fbins, device=Xf.device, dtype=Xf.real.dtype)  # [K, F]

        # 频带能量（按通道求和再对F加权）
        power = (Xf.real**2 + Xf.imag**2)               # [B, D, F]
        band_power = torch.einsum('bdf,kf->bdk', power, W)  # [B, D, K]
        band_power = band_power.mean(dim=1)                 # [B, K]

        # α: 自适应频带权重
        alpha_logits = self.alpha_mlp(band_power)          # [B, K]
        alpha = torch.softmax(alpha_logits, dim=-1)        # [B, K]

        # 频谱掩码 M = sum_k α_k * W_k
        M = torch.einsum('bk,kf->bf', alpha, W)            # [B, F]
        M = M.unsqueeze(1)                                  # [B, 1, F] 广播到通道维

        # 重加权频谱（1 + γ*M）
        Xf_masked = Xf * (1.0 + self.gamma * M)

        # 回到时域
        xt_filt = torch.fft.irfft(Xf_masked, n=L, dim=-1)  # [B, D, L]
        x_filt  = xt_filt.transpose(1, 2)                  # [B, L, D]

        if self.use_film:
            film_params = self.film(alpha)                 # [B, 2D]
            scale, bias = film_params.chunk(2, dim=-1)     # [B, D], [B, D]
            scale = scale.unsqueeze(1)                     # [B, 1, D]
            bias  = bias.unsqueeze(1)                      # [B, 1, D]
            y = x_filt * (1 + scale) + bias
        else:
            y = x_filt

        # 残差融合（恒等初始化，稳）
        out = x + self.res_scale * y
        return out, alpha  # alpha 作为 freq_prior 送给 MoE
