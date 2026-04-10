# layers/freq_features.py
"""
Per-token frequency feature extractor.

Shared infrastructure for FIR-MoE (frequency-informed routing) and
TAN (token-adaptive normalization). For each patch/token of length P,
this module computes a K-dimensional frequency-band energy distribution
via rFFT + learnable triangular band filters.

This is distinct from the global AFS-Gate (which averages over channels
and produces one K-vector per batch sample). Here we keep per-token
resolution so downstream modules can make token-level decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularBankShared(nn.Module):
    """K learnable triangular band-pass filters over normalized freq [0,1].
    Compatible with AFS-Gate's _TriangularBank but exposed as a reusable module
    so FIR-MoE / TAN can share parameters with AFS-Gate if desired.
    """
    def __init__(self, K: int, init_centers=None, init_width: float = 0.25):
        super().__init__()
        self.K = K
        if init_centers is None:
            init_centers = torch.linspace(0.05, 0.95, K)
        self.centers = nn.Parameter(init_centers.clone())
        self.widths = nn.Parameter(torch.full((K,), float(init_width)))

    def forward(self, F_bins: int, device=None, dtype=None):
        f = torch.linspace(0, 1, F_bins, device=device, dtype=dtype)
        c = torch.sigmoid(self.centers).unsqueeze(1)                 # [K,1]
        w = F.softplus(self.widths).unsqueeze(1) + 1e-6              # [K,1]
        W = torch.relu(1.0 - torch.abs(f.unsqueeze(0) - c) / w)      # [K,F]
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-9)                 # row-normalize
        return W


class PerTokenFreqExtractor(nn.Module):
    """
    Per-token frequency band energy distribution.

    Input : patches  [B*C, N, P]   (raw patch values)
       or  tokens_t  [B*C, N, D]   (token embeddings; uses d-axis FFT — less physical)
    Output: band_freq [B*C, N, K]  (normalized energy per learnable band)

    Two use-cases:
      1. FIR-MoE: concat band_freq with hidden state as router input.
      2. TAN:     use band_freq to modulate normalization strength per token.
    """
    def __init__(self, K: int = 8, share_bank: TriangularBankShared = None):
        super().__init__()
        self.K = K
        if share_bank is None:
            self.bank = TriangularBankShared(K)
            self._shared = False
        else:
            self.bank = share_bank
            self._shared = True

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [B*C, N, P] or [*, P] where P is patch length.
        Returns: same leading dims with last dim = K.
        """
        # Compute rFFT along the last (time) axis of each patch
        Xf = torch.fft.rfft(patches, dim=-1)               # [..., F]
        Fbins = Xf.shape[-1]
        W = self.bank(Fbins, device=Xf.device, dtype=Xf.real.dtype)  # [K, F]
        power = Xf.real ** 2 + Xf.imag ** 2                # [..., F]
        band_power = torch.einsum('...f,kf->...k', power, W)         # [..., K]

        # Normalize to a distribution per token (sum to 1). This makes the
        # feature scale-invariant and robust to overall amplitude differences.
        band_power = band_power / (band_power.sum(dim=-1, keepdim=True) + 1e-9)
        return band_power


def extract_patch_freq_from_sequence(
        x_raw: torch.Tensor,
        patch_len: int,
        stride: int,
        extractor: PerTokenFreqExtractor,
        n_tokens: int,
) -> torch.Tensor:
    """
    Convenience: turn a raw [B, L, C] sequence into per-patch frequency features
    [B*C, N, K], aligning to the tokenization used by the main model.

    Handles overlapping patches (stride < patch_len) and minor length mismatches.
    """
    B, L, C = x_raw.shape
    P = int(patch_len)
    N = int(n_tokens)

    # x_var: [B, C, N', P]
    x_var = x_raw.permute(0, 2, 1).contiguous()
    if x_var.shape[-1] < P:
        pad = P - x_var.shape[-1]
        x_var = torch.nn.functional.pad(x_var, (0, pad), mode='replicate')
    x_patches = x_var.unfold(dimension=-1, size=P, step=stride)      # [B,C,N',P]
    Np = x_patches.shape[2]

    if Np < N:
        last = x_patches[:, :, -1:, :].expand(-1, -1, N - Np, -1)
        x_patches = torch.cat([x_patches, last], dim=2)
    elif Np > N:
        x_patches = x_patches[:, :, :N, :]

    x_patches = x_patches.contiguous().view(B * C, N, P)             # [B*C,N,P]
    freq = extractor(x_patches)                                       # [B*C,N,K]
    return freq
