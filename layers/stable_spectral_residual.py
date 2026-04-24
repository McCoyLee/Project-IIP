import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class StableKoopmanSpectralResidual(nn.Module):
    """Stable Fourier/Koopman residual branch for zero-shot rolling forecasts.

    A time shift is diagonal in the Fourier basis: each Fourier mode is an
    eigenfunction of the Koopman shift operator. This module uses that structure
    to build an explicit horizon-aware continuation from the input history, then
    mixes it with the neural forecast through a bounded gate.

    The branch is intentionally low-capacity. It is not a replacement for the
    backbone; it is a stable residual prior for long horizons.
    """

    def __init__(
        self,
        n_vars: Optional[int] = None,
        max_modes: int = 8,
        lowpass_ratio: float = 0.35,
        max_mix: float = 0.35,
        init_gate: float = -3.0,
        trend_window: int = 192,
        use_trend: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_vars = int(n_vars) if n_vars is not None and int(n_vars) > 0 else None
        self.max_modes = int(max_modes)
        self.lowpass_ratio = float(lowpass_ratio)
        self.max_mix = float(max_mix)
        self.trend_window = int(trend_window)
        self.use_trend = bool(use_trend)
        self.eps = float(eps)

        gate_shape = (1, 1, self.n_vars) if self.n_vars is not None else (1, 1, 1)
        self.mix_logit = nn.Parameter(torch.full(gate_shape, float(init_gate)))
        self.last_stats: Dict[str, torch.Tensor] = {}

    def _gate(self, concentration: torch.Tensor, channels: int) -> torch.Tensor:
        gate = torch.sigmoid(self.mix_logit.float())
        if gate.shape[-1] != channels:
            gate = gate.mean(dim=-1, keepdim=True).expand(1, 1, channels)
        return self.max_mix * gate * concentration.unsqueeze(1)

    def _trend_forecast(self, history: torch.Tensor, horizon: int) -> torch.Tensor:
        B, C, L = history.shape
        if not self.use_trend:
            return history.mean(dim=-1, keepdim=True).expand(B, C, horizon)

        win = max(2, min(int(self.trend_window), L))
        tail = history[..., -win:]
        t = torch.arange(win, device=history.device, dtype=history.dtype)
        t = t - t.mean()
        denom = t.square().sum().clamp_min(self.eps)
        tail_mean = tail.mean(dim=-1, keepdim=True)
        slope = ((tail - tail_mean) * t.view(1, 1, win)).sum(dim=-1, keepdim=True) / denom

        steps = torch.arange(1, horizon + 1, device=history.device, dtype=history.dtype)
        return history[..., -1:] + slope * steps.view(1, 1, horizon)

    def _spectral_forecast(self, history: torch.Tensor, horizon: int):
        B, C, L = history.shape
        centered = history - history.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1)
        bins = spectrum.shape[-1]
        max_available = max(0, bins - 1)

        keep_by_ratio = int(math.floor(max_available * self.lowpass_ratio))
        keep = max(0, min(self.max_modes, keep_by_ratio, max_available))
        if keep == 0:
            seasonal = torch.zeros(B, C, horizon, device=history.device, dtype=history.dtype)
            concentration = torch.zeros(B, C, device=history.device, dtype=history.dtype)
            return seasonal, concentration

        coeff = spectrum[..., 1:keep + 1]
        modes = torch.arange(1, keep + 1, device=history.device, dtype=history.dtype)
        steps = torch.arange(L, L + horizon, device=history.device, dtype=history.dtype)
        phase = 2.0 * math.pi * steps[:, None] * modes[None, :] / float(L)
        basis = torch.polar(torch.ones_like(phase), phase)
        seasonal = (coeff.unsqueeze(-2) * basis.view(1, 1, horizon, keep)).sum(dim=-1).real
        seasonal = 2.0 * seasonal / float(L)

        power = spectrum.abs().square()
        low_power = power[..., 1:keep + 1].sum(dim=-1)
        total_power = power[..., 1:].sum(dim=-1).clamp_min(self.eps)
        concentration = (low_power / total_power).clamp(0.0, 1.0)
        return seasonal, concentration

    def forward(self, x_history: torch.Tensor, neural_forecast: torch.Tensor) -> torch.Tensor:
        """Blend neural forecast with stable spectral extrapolation.

        x_history: [B, L, C]
        neural_forecast: [B, H, C]
        """
        if x_history.dim() != 3 or neural_forecast.dim() != 3:
            raise ValueError("SKSR expects [B,L,C] history and [B,H,C] forecast tensors")

        out_dtype = neural_forecast.dtype
        device_type = x_history.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            hist = x_history.float().transpose(1, 2).contiguous()
            neural = neural_forecast.float()
            horizon = neural.shape[1]
            channels = neural.shape[-1]

            trend = self._trend_forecast(hist, horizon)
            seasonal, concentration = self._spectral_forecast(hist, horizon)
            spectral = (trend + seasonal).transpose(1, 2).contiguous()

            mix = self._gate(concentration, channels)
            mixed = neural + mix * (spectral - neural)

            self.last_stats = {
                "sksr/mix_mean": mix.detach().mean(),
                "sksr/mix_max": mix.detach().max(),
                "sksr/concentration": concentration.detach().mean(),
            }

        return mixed.to(dtype=out_dtype)
