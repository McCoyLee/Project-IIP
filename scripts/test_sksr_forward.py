import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from layers.stable_spectral_residual import StableKoopmanSpectralResidual


def test_sksr_forward_backward():
    torch.manual_seed(7)
    batch, seq_len, horizon, channels = 3, 96, 24, 5
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    base = torch.sin(t).view(1, seq_len, 1).repeat(batch, 1, channels)
    noise = 0.05 * torch.randn(batch, seq_len, channels)
    history = base + noise
    neural = history[:, -1:, :].repeat(1, horizon, 1).clone().requires_grad_(True)

    sksr = StableKoopmanSpectralResidual(
        n_vars=channels,
        max_modes=4,
        lowpass_ratio=0.5,
        max_mix=0.35,
        init_gate=-2.5,
        trend_window=48,
    )
    out = sksr(history, neural)

    assert out.shape == (batch, horizon, channels)
    assert torch.isfinite(out).all()
    assert "sksr/mix_mean" in sksr.last_stats
    assert 0.0 <= float(sksr.last_stats["sksr/mix_mean"]) <= 0.35

    loss = out.square().mean()
    loss.backward()
    assert neural.grad is not None
    assert torch.isfinite(neural.grad).all()
    assert sksr.mix_logit.grad is not None
    assert torch.isfinite(sksr.mix_logit.grad).all()


if __name__ == "__main__":
    test_sksr_forward_backward()
    print("SKSR smoke test passed")
