#!/usr/bin/env python3
"""
TAN NaN 诊断测试脚本。
在修复之前：freq_features 在 AMP autocast 下溢出 float16 → NaN。
在修复之后：freq extraction 显式运行在 float32 → 无 NaN。

用法:
    python scripts/test_tan_nan.py          # CPU 测试
    python scripts/test_tan_nan.py --cuda   # GPU + AMP autocast 测试
"""
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.freq_features import PerTokenFreqExtractor, extract_patch_freq_from_sequence
from utils.adaptive_norm import TokenAdaptiveNorm


def _check(name, t):
    has_nan = t.isnan().any().item()
    has_inf = t.isinf().any().item()
    status = "PASS" if (not has_nan and not has_inf) else "FAIL"
    print(f"  [{status}] {name}: shape={tuple(t.shape)}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
          f"nan={has_nan}, inf={has_inf}")
    return not has_nan and not has_inf


def test_freq_extraction(device, use_amp):
    print(f"\n=== Freq extraction (device={device}, amp={use_amp}) ===")
    B, L, C = 2, 672, 321
    patch_len, stride, K = 112, 24, 8
    N = (max(L, patch_len) - patch_len) // stride + 1

    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device)
    # Inject a high-mean channel (the NaN trigger: DC power overflows fp16)
    x[:, :, 0] = 5.0  # constant channel → DC power = (112*5)^2 = 313600 > 65504

    extractor = PerTokenFreqExtractor(K=K).to(device)
    ok = True

    ctx = autocast(dtype=torch.float16) if (use_amp and device != 'cpu') else torch.no_grad()
    with ctx:
        freq = extract_patch_freq_from_sequence(x, patch_len, stride, extractor, N)
        ok &= _check("freq_features", freq)
        ok &= _check("freq_sum (should ≈ 1)", freq.sum(dim=-1))

    return ok


def test_tan_forward(device, use_amp):
    print(f"\n=== TAN forward (device={device}, amp={use_amp}) ===")
    B, L, C = 2, 672, 7
    patch_len, stride, K = 96, 24, 8
    N = (max(L, patch_len) - patch_len) // stride + 1

    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device)
    # Pathological: near-constant patch with a spike
    x[:, 100:196, 0] = 3.0
    x[:, 196, 0] = -5.0

    extractor = PerTokenFreqExtractor(K=K).to(device)
    tan = TokenAdaptiveNorm(n_vars=C, patch_len=patch_len, stride=stride,
                             freq_dim=K, use_freq_cond=True).to(device)
    ok = True

    ctx = autocast(dtype=torch.float16) if (use_amp and device != 'cpu') else torch.no_grad()
    with ctx:
        freq = extract_patch_freq_from_sequence(x, patch_len, stride, extractor, N)
        ok &= _check("freq_features", freq)

        x_out, tan_ctx = tan.forward_in(x, freq, N)
        ok &= _check("TAN forward_in output", x_out)

        y = torch.randn(B, 24, C, device=device)
        freq_last = freq[:, -1, :].view(B, C, -1)
        y_out = tan.forward_out(y, tan_ctx, freq_features_last=freq_last)
        ok &= _check("TAN forward_out output", y_out)

    return ok


def test_tan_backward(device, use_amp):
    print(f"\n=== TAN backward (device={device}, amp={use_amp}) ===")
    B, L, C = 2, 672, 7
    patch_len, stride, K = 96, 24, 8
    N = (max(L, patch_len) - patch_len) // stride + 1

    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device, requires_grad=False)
    x[:, :, 0] = 5.0  # high-mean channel

    extractor = PerTokenFreqExtractor(K=K).to(device)
    tan = TokenAdaptiveNorm(n_vars=C, patch_len=patch_len, stride=stride,
                             freq_dim=K, use_freq_cond=True).to(device)
    linear = nn.Linear(C, C).to(device)

    ok = True
    ctx = autocast(dtype=torch.float16) if (use_amp and device != 'cpu') else torch.no_grad()

    if use_amp and device != 'cpu':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        with autocast(dtype=torch.float16):
            freq = extract_patch_freq_from_sequence(x, patch_len, stride, extractor, N)
            ok &= _check("freq_features", freq)
            x_out, tan_ctx = tan.forward_in(x, freq, N)
            ok &= _check("TAN forward_in", x_out)
            pred = linear(x_out[:, -24:, :])
            freq_last = freq[:, -1, :].view(B, C, -1)
            pred_denorm = tan.forward_out(pred, tan_ctx, freq_features_last=freq_last)
            ok &= _check("TAN forward_out", pred_denorm)
            target = torch.randn_like(pred_denorm)
            loss = nn.MSELoss()(pred_denorm, target)
            ok &= _check("loss", loss.unsqueeze(0))

        scaler.scale(loss).backward()
        for name, p in list(tan.named_parameters()) + list(extractor.named_parameters()):
            if p.grad is not None:
                ok &= _check(f"grad({name})", p.grad)
    else:
        freq = extract_patch_freq_from_sequence(x, patch_len, stride, extractor, N)
        ok &= _check("freq_features (no amp)", freq)

    return ok


def test_extreme_values(device, use_amp):
    print(f"\n=== Extreme value test (device={device}, amp={use_amp}) ===")
    K = 8
    extractor = PerTokenFreqExtractor(K=K).to(device)
    ok = True

    for val, desc in [(0.0, "all-zero"), (100.0, "large"), (1e-6, "tiny"), (255.0, "max-byte")]:
        patches = torch.full((4, 5, 112), val, device=device)
        if val == 0:
            patches[:, :, 0] = 0.001  # avoid all-zero FFT
        ctx = autocast(dtype=torch.float16) if (use_amp and device != 'cpu') else torch.no_grad()
        with ctx:
            freq = extractor(patches)
            ok &= _check(f"freq({desc}, val={val})", freq)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    all_ok = True

    for use_amp in ([False, True] if device == 'cuda' else [False]):
        all_ok &= test_freq_extraction(device, use_amp)
        all_ok &= test_tan_forward(device, use_amp)
        all_ok &= test_tan_backward(device, use_amp)
        all_ok &= test_extreme_values(device, use_amp)

    print("\n" + "=" * 50)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print("=" * 50)
