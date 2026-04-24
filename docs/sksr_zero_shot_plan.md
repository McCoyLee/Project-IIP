# SKSR: Stable Koopman Spectral Residual for Zero-Shot Rolling Forecasting

## Goal

The target thesis claim is not simply to add capacity to Timer-XL.  The target
is a mathematically constrained modification that improves zero-shot rolling
prediction at multiple horizons while preserving a fair comparison with the
baseline backbone.

For Table 4 style evaluation, the model is trained with `output_token_len=24`
and then evaluated by rolling to `pred_len in {96, 192, 336, 720}`.  This is a
zero-shot horizon shift: the model never directly optimizes the long horizons.

## Core Idea

SKSR adds a low-capacity residual branch after the Timer-XL forecast:

```text
final_forecast = neural_forecast + gate * (spectral_forecast - neural_forecast)
```

The spectral branch is built from the input history, not from a learned deep
network.  It uses two stable components:

1. A local least-squares trend continuation.
2. A low-frequency Fourier continuation.

The mixing gate is bounded and initialized close to zero.  Therefore the model
starts near the original baseline and learns to use the spectral prior only when
it helps.

## Mathematical Rationale

Let `S` be the time-shift operator, `(Sx)_t = x_{t+1}`.  On periodic components,
the Fourier modes are eigenfunctions of this shift:

```text
S phi_k = exp(i omega_k) phi_k
```

Thus a stable continuation over an unseen horizon can be written by propagating
Fourier phases instead of learning a separate horizon-specific head.  SKSR keeps
only low-frequency modes and bounds the residual mixture, which reduces the
risk of high-frequency error amplification during rolling prediction.

The bounded gate is

```text
g = g_max * sigmoid(theta) * rho(x)
```

where `rho(x)` is the low-frequency energy concentration of the input.  The
branch is trusted more when the signal has a strong stable low-frequency
structure and less when the signal is dominated by high-frequency variation.

## Why This Targets the Current Table 4 Weakness

The existing results show that `tan` improves several datasets but fails on
some long rolling cases, especially ETTh2.  SKSR addresses a different failure
mode: long-horizon drift under repeated 24-step rollout.  It does not replace
normalization or MoE routing; it adds a stable horizon-aware prior at the output
level.

This means the fair primary variant should be:

```text
--model timer_xl_sksr
```

without TAN or MoE.  TAN and FIR-MoE can remain ablations.

## Fairness Rules

Use the same training protocol as the Table 4 baseline:

- Same datasets.
- Same `seq_len=672`.
- Same `input_token_len=96` and `input_token_stride=24`.
- Same `output_token_len=24`.
- Same backbone size: `e_layers=3`, `d_model=256`, `n_heads=4`, `d_ff=1024`.
- Same optimizer, learning rate, epochs, patience, and seed.
- Same rolling evaluation at 96, 192, 336, and 720.

SKSR adds only one per-channel scalar gate and a deterministic spectral
continuation.  It should be reported as a low-parameter residual prior, not as a
larger backbone.

## Recommended Experiment Order

1. Run `sksr` on all 9 datasets with seed 42.
2. Compare against the already completed Table 4 baseline.
3. If SKSR beats baseline on at least 7 of 9 dataset averages, freeze the
   method and run seeds `42, 2021, 3407` for statistical support.
4. If it beats 5 or 6 datasets, run `sksr_small` to reduce possible harm on
   noisy datasets.
5. Keep `sksr_tan` as a secondary ablation, not the main claim.

## Command

```bash
export DATA_ROOT=/path/to/datasets
VARIANTS=sksr SEED=42 NGPU=8 bash scripts/jobs/table4_sksr_672rolling.sh
```

For a fresh paired comparison:

```bash
export DATA_ROOT=/path/to/datasets
VARIANTS=baseline,sksr SEED=42 NGPU=8 bash scripts/jobs/table4_sksr_672rolling.sh
```

For targeted multi-seed validation:

```bash
for s in 42 2021 3407; do
  VARIANTS=sksr SEED=$s NGPU=8 bash scripts/jobs/table4_sksr_672rolling.sh
done
```

## Thesis Wording

A concise method statement:

> We propose a Stable Koopman Spectral Residual branch for zero-shot rolling
> forecasting.  The branch exploits the fact that Fourier modes diagonalize the
> time-shift operator, producing an explicit stable continuation for unseen
> horizons.  A bounded data-dependent gate mixes this continuation with the
> neural forecast, preserving the Timer-XL baseline at initialization and
> allowing the model to use the spectral prior only when the input exhibits
> concentrated low-frequency structure.
