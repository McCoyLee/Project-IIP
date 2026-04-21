# Table 3（96-pred-96）复现排查（多卡版）

> 目标：尽量对齐 Timer-XL 论文（arXiv:2410.04803）Table 3/11。

## 1) 先区分两种模式

- `MODE=table3_matched`（默认，推荐）
  - ETTh1 使用 `d_ff=2048 + --use_norm --valid_last`
  - 这是基于你实测（以及官方脚本习惯）更接近 Table 3 的配置。
- `MODE=paper_literal`
  - 按 Table 11 字段字面执行（ETTh1 `d_ff=4096`，不加额外开关）
  - 你当前实测此模式会明显偏离 Table 3（如 MSE≈0.47）。

## 2) 多卡必须保持论文全局 batch

DDP 中 `batch_size` 是每卡值：

- `global_batch = local_batch × GPU数`
- 脚本自动使用 `local_batch = paper_batch / NPROC_PER_NODE`

如果不能整除，会直接报错退出。

## 3) 论文对齐字段（Table 11）

脚本 `paper_literal` 模式采用：

- ECL: `L=5,D=512,H=8,d_ff=2048,LR=5e-4,BS=4,Epoch=10`
- ETTh1: `L=1,D=1024,H=8,d_ff=4096,LR=1e-4,BS=32,Epoch=10`
- Traffic: `L=4,D=512,H=8,d_ff=2048,LR=5e-4,BS=4,Epoch=10`
- Weather: `L=4,D=512,H=8,d_ff=2048,LR=5e-4,BS=32,Epoch=10`
- Solar: `L=6,D=512,H=8,d_ff=2048,LR=1e-4,BS=16,Epoch=10`

并统一：

- `seq_len=672`（`T=7,P=96` 对应）
- `input_token_len=96, output_token_len=96, test_pred_len=96`

## 4) 为何会出现“mse接近但mae偏高”

常见原因不是单一 bug，而是：

1. 单次 seed 波动（尤其 DDP、多进程顺序导致非严格确定性）
2. 论文表格数值可能来自多次试验统计（论文通常不只看单次）
3. 你实际运行时混入了非 Table 11 开关（例如 use_norm/valid_last）

所以建议：

- 先跑推荐模式 `MODE=table3_matched`
- 仅把 `MODE=paper_literal` 当作“字面字段对照实验”
- 至少 3 个 seed（例如 2021/42/3407）看均值或中位数

## 5) 推荐命令（ETTh1, 8卡）

```bash
export DATA_ROOT=/path/to/datasets
SEED=2021 MODE=table3_matched NPROC_PER_NODE=8 ONLY_DATASET=etth1 \
  bash scripts/jobs/table3_96pred96_paper_multigpu.sh
```

如需工程模式：

```bash
SEED=2021 MODE=paper_literal NPROC_PER_NODE=8 ONLY_DATASET=etth1 \
  bash scripts/jobs/table3_96pred96_paper_multigpu.sh
```

## 6) 多数据集公平对比（复现 vs 实验）

已提供对比脚本：

```bash
scripts/jobs/compare_table3_multi_dataset.sh
```

它会对 `ecl/etth1/traffic/weather/solar` 依次执行：

1. `MODE=paper_literal`（字面对照）
2. `MODE=table3_matched`（推荐复现）
3. 你的实验脚本 `table3_96pred96.sh` 的 baseline

并输出统一 CSV，字段包含每个数据集三条线的 `mse/mae`，便于横向公平比较。
