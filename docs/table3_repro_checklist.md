# Table 3（96-pred-96）复现排查（多卡版）

> 目标：尽量对齐 Timer-XL 论文（arXiv:2410.04803）Table 3/11。

## 1) 先区分两种模式

- `MODE=paper_strict`（默认）
  - **只使用论文 Table 11 明示字段**（L/D/H/M、LR、Batch、Epoch、P）
  - 不额外加 `use_norm/valid_last` 这类论文表格未显式给出的开关。
- `MODE=official_script_like`
  - 允许加入官方脚本常见开关（当前是 `--use_norm --valid_last`）
  - 更偏“工程复现”，不等价于论文表格逐字段复现。

## 2) 多卡必须保持论文全局 batch

DDP 中 `batch_size` 是每卡值：

- `global_batch = local_batch × GPU数`
- 脚本自动使用 `local_batch = paper_batch / NPROC_PER_NODE`

如果不能整除，会直接报错退出。

## 3) 论文对齐字段（Table 11）

脚本 `paper_strict` 模式采用：

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

- 先跑 `MODE=paper_strict` 看“纯论文字段”结果
- 再跑 `MODE=official_script_like` 看工程可达上限
- 至少 3 个 seed（例如 2021/42/3407）看均值或中位数

## 5) 推荐命令（ETTh1, 8卡）

```bash
export DATA_ROOT=/path/to/datasets
SEED=2021 MODE=paper_strict NPROC_PER_NODE=8 ONLY_DATASET=etth1 \
  bash scripts/jobs/table3_96pred96_paper_multigpu.sh
```

如需工程模式：

```bash
SEED=2021 MODE=official_script_like NPROC_PER_NODE=8 ONLY_DATASET=etth1 \
  bash scripts/jobs/table3_96pred96_paper_multigpu.sh
```
