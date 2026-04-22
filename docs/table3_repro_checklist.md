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

它会按 **原始 table3_96pred96.sh 的 9 个数据集** 依次执行：

1. `MODE=paper_literal`
2. `MODE=table3_matched`
3. 原始 `table3_96pred96.sh` 的 baseline（同 seed / GPU 设置）

并输出统一 CSV：
- 对有论文目标值的数据集（ecl/etth1/traffic/weather/solar）给出 `|run-target|` gap
- 对其余数据集（etth2/ettm1/ettm2/exchange）保留 `target=NA`，只做方法间横向比较。

## 7) 重要更新：原始 table3_96pred96.sh 已改为论文对齐参数

为避免“compare 脚本里比较对象不公平”，`scripts/jobs/table3_96pred96.sh` 已直接改成论文对齐主干：

- lookback 统一 `seq_len=672`
- 5 个主数据集采用与 Table 3 更一致的 dataset-specific 配置
- baseline/tan/fir_moe_tan 都在同一套对齐主干上比较

这样你直接跑 `table3_96pred96.sh` 本身，就更接近论文对标口径。

## 8) 为什么会 OOM（尤其 rank1/rank2 在 test/load ckpt 阶段）

常见是某张卡（通常 GPU0）已经被其他进程占用，DDP 仍然把该卡纳入 `nproc_per_node`，导致某 rank 在加载 ckpt 时爆显存。

现在脚本支持三种方式：

1. 自动选空闲卡（默认 `AUTO_GPU=1`）
2. 手动指定 `GPU_IDS`（例如 `GPU_IDS=1,2,3,4,5,6,7`）
3. 显式 `CUDA_VISIBLE_DEVICES` + 对应 `NGPU/NPROC_PER_NODE`

并默认启用：

- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8`

用于缓解显存碎片。


- 若出现 `expandable_segment_ INTERNAL ASSERT FAILED`，请不要使用 `expandable_segments:True`。
  建议：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8`。
