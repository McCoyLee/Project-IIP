# Table 3（96-pred-96）复现排查（多卡版）

> 适用目标：尽量对齐 Timer-XL 论文（arXiv:2410.04803 / ICLR 2025）Table 3 的多变量结果。

## 先说结论：你当前结果偏高的最可能原因

你给出的命令使用的是 `scripts/jobs/table3_96pred96.sh`。这个脚本的核心设置与论文 Table 11 差异较大：

- 你当前脚本：`L=3, D=256, H=4, d_ff=1024, epochs=50`，并且附带 `--ci_backbone --cosine`。
- 论文 Table 11（Timer-XL multivariate）：主干是 `L=5, D=512, H=8, P=96, epochs=10`。

这会直接导致训练行为和结果区间与论文不一致。

---

## 多卡复现的关键原则（一定要做）

**保持“有效全局 batch size”与论文一致。**

在 DDP 中，`batch_size` 是每卡值：

- `global_batch = local_batch × GPU卡数`
- 因此应设置：`local_batch = paper_batch / GPU卡数`

如果不能整除，请改 GPU 数（建议只用 1/2/4/8 卡），不要随意改总 batch。

---

## 论文里能直接对齐的信息（Table 11）

- Timer-XL（multivariate, ECL）: `LR=5e-4, Batch=4, Epoch=10`
- ETTh1 行（multivariate）: `LR=1e-4, Batch=32, Epoch=10`
- Weather 行（multivariate）: `LR=5e-4, Batch=32, Epoch=10`
- Solar 行（multivariate）: `LR=1e-4, Batch=16, Epoch=10`

并且，论文对 multivariate 指标是“在变量维做平均”的 MSE/MAE（即 all-channel mean）。

- 补充：论文复杂度分析使用典型设置 `T=7, P=96`，对应 lookback 长度约为 `7×96=672`。因此只做 `sl96` 往往会显著偏离。

---

## 本仓库已提供的多卡对齐脚本

新增脚本：

```bash
scripts/jobs/table3_96pred96_paper_multigpu.sh
```

特点：

1. 采用 **数据集级** Table 11 配置（不是所有数据集都用同一主干）
   - ECL: `L=5,D=512,H=8,d_ff=2048,LR=5e-4,BS=4`
   - ETTh1: `L=1,D=1024,H=8,d_ff=4096,LR=1e-4,BS=32`
   - Traffic: `L=4,D=512,H=8,d_ff=2048,LR=5e-4,BS=4`
   - Weather: `L=4,D=512,H=8,d_ff=2048,LR=5e-4,BS=32`
   - Solar: `L=6,D=512,H=8,d_ff=2048,LR=1e-4,BS=16`
2. 保持 DDP 多卡
3. 自动按 `NPROC_PER_NODE` 计算每卡 batch，使全局 batch 与论文值一致
4. 仅覆盖 Table 3 的 5 个数据集（ECL/ETTh1/Traffic/Weather/Solar）

---

## 推荐运行方式（示例）

以 ETTh1 为例，8 卡复现（论文全局 batch=32）：

```bash
export DATA_ROOT=/path/to/datasets
NPROC_PER_NODE=8 ONLY_DATASET=etth1 bash scripts/jobs/table3_96pred96_paper_multigpu.sh
```

此时每卡 batch 会自动设为 `32/8=4`，全局 batch 仍为 32。

若是 ECL（论文 batch=4），建议 1/2/4 卡，不建议 8 卡（4 无法被 8 整除）。

---

## 额外注意

- 数据集路径和文件名必须与脚本一致（`ETT/ETTh1.csv`、`Electricity/ECL.csv` 等）。
- 尽量不要混入旧脚本的 `--ci_backbone` 或其它论文未给出的策略。
- 先单数据集逐个对齐（如先 ETTh1），再跑全表。
