# 研究概述：面向非平稳时间序列的频率自适应预测框架

## 一、问题与动机

长期时间序列预测（Long-term Time Series Forecasting）的核心困难在于：**同一序列中混合了多种时间尺度的模式**——缓慢漂移的趋势、周期性季节信号、突发事件与高频噪声共存。现有方法存在两个系统性缺陷：

1. **归一化粒度过粗**：主流方法（RevIN、Non-stationary Transformer）对整个输入窗口计算一组全局均值/方差。但一个 672 步的窗口可能跨越分布突变（如电力数据中的设备故障），全局统计量将突变区域过度平滑，同时对平稳区域引入不必要的偏差。

2. **MoE 路由缺乏频率先验**：Mixture-of-Experts 在 NLP 中表现优异，但直接迁移到时序场景时，由于同一通道的 patch token 高度相似，路由器难以区分不同 token 的时间模式特征，导致专家坍缩（多个专家学到相同行为）。

**统一洞察**：这两个问题的根源相同——模型缺乏对**局部频率特性**的感知。我们提出：时间序列每个局部片段的频率分布应当显式地指导模型在归一化和计算两个层面的自适应行为。

## 二、方法

### 2.1 核心贡献：Token 自适应归一化（Token-Adaptive Normalization, TAN）

TAN 将归一化决策从全局序列级细化到 patch-token 级。对每个 patch token $i$：

$$x_{\text{out},i} = (1 - g_i) \cdot x_i + g_i \cdot \frac{x_i - \mu_i}{\sigma_i}$$

其中 $\mu_i, \sigma_i$ 是该 patch 的局部统计量（detached，不参与反传），$g_i = \sigma(\mathbf{w}^\top \mathbf{f}_i + b)$ 是由该 token 的**频率特征** $\mathbf{f}_i \in \mathbb{R}^K$ 驱动的门控信号。

**频率特征提取**：对每个 patch 做 rFFT，通过 $K$ 个可学习三角带通滤波器得到归一化的频带能量分布 $\mathbf{f}_i \in \Delta^{K-1}$（单纯形上的概率向量）。

**设计直觉**：
- 高频为主的 token（噪声/突变区域）→ $g_i \approx 1$ → 强局部归一化，抑制异常值
- 低频为主的 token（平稳趋势段）→ $g_i \approx 0$ → 接近恒等，保留原始信息

**反归一化**同样是自适应的，使用最后一个输入 token 的频率特征条件化反归一化强度。

**数值稳定性设计**：
- 门控偏置初始化为 $-2$（$\sigma(-2) \approx 0.12$），训练初期接近恒等，不伤害 baseline 收敛
- 统计量 detach，避免梯度穿越统计量引发训练后期 NaN
- 全程 float32 计算（与 PyTorch 的 LayerNorm/BatchNorm 在 AMP 下的行为一致），防止 GradScaler 导致的梯度溢出

### 2.2 补充贡献：频率引导的专家路由（Frequency-Informed MoE Routing, FIR-MoE）

在 MoE 路由器的输入中融合频率特征：

$$\text{logits}_i = \mathbf{W}_{\text{gate}} \cdot [\mathbf{h}_i; \mathbf{f}_i]$$

其中 $\mathbf{h}_i$ 是 token 的隐藏表示，$\mathbf{f}_i$ 是频率特征。额外引入专家频率特化正则：

$$\mathcal{L}_{\text{spec}} = -\frac{1}{E}\sum_{e=1}^{E} D_{\text{KL}}(\mathbf{p}_e \| \text{uniform}_K)$$

其中 $\mathbf{p}_e$ 是专家 $e$ 被分配 token 的平均频率分布。该正则鼓励每个专家关注不同的频带，实现自然的"趋势专家"/"季节性专家"/"噪声专家"特化。

### 2.3 频谱一致性正则（Spectral Consistency Regularization）

在 MSE loss 基础上添加频域约束，惩罚预测信号与真实信号的功率谱差异：

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_s \cdot \frac{1}{F}\sum_{k=0}^{F-1} \left| |X_{\text{pred}}[k]|^2 - |X_{\text{true}}[k]|^2 \right|$$

与 TAN 形成"输入-输出双端频率自适应"的闭环：TAN 在输入侧做频率条件化归一化，频谱 loss 在输出侧做频率一致性监督。

### 2.4 统一框架

TAN、FIR-MoE 和频谱 loss **共享同一个核心洞察**：局部频率特性应指导模型的自适应行为。

| 模块 | 作用位置 | 频率信息用途 |
|------|---------|-------------|
| TAN | 输入归一化 | 频带能量 → 归一化门控 $g_i$ |
| FIR-MoE | 推理层路由 | 频带能量 → 专家分配权重 |
| 频谱 loss | 输出监督 | 功率谱差异 → 辅助训练信号 |

## 三、实验设计

### 3.1 主实验

**基座模型**：TimerXL（基于 GPT 架构的 Patch-level 自回归时序 Transformer）

**数据集（9 个）**：ECL (321 vars), ETTh1, ETTh2, ETTm1, ETTm2 (7 vars each), Weather (21), Traffic (862), Solar (137), Exchange (8)

**预测长度**：$H \in \{96, 192, 336, 720\}$

**对比模型**：
| 变体 | 描述 |
|------|------|
| Baseline | TimerXL 无增强 |
| TAN | TimerXL + Token 自适应归一化 |
| FIR-MoE + TAN | 完整方法（MoE + 频率路由 + TAN） |

**指标**：MSE, MAE

### 3.2 消融实验

在 ECL、ETTm1、Traffic、Weather 上，pred_len ∈ {96, 336}：

| 变体 | 说明 |
|------|------|
| TAN (full) | 完整 TAN（K=8, freq_cond=True） |
| w/o freq cond | TAN 去掉频率条件（固定门控） |
| K=4 | 减少频带数 |
| K=16 | 增加频带数 |
| TAN + spec loss | TAN + 频谱一致性 loss |

### 3.3 可解释性分析

- TAN 门控值 $g_i$ 沿时间轴的可视化：展示在分布突变点附近 $g$ 如何自适应跃变
- 专家频率签名热力图 $[\text{Experts} \times \text{Freq Bands}]$：展示不同专家的频率特化模式
- 跨数据集对比：周期数据（ECL）vs 趋势数据（ETT）vs 噪声数据（Weather）

## 四、实验结果

### 4.1 Phase 4 全量 Benchmark（7 数据集 × 4 预测长度 × 3 变体）

**MSE 胜率统计**：

| 方法 | MSE 胜出次数（/28） | MAE 胜出次数（/28） |
|------|-------------------|-------------------|
| baseline | 6 | 4 |
| **TAN** | **14** | **13** |
| FIR-MoE+TAN | 8 | 11 |

**按数据集平均 MSE**：

| Dataset | n_vars | baseline | TAN | FIR-MoE+TAN | 最优 |
|---------|--------|----------|-----|-------------|------|
| ECL | 321 | 0.5966 | 0.6133 | **0.5826** | FIR-MoE+TAN |
| ETTh1 | 7 | 0.2327 | **0.2099** | 0.2218 | TAN |
| ETTh2 | 7 | 0.2093 | **0.2065** | 0.2359 | TAN |
| ETTm1 | 7 | 0.1109 | **0.0884** | 0.1341 | TAN |
| ETTm2 | 7 | 0.1200 | **0.1108** | 0.1276 | TAN |
| Weather | 21 | **0.2543** | 0.2563 | 0.2583 | baseline |
| Traffic | 862 | 0.4797 | **0.4567** | 0.5274 | TAN |

### 4.2 关键观察

1. **TAN 是最稳定的改进**：在 5/7 数据集上取得最低平均 MSE，长预测（720 步）优势尤为明显（如 ETTm1-720：0.1397 vs baseline 0.1685，−17.1%）

2. **FIR-MoE+TAN 在高维数据集上有竞争力**：ECL (321 vars) 上平均 MSE 最低，但在低维数据集上 MoE 的 aux loss 引入退化

3. **Weather 数据集区分度低**：三种方法差异 < 2%，可能因为 Weather 的突变模式对归一化策略不敏感

## 五、代码结构

```
layers/freq_features.py       # 共享频率特征提取器（rFFT + 三角带通滤波）
utils/adaptive_norm.py        # TAN 实现（forward_in / forward_out）
layers/moe_ffn.py             # FIR-MoE 路由器扩展
models/timer_xl.py            # 主模型集成
exp/exp_forecast.py           # 训练循环（含频谱 loss）
scripts/jobs/
  phase4_full_benchmark.sh    # 全量 benchmark（9 数据集 × 4 预测长度）
  phase5_ablation.sh          # TAN 消融实验
```

## 六、下一步

1. **消融实验**（Phase 5）：验证 TAN 各组件贡献，包括频率条件、频带数 K、频谱 loss
2. **可解释性可视化**：TAN 门控值时间演化、专家频率签名热力图
3. **外部 baseline 对比**：引用 PatchTST、iTransformer、DLinear 等已发表结果
4. **论文撰写**
