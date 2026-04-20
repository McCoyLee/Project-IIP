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

### 2.3 两者的统一

TAN 和 FIR-MoE **共享同一个频率特征提取器**，构成从预处理到推理的全链路频率自适应框架：

| 模块 | 自适应对象 | 频率特征用途 |
|------|-----------|-------------|
| TAN | 归一化强度 | 频带能量 → 归一化门控 $g_i$ |
| FIR-MoE | 专家分配 | 频带能量 → 路由权重 |

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
| FIR-MoE + TAN | 完整方法 |

**指标**：MSE, MAE

### 3.2 消融实验

**TAN 消融**：
- w/o freq conditioning（固定门控值，不用频率条件）
- w/o local stats（仅用全局统计量 + 频率门控）
- 不同频带数 $K \in \{4, 8, 16\}$

**FIR-MoE 消融**：
- w/o freq input（退化为标准 MoE）
- w/o specialization loss（有频率输入但不约束特化）

### 3.3 可解释性分析

- TAN 门控值 $g_i$ 沿时间轴的可视化：展示在分布突变点附近 $g$ 如何自适应跃变
- 专家频率签名热力图 $[\text{Experts} \times \text{Freq Bands}]$：展示不同专家的频率特化模式
- 跨数据集对比：周期数据（ECL）vs 趋势数据（ETT）vs 噪声数据（Weather）

## 四、初步结果（ECL 数据集, $H=24$）

| 模型 | MSE | MAE | vs Baseline |
|------|-----|-----|-------------|
| Baseline | 0.4742 | 0.3611 | — |
| TAN | **0.3768** | **0.3387** | **−20.5%** |
| FIR-MoE + TAN | 0.3871 | 0.3482 | −18.4% |

TAN 在 ECL 上取得显著提升，验证了"频率条件化的局部归一化"对多变量周期数据的有效性。

## 五、代码结构

```
layers/freq_features.py     # 共享频率特征提取器（rFFT + 三角带通滤波）
utils/adaptive_norm.py      # TAN 实现（forward_in / forward_out）
layers/moe_ffn.py           # FIR-MoE 路由器扩展
models/timer_xl.py          # 主模型集成
scripts/jobs/               # 实验脚本
```

## 六、下一步

1. 在 9 个数据集 × 4 个预测长度上完成全量 benchmark
2. 探索频谱一致性正则 $\mathcal{L}_{\text{spec}} = \lambda_s \sum_k \left| |X_{\text{pred}}[k]|^2 - |X_{\text{true}}[k]|^2 \right|$ 作为与 TAN 互补的输出侧频率约束
3. 消融实验与可解释性可视化
4. 论文撰写
