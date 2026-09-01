# 归一化：BatchNorm / LayerNorm / RMSNorm（DL 八股 01）

> **更新时间**：2026-08-31

> **标签**：BatchNorm、LayerNorm、RMSNorm、Pre-LN、面试八股

> **一句话**：归一化把每层输入拉回稳定分布，让梯度更好传播、允许更大学习率；BN 沿 batch 维统计（依赖 batch、训练/推理行为不同），LN 沿单样本特征维统计（变长序列友好，是 Transformer 的选择），RMSNorm 进一步去掉均值项只做缩放，是现代 LLM 的默认。

> **关联阅读**：[[/docs/dl/gradient-vanishing-exploding-residual.md]]、[[/docs/llm/transformer-principle.md]]

---

## 1. 为什么需要归一化

深层网络里，前层参数一变，后层输入分布就漂移（原论文称 **Internal Covariate Shift, ICS**），导致：饱和激活进死区、梯度尺度失控、学习率必须调很小、初始化极其敏感。

> **重要更新**：后续研究（Santurkar et al., *How Does Batch Normalization Help Optimization?*, NeurIPS 2018）证明 BN 的收益**主要不是**消除 ICS，而是**让损失曲面更平滑（Lipschitz 性质更好）**，从而允许更大学习率、更快收敛。面试时能补这一句，明显区别于纯背八股。

---

## 2. BatchNorm

对一个 mini-batch，在**每个通道/特征维**上沿 batch（以及 CNN 的 H、W）求统计量：

$$\hat x = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat x+\beta$$

- $\gamma,\beta$ 是**可学习**的缩放与平移，保证归一化不会削弱表达能力（极端情况下可学回恒等映射）；
- CNN 中 BN 参数量 = $2C$（每通道一对），统计维度是 $(N,H,W)$；
- **训练**用当前 batch 统计量；**推理**用训练期的**移动平均**（running mean/var）→ 训练/推理行为不一致，这是 BN 所有坑的根源。

### 2.1 BN 的收益

平滑损失曲面、允许更大 lr、降低初始化敏感度、有轻微正则效果（batch 噪声）、缓解梯度消失。

### 2.2 BN 的坑（高频）

| 问题 | 原因 / 解法 |
|------|------------|
| 小 batch 效果差（检测/分割常见 bs=2） | 统计量噪声大 → 用 GroupNorm、SyncBN（跨卡同步）、或冻结 BN |
| 变长序列（NLP）不适用 | 同一时间步的有效样本数不同、padding 污染统计量 → 用 LN |
| 训练/推理不一致 | 必须 `model.eval()`；小数据微调时可冻结 BN 统计量 |
| 与 dropout 顺序敏感 | 常见结论：Conv → BN → 激活 → (Dropout)；BN 之后 dropout 会扰动统计量（"variance shift"） |
| 在线学习/batch=1 | 不可用 |
| 有 BN 时 conv 的 bias 冗余 | BN 会减均值，前面的 bias 被吸收 → `bias=False` |

> 面试高频：**BN 放在激活前还是后？** → 原论文放在激活**前**（Conv-BN-ReLU），也是主流实现；有实验表明 Conv-ReLU-BN 在某些任务上略好，属于经验性问题，答"原论文与主流是激活前，并说明理由"即可。

---

## 3. LayerNorm

对**单个样本**在特征维（最后一维或若干维）上做归一化：

$$\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\frac1d\sum_i x_i,\ \sigma^2=\frac1d\sum_i (x_i-\mu)^2$$

- **与 batch 无关** → 训练/推理完全一致、支持 batch=1、支持变长序列；
- Transformer / RNN 的标准选择。

> 面试高频：**NLP 为什么用 LN 不用 BN？**
> 1. 序列变长 + padding → 沿 batch 维统计不稳定甚至错误；
> 2. 同一位置不同句子的 token 语义无关，"跨样本同位置"做归一化没有语义基础；
> 3. 推理常是 batch=1 / 流式，BN 的 running stats 与训练分布不匹配；
> 4. 自回归解码逐 token 生成，序列长度动态变化，LN 天然兼容。

### 3.1 归一化家族对比

| 方法 | 统计维度（CNN 记号 N,C,H,W） | 依赖 batch | 典型场景 |
|------|------------------------------|-----------|----------|
| BatchNorm | (N,H,W) 每通道 | 是 | CNN 大 batch 分类 |
| LayerNorm | (C,H,W) 每样本 | 否 | Transformer、RNN |
| InstanceNorm | (H,W) 每样本每通道 | 否 | 风格迁移（去实例风格） |
| GroupNorm | 每样本、通道分 G 组 | 否 | 检测/分割小 batch（G=32 常用） |
| RMSNorm | 每样本特征维，仅缩放 | 否 | 现代 LLM |
| WeightNorm | 对权重而非激活 | 否 | 少用 |

---

## 4. RMSNorm

去掉减均值（re-centering）操作，只做均方根缩放：

$$\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma$$

- **少了求均值和减均值两步**，也少了 $\beta$ 参数 → 计算量与显存都省，速度提升可观（LN 在 LLM 中占不可忽略的 kernel 时间）；
- 实验表明**效果与 LN 相当**（Zhang & Sennrich, 2019）：稳定训练的关键是**re-scaling 不变性**，而非 re-centering；
- LLaMA、Qwen、GLM、DeepSeek 等主流开源 LLM 全部采用 RMSNorm。

---

## 5. Post-LN vs Pre-LN（Transformer 必考）

| 结构 | 公式 | 特点 |
|------|------|------|
| **Post-LN**（原论文 2017） | $x \leftarrow \mathrm{LN}(x + \mathrm{Sublayer}(x))$ | 表达能力略好，但深层梯度不稳，**必须 warmup**，深度上不去 |
| **Pre-LN**（现代主流） | $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LN}(x))$ | 残差主干是干净的恒等路径，梯度直达底层，训练稳、可堆很深、对 warmup 不敏感 |

进一步演进：
- **Sandwich / Peri-LN**：在子层输出再加一个 LN，进一步压制激活范数增长；
- **DeepNorm**（微软）：残差加权 + 特定初始化，稳定训练千层 Transformer；
- **QK-Norm**：对 Q/K 单独做归一化，抑制注意力 logits 爆炸，被多个近期大模型采用。

> 面试高频：**为什么 Pre-LN 更稳？** → Post-LN 的每个残差块输出都被 LN 重新缩放，反向传播时梯度要穿过多层 LN 的雅可比，深层容易衰减/放大；Pre-LN 的恒等分支不经过 LN，梯度可无衰减地流回浅层（与 ResNet 的 identity mapping 同理）。

---

## 6. 手撕代码

```python
import torch, torch.nn as nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, x):                     # x: (N, d)
        if self.training:
            mu = x.mean(0)
            var = x.var(0, unbiased=False)    # 归一化用有偏方差
            with torch.no_grad():             # running stats 用无偏方差更新
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu)
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * x.var(0, unbiased=True))
        else:
            mu, var = self.running_mean, self.running_var
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # 关键：均方根在 fp32 下计算，避免低精度溢出
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight
```

---

## 7. 面试高频问题速查

1. **BN 的公式与可学习参数？** → 减均值除标准差后 $\gamma$ 缩放 $\beta$ 平移；$\gamma,\beta$ 保证不损失表达能力。
2. **BN 训练和推理有什么不同？** → 训练用 batch 统计量，推理用移动平均；忘记 `eval()` 是经典事故。
3. **BN 为什么有效？** → 平滑损失曲面、允许大 lr（新解释）；原论文归因于减少 ICS，已被后续工作修正。
4. **BN 为什么怕小 batch？** → 统计量估计噪声大 → GroupNorm / SyncBN。
5. **NLP 为什么用 LN？** → 变长序列、padding、batch=1 推理、跨样本同位置无语义可比性。
6. **LN 与 BN 的统计维度差别？** → LN 在单样本特征维，BN 在 batch（及空间）维，二者"转置"关系。
7. **RMSNorm 省了什么？为什么还有效？** → 省减均值与 $\beta$；稳定训练主要靠 re-scaling 不变性。
8. **Pre-LN 与 Post-LN 区别？** → LN 在残差内 vs 残差外；Pre-LN 训练稳、可深、warmup 需求低，是现代默认。
9. **GroupNorm 什么时候用？** → 检测/分割等小 batch 场景，G 通常取 32。
10. **BN 层能和卷积融合吗？** → 推理时可把 BN 折进卷积权重与偏置（BN folding），是部署常规优化。
11. **有 BN 还需要 dropout 吗？** → BN 已带轻微正则，二者叠加可能冲突；现代 CNN 多用 BN + 数据增强而少用 dropout。
12. **归一化能替代好的初始化吗？** → 能大幅降低对初始化的敏感度，但极深网络仍需配合合适初始化/残差（见 [[/docs/dl/gradient-vanishing-exploding-residual.md]]）。

---

## 参考

- Ioffe & Szegedy, *Batch Normalization*, arXiv:1502.03167
- Ba et al., *Layer Normalization*, arXiv:1607.06450
- Wu & He, *Group Normalization*, arXiv:1803.08494
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv:1910.07467
- Santurkar et al., *How Does Batch Normalization Help Optimization?*, arXiv:1805.11604
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv:2002.04745
