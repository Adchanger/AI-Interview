# 损失函数：交叉熵、Focal Loss、标签平滑与对比损失（DL 八股 06）

> **更新时间**：2026-08-31

> **标签**：损失函数、交叉熵、FocalLoss、标签平滑、对比学习、面试八股

> **一句话**：损失函数决定"模型往哪儿优化"——分类用交叉熵（等价 KL/最大似然），不平衡与难例用 Focal Loss，过度自信用标签平滑，表示学习用 InfoNCE 类对比损失，回归按对离群点的容忍度在 MSE/MAE/Huber 之间选。

> **关联阅读**：[[/docs/ml/logistic-regression.md]]、[[/docs/ml/feature-engineering-and-imbalance.md]]、[[/docs/llm/vlm-evolution.md]]

---

## 1. 分类损失

### 1.1 交叉熵（Cross Entropy）

$$\mathcal{L}_{CE} = -\sum_{k} y_k\log \hat p_k \;\xrightarrow{\text{one-hot}}\; -\log \hat p_{y}$$

**三个等价视角**（面试可任选切入）：
1. **最大似然**：多项分布的负对数似然；
2. **KL 散度**：$\mathrm{KL}(p\|\hat p) = H(p,\hat p) - H(p)$，真实分布固定时最小化 CE ⇔ 最小化 KL；
3. **信息论**：用 $\hat p$ 编码来自 $p$ 的样本所需的额外比特数。

**梯度极简**：softmax + CE 对 logits 的梯度是 $\hat p - y$ —— 这就是为什么框架把两者融合实现（`CrossEntropyLoss` 输入 logits，不要自己先做 softmax）。

> 面试高频：**为什么分类用 CE 而不是 MSE？** → ① 梯度不被激活饱和抹掉（MSE 梯度含 $\sigma'$）；② 与伯努利/多项似然一致；③ sigmoid/softmax + CE 对参数是凸的（单层情形）。详见 [[/docs/ml/logistic-regression.md]]。

### 1.2 数值稳定：LogSumExp

$$\log\sum_j e^{z_j} = \max_j z_j + \log\sum_j e^{z_j-\max_j z_j}$$

`log_softmax` 与 `BCEWithLogitsLoss` 内部都是这么做的，手撕时必须写出减最大值这一步。

### 1.3 Focal Loss

$$\mathcal{L}_{FL} = -\alpha_t(1-p_t)^{\gamma}\log p_t$$

- $p_t$ 是模型对**真实类别**的预测概率；
- $(1-p_t)^\gamma$ 是**调制因子**：易分样本（$p_t\to1$）权重趋 0，把梯度让给难例；论文默认 $\gamma=2,\ \alpha=0.25$；
- 解决的是**极端前景/背景不平衡**（RetinaNet 中 1:1000 量级）；
- 注意：$\alpha$ 平衡类别频率，$\gamma$ 平衡难易，**两者作用不同**，是常考细节；$\gamma=0$ 时退化为加权 CE。

### 1.4 标签平滑（Label Smoothing）

$$y_k^{LS} = (1-\epsilon)y_k + \frac{\epsilon}{K}$$

- 典型 $\epsilon=0.1$（Inception-v3、Transformer 原论文均用）；
- **为什么有效**：防止模型把正确类 logit 推向无穷（过度自信）、缩小类内特征簇的过度收缩、提升**校准性**与鲁棒性；
- 代价：会**损害知识蒸馏**（Müller et al. 指出平滑会"擦除"类间相似度信息，教师模型不宜用 LS）；检索/度量学习任务也需谨慎。

### 1.5 其他分类相关损失

| 损失 | 场景 |
|------|------|
| BCE（多标签） | 每类独立 sigmoid，标签非互斥 |
| Hinge / squared hinge | SVM 系，见 [[/docs/ml/svm.md]] |
| Dice / Tversky | 分割中前景占比极小 |
| ArcFace / CosFace | 人脸识别，在角度空间加 margin 提升判别性 |
| KL 蒸馏损失 | $T^2\cdot\mathrm{KL}(p_T^{teacher}\|p_T^{student})$，温度 $T$ 软化分布 |

---

## 2. 回归损失

| 损失 | 表达式 | 特点 |
|------|--------|------|
| MSE (L2) | $(y-\hat y)^2$ | 处处可导、对离群点**敏感**（误差平方放大）；最优解是条件均值 |
| MAE (L1) | $|y-\hat y|$ | 鲁棒、0 点不可导；最优解是条件**中位数** |
| **Huber / Smooth L1** | 小误差用平方、大误差转线性 | 兼顾两者，检测框回归标配（$\delta$ 是切换阈值） |
| LogCosh | $\log\cosh(y-\hat y)$ | 光滑版 Huber |
| Quantile loss | $\max(\tau e, (\tau-1)e)$ | 分位数回归，做预测区间 |
| IoU / GIoU / DIoU / CIoU | 直接优化框重叠 | 目标检测，比 L1 更贴合评测指标 |

> 面试高频：**为什么 MSE 的最优预测是均值、MAE 是中位数？** → 对 $\mathbb{E}[(y-c)^2]$ 求导得 $c=\mathbb{E}[y]$；对 $\mathbb{E}|y-c|$ 求次导为 0 得 $c$ 是中位数。所以数据有重尾离群时 MAE/Huber 更稳。

---

## 3. 对比学习与表示学习损失

### 3.1 InfoNCE

$$\mathcal{L} = -\log\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}{\sum_{j}\exp(\mathrm{sim}(z_i,z_j)/\tau)}$$

- 本质是**在一堆负样本中做 softmax 分类**（正样本为正确类）；
- **温度 $\tau$** 是关键超参：$\tau$ 小 → 分布更尖锐、更关注难负样本、梯度更集中但训练不稳；$\tau$ 大 → 更平滑、对负样本区分弱；
- 负样本数量越多下界越紧（SimCLR 靠大 batch，MoCo 靠队列 + 动量编码器）；
- **CLIP** 用对称的图文 InfoNCE（image→text 与 text→image 各一次），温度**可学习**。

### 3.2 其他

| 损失 | 说明 |
|------|------|
| Triplet Loss | $\max(0, d(a,p)-d(a,n)+m)$，需难例挖掘否则收敛慢 |
| Contrastive (Siamese) | 正样本拉近、负样本推开至 margin |
| BYOL / SimSiam | 无负样本，靠预测头 + stop-gradient 防坍缩 |
| VICReg / Barlow Twins | 用方差-不变-协方差正则显式防维度坍缩 |

---

## 4. LLM 相关损失（衔接大模型部分）

| 阶段 | 损失 |
|------|------|
| 预训练 | 因果语言建模的 token 级交叉熵（next-token prediction）；perplexity = $e^{\mathcal{L}}$ |
| SFT | 同上，但**只对 answer 部分计算 loss**（prompt token 的 label 置 -100），见 [[/docs/llm/sft-lora-peft.md]] |
| 奖励模型 | pairwise ranking loss：$-\log\sigma(r(x,y_w)-r(x,y_l))$ |
| DPO | $-\log\sigma\big(\beta\log\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\big)$，见 [[/docs/llm/rlhf-ppo-dpo.md]] |
| PPO | 截断重要性采样目标 + KL 惩罚 + 价值函数损失 |
| 稳定性辅助项 | z-loss（约束 logits 范数）、MoE 负载均衡辅助损失，见 [[/docs/llm/moe-mixture-of-experts.md]] |

---

## 5. 手撕代码

```python
import torch, torch.nn as nn, torch.nn.functional as F

def cross_entropy_manual(logits, target, smoothing=0.0):
    """logits: (N, K), target: (N,) 类别索引；含标签平滑"""
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # log_softmax
    n, k = logits.shape
    nll = -logp[torch.arange(n), target]
    if smoothing > 0:
        smooth = -logp.mean(dim=-1)
        return ((1 - smoothing) * nll + smoothing * smooth).mean()
    return nll.mean()

def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    """二分类 Focal Loss，logits/target 形状均为 (N,)"""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t).pow(gamma) * ce).mean()

def info_nce(z1, z2, tau=0.07):
    """z1,z2: (N,d) 一一对应的正样本对（如 CLIP 的图/文特征）"""
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau                 # (N,N)，对角为正样本
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))
```

---

## 6. 面试高频问题速查

1. **交叉熵与 KL 散度的关系？** → $CE = H(p) + KL(p\|\hat p)$，真实分布固定时二者等价。
2. **softmax+CE 对 logits 的梯度？** → $\hat p - y$，简洁且无饱和问题。
3. **为什么框架把 softmax 融进 CE？** → 数值稳定（LogSumExp）+ 少一次冗余计算。
4. **Focal Loss 的两个超参分别管什么？** → $\gamma$ 调难易样本权重，$\alpha$ 调类别不平衡。
5. **Focal Loss 一定优于加权 CE 吗？** → 不一定；对标注噪声更敏感（噪声样本正是"难例"），需实测。
6. **标签平滑为什么有效、什么时候别用？** → 抑制过度自信、改善校准；蒸馏的教师、度量/检索任务慎用。
7. **MSE 与 MAE 的最优解？** → 条件均值 vs 条件中位数；离群点多用 MAE/Huber。
8. **Huber 的切换点怎么定？** → $\delta$ 视噪声尺度，检测中 Smooth L1 取 1；可按残差分位数调。
9. **InfoNCE 的温度作用？** → 控制分布尖锐度与对难负样本的关注度；过小不稳、过大区分弱。
10. **对比学习为什么需要大量负样本？** → 互信息下界随负样本数变紧；MoCo 用队列、SimCLR 用大 batch 解决。
11. **多标签为什么不能用 softmax？** → softmax 强制概率和为 1（互斥假设），多标签应逐类 sigmoid + BCE。
12. **SFT 为什么只对回答算 loss？** → 目标是学"给定指令如何回答"，对 prompt 计算 loss 会让模型去拟合用户输入分布，浪费容量甚至损害指令遵循。

---

## 参考

- Lin et al., *Focal Loss for Dense Object Detection*, arXiv:1708.02002
- Szegedy et al., *Rethinking the Inception Architecture*（label smoothing）, arXiv:1512.00567
- Müller et al., *When Does Label Smoothing Help?*, arXiv:1906.02629
- Oord et al., *Representation Learning with Contrastive Predictive Coding (InfoNCE)*, arXiv:1807.03748
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, arXiv:2103.00020
- Chen et al., *A Simple Framework for Contrastive Learning (SimCLR)*, arXiv:2002.05709
