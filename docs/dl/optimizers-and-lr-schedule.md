# 优化器与学习率策略：SGD / Adam / AdamW / warmup（DL 八股 03）

> **更新时间**：2026-08-31

> **标签**：优化器、Adam、AdamW、warmup、学习率调度、面试八股

> **一句话**：SGD+Momentum 靠动量抹平震荡、泛化好但需精调；Adam 用一阶/二阶动量做逐参数自适应步长、收敛快；AdamW 把权重衰减从梯度里解耦，是当前大模型训练的默认，配合 **warmup + cosine decay** 与梯度裁剪。

> **关联阅读**：[[/docs/ml/bias-variance-and-regularization.md]]、[[/docs/engineering/distributed-training.md]]

---

## 1. 从 SGD 到自适应优化器

### 1.1 SGD 与 Momentum

$$\text{SGD}:\ \theta \leftarrow \theta - \eta g_t$$

$$\text{Momentum}:\ v_t = \beta v_{t-1} + g_t,\quad \theta\leftarrow\theta-\eta v_t\ \ (\beta{=}0.9)$$

动量把历史梯度做指数加权平均：在**一致方向上累积加速**、在**震荡方向上相互抵消**，等效"小球带惯性滚下山谷"，显著改善病态曲率（ill-conditioned）下的锯齿震荡。

**Nesterov（NAG）**：在"预测位置"求梯度 $g_t=\nabla L(\theta-\eta\beta v_{t-1})$，相当于提前刹车，理论收敛更好。

### 1.2 AdaGrad → RMSProp → Adam

| 优化器 | 关键式 | 问题/改进 |
|--------|--------|----------|
| AdaGrad | $\theta\!-\!\frac{\eta}{\sqrt{\sum_{\tau\le t} g_\tau^2}+\epsilon}g_t$ | 分母单调累加 → 学习率**衰减到 0**，训不久；适合稀疏特征 |
| RMSProp | 用 EMA 代替累加：$s_t=\rho s_{t-1}+(1-\rho)g_t^2$ | 解决 AdaGrad 学习率消失 |
| **Adam** | 一阶动量 + 二阶动量 + 偏差修正 | RMSProp + Momentum 的合体，工业默认 |

**Adam 完整更新式**：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$$

$$\theta\leftarrow\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

默认 $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$；LLM 训练常用 $\beta_2=0.95$（对梯度尖峰更鲁棒，GPT-3、LLaMA 等采用）。

> 面试高频：**偏差修正是干什么的？** → $m_0=v_0=0$，初期 EMA 被 0 拉低（$\mathbb{E}[m_t]\approx(1-\beta_1^t)\mathbb{E}[g]$），除以 $1-\beta^t$ 还原无偏估计。否则前几百步实际步长被严重低估。$\beta_2=0.999$ 时二阶矩需要约上千步才"预热"完成——这也是需要 warmup 的原因之一。

### 1.3 Adam 的优缺点

**优点**：逐参数自适应（稀疏梯度友好）、对学习率不敏感、收敛快、几乎不用调参。

**缺点**：
- **泛化常略逊于 SGDM**（尤其 CV 分类任务，ResNet 系列仍偏爱 SGDM）；
- 显存开销：**每参数额外 2 个 fp32 状态**（$m,v$）；
- 早期可能不收敛（Reddi et al. 指出二阶矩非单调导致的问题）→ **AMSGrad**；
- 对 $\epsilon$、$\beta_2$ 敏感于数值精度（bf16 训练需状态用 fp32）。

---

## 2. AdamW：解耦权重衰减

L2 正则写进损失时，梯度里多一项 $\lambda\theta$，它会**被 $\sqrt{\hat v}$ 缩放**：梯度大的参数（$\hat v$ 大）实际受到的衰减更小，与"均匀收缩权重"的初衷不符。

**AdamW** 把衰减直接作用在参数上：

$$\theta \leftarrow \theta - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta\Big)$$

（实现上写作 $\theta\leftarrow(1-\eta\lambda)\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$）

- 在 SGD 下 L2 与 weight decay 等价；在 Adam 下**不等价**，所以 AdamW 才是正确写法；
- LLM 训练标配：$\lambda=0.1$（GPT-3/LLaMA 量级），且 **bias 与 Norm 层参数不做 decay**。

### 2.1 大模型时代的新优化器

| 优化器 | 一句话 |
|--------|--------|
| **Adafactor** | 用行列因子分解近似二阶矩，显存从 $O(d^2)$ 降到 $O(d)$；T5 使用 |
| **LAMB / LARS** | 逐层自适应信任比，支持超大 batch（BERT 76 分钟训练） |
| **8-bit Adam**（bitsandbytes） | 优化器状态量化到 8bit，显存省约 3/4 |
| **Lion** | 只保留一阶动量 + 符号更新，显存更省，部分任务优于 AdamW |
| **Sophia / 二阶方法** | 近似 Hessian 对角信息，宣称预训练加速；工程采用仍有限 |
| **Muon**（2025 起流行） | 对矩阵参数做正交化/牛顿-舒尔茨迭代更新，在部分 LLM 预训练里显示更好的 loss-算力曲线 |

> 回答策略：主线答 AdamW，再补一句"显存受限用 8-bit Adam / Adafactor，近期有 Lion、Muon 等新方案在预训练上被验证"，既稳又有信息量。

---

## 3. 学习率调度

### 3.1 Warmup（必考）

前 $T_w$ 步把 lr 从 ~0 线性升到峰值。**为什么必须**：

1. **Adam 二阶矩估计不准**：训练初期 $\hat v$ 样本太少、方差极大，直接用峰值 lr 会产生离谱的大步长（RAdam 论文的核心论证）；
2. **初期梯度方向噪声大**：随机初始化下大步更新会把模型推到坏区域，后续难恢复；
3. **Post-LN Transformer 梯度不稳**：不 warmup 直接发散（Xiong et al. 2020）；
4. **大 batch 训练**：等效 lr 很大，需要缓慢升温（Goyal et al. 的 linear scaling + warmup）。

典型设置：总步数的 1%~2%（几百到几千步）。

### 3.2 常见衰减策略

| 策略 | 说明 |
|------|------|
| Step decay | 每 N epoch 乘 0.1，CV 经典 |
| **Cosine decay** | $\eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})(1+\cos\frac{\pi t}{T})$，LLM 默认，末端常降到峰值的 10% |
| Linear decay | 简单、BERT 微调常用 |
| Inverse sqrt | $\eta\propto 1/\sqrt{t}$，原始 Transformer 论文用（配 warmup） |
| **WSD / trapezoid** | Warmup-Stable-Decay：恒定 lr 长期训练 + 末端快速衰减，便于中途续训与数据配比调整（MiniCPM 等采用） |
| Cyclical / 余弦重启 | SGDR，跳出局部区域 |
| ReduceLROnPlateau | 指标不降就降 lr，小任务实用 |

### 3.3 其他训练稳定手段

- **梯度裁剪**：`clip_grad_norm_(params, 1.0)` 是 LLM 标配，压制 loss spike；
- **梯度累积**：小显存模拟大 batch，注意 loss 要按累积步数取平均；
- **学习率与 batch 的关系**：线性缩放法则（batch ×k → lr ×k，配 warmup）；平方根缩放在 Adam 下也常被采用；
- **EMA 权重**：对参数做指数滑动平均，评测更稳（扩散模型、部分 LLM SFT 使用）；
- **loss spike 处理**：跳过异常 batch、回滚到上个 checkpoint 并跳过数据、降低 $\beta_2$、加 QK-Norm/z-loss。

---

## 4. 手撕代码：Adam / AdamW

```python
import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)      # 偏差修正
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * p)
            #                                          ↑ 解耦的权重衰减（AdamW 关键）

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def lr_at(step, warmup, total, peak, min_ratio=0.1):
    """线性 warmup + cosine 衰减"""
    import math
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
```

---

## 5. 面试高频问题速查

1. **动量为什么加速收敛？** → 一致方向累积、震荡方向抵消，缓解病态曲率下的锯齿。
2. **AdaGrad 的缺陷？** → 二阶累加单调递增使 lr 趋 0；RMSProp 用 EMA 修正。
3. **Adam 的完整更新公式？** → 一阶/二阶 EMA + 偏差修正 + 除以 $\sqrt{\hat v}$。
4. **偏差修正为什么必要？** → 零初始化让 EMA 前期偏小，除以 $1-\beta^t$ 得无偏估计。
5. **Adam 与 AdamW 的区别？** → weight decay 是否被自适应学习率缩放；AdamW 解耦，效果更好。
6. **Adam 的显存开销？** → 每参数 2 个 fp32 状态（8 bytes），混合精度全参训练约 16 bytes/param，见 [[/docs/engineering/distributed-training.md]]。
7. **为什么 SGDM 在 CV 上泛化更好？** → 一种解释是 Adam 更易收敛到尖锐极小点、自适应缩放破坏了梯度噪声的正则效应；工程结论多于理论定论。
8. **为什么必须 warmup？** → 二阶矩估计不准 + 初期梯度噪声大 + Post-LN 不稳 + 大 batch 等效 lr 高。
9. **cosine 衰减为什么好用？** → 前期保持较大 lr 探索、后期平滑降到很小便于收敛，无需手调阶梯点。
10. **WSD 调度的优势？** → 稳定段可无限续训、随时决定衰减点，便于中途改数据配比、做 continue pretrain。
11. **梯度裁剪的作用？** → 限制梯度范数，防 loss spike/爆炸，LLM 常设 1.0。
12. **batch 变大 lr 怎么调？** → 线性缩放（配 warmup）或平方根缩放，需实测；同时注意 batch 太大会损失泛化。

---

## 参考

- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, arXiv:1412.6980
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW)*, arXiv:1711.05101
- Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond (RAdam)*, arXiv:1908.03265
- Goyal et al., *Accurate, Large Minibatch SGD*, arXiv:1706.02677
- Chen et al., *Symbolic Discovery of Optimization Algorithms (Lion)*, arXiv:2302.06675
- Hu et al., *MiniCPM*（WSD 学习率调度）, arXiv:2404.06395
