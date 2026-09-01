# 激活函数：Sigmoid → ReLU → GELU → SwiGLU（DL 八股 02）

> **更新时间**：2026-08-31

> **标签**：激活函数、ReLU、GELU、SwiGLU、面试八股

> **一句话**：激活函数提供非线性（否则多层网络等价于单层线性变换）；Sigmoid/Tanh 因饱和导致梯度消失被 ReLU 取代，Transformer 时代用更平滑的 GELU，现代 LLM 的 FFN 统一用 **SwiGLU**（门控 + Swish），并把中间维度调成约 $\frac83 d$ 以保持参数量。

> **关联阅读**：[[/docs/dl/gradient-vanishing-exploding-residual.md]]、[[/docs/llm/transformer-principle.md]]

---

## 1. 为什么必须有激活函数

若没有非线性，$W_2(W_1x)=(W_2W_1)x$，无论多少层都还是一个线性映射，表达能力等于单层，无法拟合非线性决策边界。激活函数 + 足够宽度构成**通用逼近**能力。

---

## 2. 经典激活函数对比

| 函数 | 表达式 | 导数特点 | 优点 | 缺点 |
|------|--------|----------|------|------|
| Sigmoid | $\frac1{1+e^{-x}}$ | $\sigma(1-\sigma)\le 0.25$ | 输出 (0,1) 可作概率/门控 | **梯度消失**、输出非零均值、含 exp 较贵 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2\le 1$ | 零均值，比 sigmoid 好 | 仍饱和梯度消失 |
| **ReLU** | $\max(0,x)$ | 正区间恒为 1 | 不饱和、计算极快、稀疏激活 | **神经元死亡**、非零均值、0 点不可导（工程取 0） |
| LeakyReLU | $\max(\alpha x, x)$，$\alpha$=0.01 | 负区间 $\alpha$ | 缓解死亡 | 多一个超参 |
| PReLU | $\alpha$ 可学 | — | 更灵活 | 略增参数、易过拟合 |
| ELU | $x$ / $\alpha(e^x-1)$ | 负区间平滑 | 输出接近零均值 | 含 exp，慢 |
| **GELU** | $x\cdot\Phi(x)$ | 平滑非单调 | Transformer 默认（BERT/GPT） | 比 ReLU 贵（有 tanh 近似） |
| Swish/SiLU | $x\cdot\sigma(\beta x)$ | 平滑非单调 | 与 GELU 接近，深网表现好 | 同上 |
| Softplus | $\log(1+e^x)$ | 处处光滑 | ReLU 的光滑版 | 慢、无稀疏性 |
| Mish | $x\tanh(\mathrm{softplus}(x))$ | 平滑 | 视觉任务偶有增益 | 更贵 |

### 2.1 梯度消失的量化直觉

Sigmoid 导数最大 0.25，$L$ 层链式相乘上界 $0.25^L$：10 层就是 $10^{-6}$ 量级 → 底层几乎收不到梯度。ReLU 正区间导数恒为 1，链式相乘不衰减，这是深网可训练的关键之一。

### 2.2 ReLU 神经元死亡

若某神经元的输入长期为负（例如被一次大梯度更新把 bias 推得很负），其输出恒 0、梯度恒 0，**永久失活**。缓解：LeakyReLU/ELU/GELU、更小学习率、合适初始化（He 初始化）、加 BN。

> 面试高频：**ReLU 不是处处可导，为什么能用？** → 只在 $x=0$ 一点不可导，测度为零；实现上取次梯度（PyTorch 取 0），不影响 SGD 收敛。

---

## 3. GELU：Transformer 时代的默认

$$\mathrm{GELU}(x)=x\cdot\Phi(x)=x\cdot\frac12\Big[1+\mathrm{erf}\big(\tfrac{x}{\sqrt2}\big)\Big]$$

近似式（早期实现常用）：

$$\mathrm{GELU}(x)\approx 0.5x\Big(1+\tanh\big[\sqrt{2/\pi}\,(x+0.044715x^3)\big]\Big)$$

**直觉**：ReLU 是"硬门控"（按 $x>0$ 硬性保留），GELU 是"按输入大小的概率软门控"——$x$ 越大越可能被保留。平滑性带来更好的梯度性质，负区间保留少量信息（非单调），BERT / GPT-2 / GPT-3 / ViT 都用它。

---

## 4. SwiGLU：现代 LLM 的 FFN 标配

标准 FFN：$\mathrm{FFN}(x)=W_2\,\phi(W_1x)$，隐藏维通常 $4d$。

**GLU 家族**引入门控：$\mathrm{GLU}(x)=(W_1x)\odot\sigma(W_gx)$。把门控激活换成 Swish 得 **SwiGLU**：

$$\mathrm{FFN}_{\text{SwiGLU}}(x) = W_2\Big(\mathrm{Swish}(W_{\text{gate}}x)\odot (W_{\text{up}}x)\Big)$$

- **三个权重矩阵**（gate、up、down），比标准 FFN 多一个 → 为保持参数量/FLOPs 不变，中间维取 $\frac23\times 4d=\frac83 d$（LLaMA 实际取接近 $\frac83 d$ 并对齐到 256 的倍数，如 LLaMA-7B：$d=4096$，intermediate=11008）；
- Shazeer 在 *GLU Variants Improve Transformer*（arXiv:2002.05202）中实验证明 GLU 变体（GEGLU/SwiGLU）在同参数量下困惑度更低，论文原话是这些结构的成功"we offer no explanation"（属于经验性发现）；
- LLaMA / Qwen / Mistral / DeepSeek 系列均采用 SwiGLU。

> 面试高频：**LLaMA 的 FFN 为什么是 11008 而不是 16384？** → SwiGLU 有三个矩阵，为保持总参数量与 $4d$ 的两矩阵 FFN 相当，中间维缩到约 $\frac83 d \approx 10922$，再对齐硬件友好的倍数得 11008。

---

## 5. 输出层激活

| 任务 | 输出激活 | 配套损失 |
|------|----------|----------|
| 二分类 | Sigmoid | BCE（实现用 `BCEWithLogits` 更稳） |
| 多分类（互斥） | Softmax | CE（实现里 softmax 融合进 CE） |
| 多标签 | 每维 Sigmoid | BCE |
| 回归 | 无（线性） | MSE / MAE / Huber |

**Softmax 数值稳定**：先减去最大值 $\mathrm{softmax}(x_i)=\frac{e^{x_i-\max x}}{\sum_j e^{x_j-\max x}}$，否则 $e^{x}$ 溢出。这是手撕代码高频考点。

---

## 6. 手撕代码

```python
import torch, torch.nn as nn, torch.nn.functional as F

def stable_softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values     # 防溢出
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)

def gelu_tanh(x):                                    # tanh 近似版 GELU
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

class SwiGLUFFN(nn.Module):
    """LLaMA 风格 FFN：hidden ≈ 8/3 * d，三个线性层，无 bias"""
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or int(8 * d / 3)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

---

## 7. 面试高频问题速查

1. **为什么需要激活函数？** → 没有非线性，多层等价单层线性映射。
2. **Sigmoid 的问题？** → 梯度最大 0.25 导致梯度消失、输出非零均值使梯度更新呈锯齿、exp 计算贵。
3. **ReLU 为什么缓解梯度消失？** → 正区间导数恒 1，链式乘积不衰减；同时带来稀疏激活与极快计算。
4. **ReLU 死亡是什么、怎么解决？** → 输入长期为负则梯度永远为 0；用 LeakyReLU/ELU/GELU、调小 lr、He 初始化、加归一化。
5. **GELU 与 ReLU 的区别？** → GELU 是按标准正态 CDF 的软门控、平滑非单调、负区间保留信息，Transformer 默认。
6. **SwiGLU 是什么？为什么用？** → Swish 门控的 GLU 变体，三矩阵结构，同等参数下效果更好，现代 LLM 标配。
7. **SwiGLU 中间维为什么是 8/3 d？** → 补偿多出的门控矩阵，使参数量/FLOPs 与 $4d$ 的传统 FFN 对齐。
8. **零均值输出为什么重要？** → 非零均值使同层权重梯度符号一致，产生锯齿式更新、收敛变慢（Tanh 优于 Sigmoid 的原因之一）。
9. **Softmax 怎么防溢出？** → 减去最大 logit，再做指数与归一化。
10. **哪里还在用 Sigmoid/Tanh？** → 输出层概率、LSTM/GRU 的门控、注意力/MoE 的门控打分。
11. **激活函数需要归一化配合吗？** → 是，BN/LN 把输入拉回激活敏感区，避免进入饱和/死区。

---

## 参考

- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv:1606.08415
- Ramachandran et al., *Searching for Activation Functions (Swish)*, arXiv:1710.05941
- Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, arXiv:2302.13971
