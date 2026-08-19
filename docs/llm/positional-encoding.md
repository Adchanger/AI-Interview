# 大模型位置编码（LLM 八股 02）

> **更新时间**：2026-08-19

> **标签**：位置编码、RoPE、LLM、Transformer

> **一句话**：自注意力对 token 顺序完全"无感"，位置编码（Positional Encoding）负责注入位置信息；从绝对位置编码（正余弦、可学习）演进到相对位置编码（RoPE、ALiBi），是现代 LLM（LLaMA、Qwen 等）长度外推能力的基石。

> **关联阅读**：[[/docs/llm/transformer-principle.md]]（Transformer 原理与自注意力基础）

---

## 1. 背景：为什么需要位置编码

自注意力是**置换不变**（permutation invariant）的：把输入 token 顺序打乱，注意力计算出的结果不变（只是行跟着换）。因为注意力只看"两两之间的相似度"，不感知"谁在前谁在后"：

- "我打你"和"你打我"，在注意力眼里是同一个 bag of words；
- 没有位置信息，模型无法建模语序、语法、时序逻辑。

所以必须在输入侧注入位置信息，让模型知道"每个 token 在序列中的位置"。这就是位置编码。

**核心矛盾**：自注意力能并行（所有位置同时算），代价就是丢失顺序 → 需要位置编码补偿。

---

## 2. 绝对位置编码（Absolute Positional Encoding）

### 2.1 正余弦位置编码（原论文方案）

Transformer 论文用正弦/余弦函数，把位置 $pos$ 映射成与 embedding 同维的向量：

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)$$

$$PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)$$

特点：

1. **无需训练参数**，可外推到训练未见过的长度；
2. **相对位置可表示**：$PE_{(pos+k)}$ 是 $PE_{(pos)}$ 的线性函数（三角恒等式），理论上模型能学到相对距离；
3. 不同频率组合保证每个位置向量唯一；
4. 论文里位置向量**直接加**到 embedding 上（$x + PE$）。

缺点：

- 高频信息在长序列下会"混叠"（周期重复导致不同位置编码相似）；
- 外推能力有限：超出训练长度后编码模式与训练分布不一致，性能骤降。

### 2.2 可学习位置编码（BERT / GPT-2）

- BERT：把位置编码当**可训练参数**（$n_{\mathrm{positions}} \times d_{\mathrm{model}}$ 的嵌入表），随模型一起学习；
- GPT 系列：同样用可学习绝对位置编码，但同样**无法外推**——训练 2048 长度，推理 4096 就崩；
- 优点：灵活、简单；缺点：需要更多数据学习、参数随长度线性增长、**完全没有外推能力**。

> 面试高频：为什么正余弦比可学习好外推？→ 正余弦是函数生成、长度无关；可学习是查表，表只到训练长度。

---

## 3. 相对位置编码（Relative Positional Encoding）

绝对位置编码的问题：**模型学的是"绝对位置 i"而不是"相对距离 j-i"**。但语言结构本质上依赖相对关系（"动词离主语近"），且长度外推要求模型对"没见过的距离"也有效——这只能是相对的。

### 3.1 相对位置思想

Transformer-XL / T5 的做法：注意力分数里显式加入**相对距离偏置**：

$$\mathrm{score}(i,j) = q_i^{\top}k_j + b_{j-i}$$

其中 $b$ 是相对距离的可学习参数或函数。模型不再关注"i 是多少"，而关注"j 比 i 远多少"。

### 3.2 RoPE（Rotary Positional Embedding，旋转位置编码）—— 现代主流

**RoFormer（Su et al. 2021）** 提出，是 LLaMA、Qwen、Baichuan、GLM 等现代 LLM 的标配。

**核心思想**：把位置信息通过**旋转矩阵**乘进 Q/K 向量，让 $q_i$ 与 $k_j$ 的点积**只依赖相对位置 $j - i$**：

$$q_i^{\top}k_j = (R_iq)^{\top}(R_jk) = q^{\top}R_{j-i}k$$

- $R_i$ 是分块旋转矩阵（对向量按 2 维一组做旋转，旋转角 $= i\theta$，$\theta$ 按频率递减）；
- 旋转角度随位置线性增长，频率类似正余弦的多频设计；
- 关键性质：**旋转后点积 = 相对距离的函数**，天然具备相对位置语义；
- 实现时用复数乘法或 $\begin{bmatrix} x & -y \\ y & x \end{bmatrix}$ 分块矩阵，无需额外参数。

**为什么 RoPE 成为主流**：

1. **长度外推好**：相对位置语义 → 对未见长度的距离依然有效；
2. **无额外参数**：不占模型参数量；
3. **与注意力兼容**：直接作用于 Q/K，不改变注意力结构，FlashAttention 等加速方案无缝兼容；
4. **相对距离编码随旋转频率自然衰减**：远的 token 旋转角大，内积振荡衰减，天然有"近强远弱"的归纳偏置。

**RoPE 的局限与改进**：训练长度 4096 直接外推到 100k 仍会崩（频率超出训练分布），所以有：

- **NTK-aware / YaRN / Dynamic NTK**：推理时按新长度**插值**旋转频率，实现"免费"外推（不重新训练）;
- **高频旋转"改造"**：对高频维降低旋转速度，保留局部细节（NTK 背后的直觉）；
- **LongRoPE**：找到最优频率配置 + 微调，外推到 100 万 token。

> 📌 配图待补：RoPE 旋转位置编码示意图（旋转矩阵作用于 Q/K 的机制）

### 3.3 ALiBi（Attention with Linear Biases）

**ALiBi（Press et al. 2022）**：不加位置编码向量，直接在注意力分数上加**线性距离惩罚**：

$$\mathrm{score}(i,j) = q_i^{\top}k_j - m \cdot |i - j|$$

- $m$ 是每头的固定斜率（头越深斜率越大，类似多频设计）；
- **零额外参数**、**训练时不加位置编码**（省一层 embedding）；
- 训练短（1024）外推长（2048/4096）效果好，曾用于 BLOOM、MosaicML 模型；
- 局限：线性惩罚对超长距离衰减过快，超长上下文弱于 RoPE 系。

> 面试高频：RoPE 和 ALiBi 的区别？→ RoPE 把相对位置编进 Q/K 的旋转（乘法），ALiBi 直接加到注意力分数（加法偏置）；RoPE 外推上限更高且主流。

---

## 4. 演进对比一览

| 方案 | 类型 | 参数 | 外推能力 | 代表模型 |
|------|------|------|----------|----------|
| 正余弦 PE | 绝对 | 无 | 一般（理论上可，实践衰减） | Transformer 原论文 |
| 可学习 PE | 绝对 | 有（$n \times d$ 表） | 差（查表止于训练长度） | BERT、GPT-2 |
| 相对距离偏置 | 相对 | 有（小表） | 中 | Transformer-XL、T5 |
| **RoPE** | 相对（旋转） | **无** | **好（+NTK/YaRN 可超长外推）** | **LLaMA、Qwen、GLM** |
| ALiBi | 相对（线性偏置） | **无** | 好（短训长推） | BLOOM |

**现代主流结论**：Decoder-only LLM 几乎统一用 **RoPE**（或其变体），因为相对位置 + 无参数 + 外推友好 + 兼容 FlashAttention。

---

## 5. 面试高频问题速查

1. **为什么 Transformer 需要位置编码？** → 自注意力置换不变，对顺序无感（详见 1）。
2. **正余弦位置编码是怎么设计的？** → 多频 sin/cos，偶/奇维分开，位置唯一且相对位置可线性表示（详见 2.1）。
3. **绝对 vs 相对位置编码的本质区别？** → 绝对关心"我在第几个"，相对关心"你离我多远"；语言依赖相对关系，外推需要相对（详见 3）。
4. **RoPE 是怎么实现相对位置的？** → 旋转矩阵把 q/k 按位置旋转，点积结果只依赖角度差（相对位置）（详见 3.2）。
5. **RoPE 为什么能外推？** → 相对位置语义 + 多频旋转，对未见距离依然有效；配合 NTK/YaRN 插值频率可超长外推（详见 3.2）。
6. **RoPE 与 ALiBi 的区别？** → 旋转乘法 vs 线性加法偏置；RoPE 兼容 FlashAttention 且超长更强（详见 3.3）。
7. **可学习位置编码的缺点？** → 参数随长度线性增长 + 查表无法外推（详见 2.2）。
8. **NTK/YaRN 外推的原理？** → 推理时按新长度缩放旋转频率（插值），让频率分布匹配新长度，无需重训（详见 3.2）。
9. **位置编码加在哪？** → 绝对 PE 加在 embedding 后（$x + PE$）；RoPE 乘进 Q/K；ALiBi 加在注意力分数（详见 2/3）。

---

## 6. 手撕：RoPE 实现（面试手撕）

```python
import torch
import torch.nn as nn
import math

def rope_embedding(seq_len, dim, base=10000.0, device='cpu'):
    """生成 RoPE 旋转角度矩阵 (seq_len, dim/2)"""
    freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device) / dim))  # (dim/2,)
    pos = torch.arange(seq_len, device=device)                              # (seq_len,)
    return torch.outer(pos, freq)                                           # 位置 × 频率

def apply_rope(x, theta):
    """
    x: (batch, seq_len, dim) 或 (batch, heads, seq_len, dim)
    theta: (seq_len, dim/2) 旋转角度
    把 x 的每两维 (x1,x2) 旋转 theta 角：x1' = x1 cosθ - x2 sinθ, x2' = x1 sinθ + x2 cosθ
    """
    # 拆成两半（相邻对）
    x1 = x[..., 0::2]   # 偶维
    x2 = x[..., 1::2]   # 奇维
    cos_t = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim/2)
    sin_t = torch.sin(theta).unsqueeze(0).unsqueeze(0)
    rot_x1 = x1 * cos_t - x2 * sin_t
    rot_x2 = x1 * sin_t + x2 * cos_t
    # 交错还原
    return torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)

def rope_attention(q, k, v, theta):
    """RoPE 版注意力：先旋转 q/k，再做标准注意力"""
    q, k = apply_rope(q, theta), apply_rope(k, theta)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
```

> 要点：旋转作用在 Q/K 上（V 不用旋转），因为相对位置信息只影响"查询-键"的匹配。

---

## 7. 参考

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press et al., *Train Short, Test Long: Attention with Linear Biases*, 2022, [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Vaswani et al., *Attention Is All You Need*, 2017（正余弦 PE）→ [[/docs/llm/transformer-principle.md]]
- NTK-aware / YaRN（Peng et al. 2023）、LongRoPE（Ding et al. 2024）
