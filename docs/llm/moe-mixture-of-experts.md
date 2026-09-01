# MoE 混合专家：路由、负载均衡与专家塌陷（LLM 八股 12）

> **更新时间**：2026-08-31

> **标签**：MoE、稀疏激活、路由、负载均衡、专家并行、面试八股

> **一句话**：MoE 把 FFN 换成"多个专家 + 一个路由器"，每个 token 只激活 Top-k 个专家，从而**参数量与计算量解耦**——总参数可涨十倍而单 token FLOPs 几乎不变；代价是全部专家权重都要驻留显存、需要负载均衡与专家并行通信。

> **关联阅读**：[[/docs/llm/deepseek-family.md]]、[[/docs/engineering/distributed-training.md]]、[[/docs/llm/pretraining-and-scaling-law.md]]

---

## 1. 结构：把 FFN 替换成专家池

标准 Transformer 每层：Attention → FFN。MoE 层把 FFN 替换为 $N$ 个并列的 FFN（专家）+ 路由器（gating network）：

$$y = \sum_{i\in \mathcal{T}(x)} g_i(x)\cdot E_i(x),\qquad g(x)=\mathrm{Softmax}(\mathrm{TopK}(W_g x))$$

- $\mathcal{T}(x)$：路由选出的 Top-k 专家集合（k 通常 1~8）；
- $g_i$：门控权重，对被选专家的输出加权求和；
- **通常隔层替换**（如每 2 层一个 MoE 层），也有全替换的设计；注意力部分一般保持稠密。

### 1.1 核心收益

| 指标 | 稠密模型 | MoE |
|------|----------|-----|
| 总参数 | $P$ | $\gg P$（专家数倍） |
| **单 token 激活参数** | $P$ | 只有 Top-k 专家那部分 |
| 训练 FLOPs/token | 高 | 与激活参数成正比（低） |
| 显存占用 | $P$ | **全部专家都要放**（不省） |
| 通信 | 常规 | 多出 All-to-All（专家并行） |

> 面试高频：**MoE 为什么省算力但不省显存？** → 稀疏激活只减少**计算**（每 token 只过 k 个专家），但所有专家权重都必须驻留在显存/多卡上以供任意 token 路由；因此 MoE 是"用显存和通信换算力效率"。

### 1.2 代表模型（对照记忆）

| 模型 | 专家配置 | 总参 / 激活参 |
|------|----------|--------------|
| Switch Transformer (2021) | **Top-1** 路由，最多上千专家 | 最大 1.6T |
| GShard | Top-2 | — |
| **Mixtral 8×7B** | 8 专家 Top-2 | 46.7B / 约 12.9B |
| **DeepSeek-V3** | 256 个路由专家 + 1 个共享专家，Top-8 | 671B / 37B |
| Qwen3-MoE、Kimi、GLM-MoE 等 | 细粒度专家 + 共享专家已成主流范式 | — |

![DeepSeek MoE 的细粒度专家 + 共享专家](../images/deepseek-moe-perf-01.png)

图1：DeepSeekMoE 的细粒度专家切分与共享专家隔离（来源：DeepSeek-V2/V3 技术报告）

---

## 2. 路由（Routing）

### 2.1 Top-k 的取舍

- **Top-1**（Switch）：通信与计算最省，实现最简，但对路由错误更敏感；
- **Top-2**（GShard/Mixtral）：有"第二意见"，训练更稳，是长期默认；
- **Top-8 + 细粒度专家**（DeepSeek-V3）：把专家切小、选更多个，组合数大幅上升 → 表达更灵活、专业化更强，同时保持激活比例低。

### 2.2 容量因子与 token drop

每个专家有容量上限 $C=\text{capacity\_factor}\times\frac{\text{tokens}}{N}$：
- 超出容量的 token 被 **drop**（直接走残差）或 **重路由**到次优专家；
- 容量因子 >1（如 1.25）留缓冲；太大浪费算力，太小丢 token 伤效果；
- 推理时通常不 drop（可变 batch，实现上用分组 GEMM）。

### 2.3 其他路由变体

| 方案 | 思路 |
|------|------|
| Expert Choice | 反转选择方向：**专家选 token**，天然负载均衡、无需 drop |
| Hash / 随机路由 | 无学习的路由基线，说明部分收益来自"容量"而非"智能路由" |
| Soft MoE | 对 token 做加权组合再送专家，完全可微、无离散路由 |
| 共享专家（DeepSeek） | 少数专家**所有 token 都过**，负责通用知识，让路由专家专注差异化 |
| **无辅助损失的均衡**（DeepSeek-V3） | 用可动态调整的 **per-expert bias** 影响路由打分实现均衡，避免辅助损失干扰主目标 |

---

## 3. 负载均衡与专家塌陷（高频难题）

### 3.1 问题

路由器是学出来的，容易形成**马太效应**：少数专家被频繁选中 → 得到更多梯度 → 变得更强 → 被更频繁选中。极端情况就是 **专家塌陷（expert collapse）**：绝大多数 token 挤到少数专家，其余专家几乎不被训练，MoE 退化成小稠密模型；同时并行时出现严重的**负载倾斜**（某卡忙死、其他卡空转，木桶效应拖慢整体）。

### 3.2 解法

1. **辅助负载均衡损失**（Switch Transformer 经典式）：
   $$\mathcal{L}_{aux} = \alpha\cdot N\sum_{i=1}^{N} f_i\cdot P_i$$
   $f_i$ 为分配给专家 $i$ 的 token 比例，$P_i$ 为路由概率均值；当两者都均匀时该项最小。$\alpha$ 常取 $10^{-2}$ 量级，太大伤主任务、太小不起效。
2. **路由 z-loss**：约束路由 logits 的量级，防止数值不稳与极端 softmax。
3. **加噪声路由**（Noisy Top-k Gating，Shazeer 2017）：在 gating logits 上加高斯噪声鼓励探索。
4. **容量限制 + drop**：物理上限制单专家吃掉的 token 量。
5. **专家 dropout / 随机化初始化路由器**。
6. **无辅助损失方案**（DeepSeek-V3）：为每个专家维护一个偏置项，按实时负载动态增减，负载高的专家被"降权"，实现均衡且不引入与主 loss 冲突的梯度。
7. **序列级/设备级均衡约束**：限制单条序列或单个设备上的分布不均。

![MoE 负载均衡示意](../images/moe-load-balance-01.png)

图2：MoE 负载均衡策略示意（来源：DeepSeek 技术报告相关插图）

---

## 4. 训练与部署工程

### 4.1 专家并行（EP）

专家分布到不同设备，每层需要两次 **All-to-All**（把 token 发到目标专家所在设备、再把结果收回）：

- 通信量与 token 数、hidden size 成正比，是 MoE 的主要额外开销；
- 优化手段：计算-通信重叠（DeepSeek 的 **DualPipe**）、拓扑感知路由（限制一个 token 最多跨几个节点）、fp8 通信、分组 GEMM；
- 与 TP/PP/DP 组合成多维并行，见 [[/docs/engineering/distributed-training.md]]。

### 4.2 推理特点

- **显存**：必须放下全部专家（Mixtral 8×7B 约 47B 参数，bf16 需 ~94GB）；
- **计算**：等效于 ~13B 稠密模型，因此"显存大但算得快"；
- batch 内不同 token 路由到不同专家 → **kernel 需要分组/排序**（sort-by-expert + grouped GEMM），实现复杂度高于稠密模型；
- 小 batch 时专家利用率低，MoE 的吞吐优势要在大 batch 下才充分体现。

### 4.3 其他实践坑

- **微调更易过拟合**：专家多、单专家数据少；常做法是只微调注意力/共享参数，或用更强正则、LoRA；
- **量化**：专家权重可量化，但路由器对精度敏感，一般保持较高精度；
- **推理确定性**：容量溢出/drop 策略会让输出依赖 batch 组成，需注意可复现性。

---

## 5. 手撕代码：Top-k 路由 MoE 层（含辅助损失）

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d, n_expert=8, k=2, hidden=None, alpha=1e-2):
        super().__init__()
        hidden = hidden or 4 * d
        self.k, self.n, self.alpha = k, n_expert, alpha
        self.gate = nn.Linear(d, n_expert, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d))
            for _ in range(n_expert))

    def forward(self, x):                       # x: (B, T, d)
        B, T, d = x.shape
        flat = x.reshape(-1, d)                 # (M, d)
        logits = self.gate(flat)                # (M, N)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)  # 重归一化门控权重

        out = torch.zeros_like(flat)
        for e in range(self.n):                 # 实际实现用 grouped GEMM，这里为可读性循环
            sel = (topi == e)
            if not sel.any():
                continue
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            w = (topv * sel).sum(-1)[rows].unsqueeze(-1)
            out[rows] += w * self.experts[e](flat[rows])

        # 负载均衡辅助损失：f_i（实际分配比例） · P_i（平均路由概率）
        with torch.no_grad():
            one_hot = F.one_hot(topi, self.n).sum(1).float()   # (M, N)
        f = one_hot.mean(0)
        P = probs.mean(0)
        aux = self.alpha * self.n * (f * P).sum()
        return out.view(B, T, d), aux
```

---

## 6. 面试高频问题速查

1. **MoE 的核心收益是什么？** → 参数量与单 token 计算量解耦，用同等训练/推理算力换更大容量。
2. **MoE 省显存吗？** → 不省，全部专家都要驻留；省的是 FLOPs。
3. **替换的是哪部分？** → 通常是 FFN（可隔层替换），注意力一般保持稠密。
4. **Top-1 与 Top-2 怎么选？** → Top-1 最省通信但更依赖路由准确性；Top-2 更稳；细粒度专家可用更大 k。
5. **什么是专家塌陷？怎么解决？** → 路由马太效应导致少数专家吃掉大部分 token；解法：辅助均衡损失、z-loss、噪声路由、容量限制、共享专家、动态 bias 均衡（DeepSeek-V3）。
6. **负载均衡损失的形式？** → $\alpha N\sum_i f_iP_i$，$f_i$ 分配比例、$P_i$ 平均门控概率。
7. **容量因子是什么？溢出怎么办？** → 单专家可处理 token 上限系数；溢出 token 被丢弃（走残差）或重路由。
8. **MoE 的主要通信开销？** → 专家并行的两次 All-to-All；靠计算通信重叠、限制跨节点路由、低精度通信优化。
9. **MoE 推理为什么大 batch 才划算？** → 小 batch 下专家利用率低、分组 GEMM 效率差，无法摊薄权重读取。
10. **共享专家有什么用？** → 承载通用知识，减少路由专家的知识冗余，让专业化更彻底（DeepSeekMoE）。
11. **MoE 微调有什么坑？** → 更易过拟合、路由分布可能漂移；常只训部分模块或用 LoRA + 更强正则。
12. **MoE 与 Scaling Law 的关系？** → 在固定训练 FLOPs 下，MoE 通常给出更低的 loss（更高效的算力利用），但显存/通信成本与推理复杂度上升。

---

## 参考

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*, arXiv:1701.06538
- Fedus et al., *Switch Transformers*, arXiv:2101.03961
- Lepikhin et al., *GShard*, arXiv:2006.16668
- Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, arXiv:2202.09368
- Jiang et al., *Mixtral of Experts*, arXiv:2401.04088
- Dai et al., *DeepSeekMoE*, arXiv:2401.06066；DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437
