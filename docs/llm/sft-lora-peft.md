# SFT 与 PEFT：LoRA / QLoRA 与灾难性遗忘（LLM 八股 18）

> **更新时间**：2026-08-31

> **标签**：SFT、LoRA、QLoRA、PEFT、灾难性遗忘、面试八股

> **一句话**：SFT 用「指令-回答」对做因果语言建模（只对回答算 loss），LoRA 用低秩矩阵 $BA$ 近似权重更新、只训 0.1%~1% 参数且可合并回主干零推理开销，QLoRA 再把主干量化到 NF4 让单卡微调 65B 成为可能。

> **关联阅读**：[[/docs/llm/rlhf-ppo-dpo.md]]、[[/docs/llm/quantization.md]]、[[/docs/engineering/distributed-training.md]]

---

## 1. SFT（Supervised Fine-Tuning）

### 1.1 目标与关键细节

损失还是 token 级交叉熵，但有几个必须答对的工程点：

| 细节 | 做法与原因 |
|------|------------|
| **只对 answer 算 loss** | prompt 部分 label 设为 `-100`（忽略）。目标是学"如何回答"，对指令本身建模会浪费容量并可能损害指令遵循 |
| **Chat Template 必须一致** | 训练拼接格式与推理必须完全相同，否则效果断崖式下降 |
| **EOS 必须训到** | 不训 `<eos>`/`<|im_end|>` 会导致模型停不下来 |
| **Packing** | 多条样本拼成定长以提高吞吐；必须做**注意力隔离**（block-diagonal mask），否则跨样本串味 |
| **超参** | lr 1e-5~2e-5（全参）/ 1e-4~2e-4（LoRA）、epoch 1~3、cosine + 少量 warmup、bf16、grad clip 1.0 |
| **数据 > 数量** | LIMA 的结论：1k 条高质量多样数据即可获得很好的对齐效果（"表面对齐假设"：知识来自预训练，SFT 主要教格式与风格） |
| **多轮对话** | 每轮 assistant 内容都算 loss（或只算最后一轮，取决于策略），mask 要写对 |

> 面试高频：**SFT 能给模型注入新知识吗？** → 效率很低且风险高。SFT 主要**激发/规范**已有能力（对齐格式、风格、任务模式）；注入新知识优先用**继续预训练**或 **RAG**，硬用 SFT 灌知识容易引发幻觉（模型学会"自信地编"）。

### 1.2 数据构造

- 来源：人工标注、蒸馏更强模型（注意许可）、真实日志清洗、self-instruct/Evol-Instruct 合成；
- 质量控制：去重（语义级）、难度分层、指令多样性（任务类型/长度/语言）、答案正确性校验（代码跑测试、数学验算）、去除拒答/敷衍样本；
- 配比：通用对话 + 代码 + 数学 + 多语言 + 长文本 + 安全样本，比例决定能力画像。

---

## 2. PEFT 家族

| 类别 | 方法 | 一句话 |
|------|------|--------|
| **重参数化** | **LoRA**、AdaLoRA、DoRA、PiSSA、rsLoRA | 低秩增量，可合并、零推理开销 |
| Adapter | Houlsby Adapter、Parallel Adapter、(IA)³ | 插入小模块，推理有额外延迟 |
| **Prompt 类** | Prefix-Tuning、P-Tuning v2、Prompt Tuning | 学习"虚拟 token"，占用上下文长度 |
| 选择性 | BitFit（只训 bias）、LayerNorm-tuning | 极省参数，能力上限低 |
| 组合 | QLoRA、LoRA+、LongLoRA、MoE-LoRA | 与量化/长上下文/稀疏结合 |

---

## 3. LoRA（Hu et al. 2021）

### 3.1 原理

假设微调时的权重更新 $\Delta W$ 具有**低内在秩**，于是把它分解为两个小矩阵：

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

- $A$ 用高斯初始化，**$B$ 初始化为 0** → 训练开始时 $\Delta W=0$，等价于原模型（保证不破坏起点）；
- 训练时冻结 $W_0$，只更新 $A,B$ → 可训练参数从 $dk$ 降到 $r(d+k)$；
- **$\alpha/r$ 缩放**：让不同 $r$ 下的有效更新幅度可比，调 $r$ 时不必重调 lr。常见配置 $r=8\sim64$，$\alpha=2r$ 或 $\alpha=16/32$。

### 3.2 关键工程问题

**① 加在哪些模块？**
最初论文只加在注意力的 $W_q,W_v$；实践证明**加在所有线性层（q,k,v,o + FFN 的 gate/up/down）效果更好**，QLoRA 论文明确指出"加满所有线性层"对逼近全参微调很关键。

**② 显存省在哪？**
主要省**优化器状态与梯度**（AdamW 两个状态只对 LoRA 参数分配）。但**激活值仍需保存**（因为要反传到 LoRA 层），所以省的不是全部：全参 16 bytes/param → LoRA 约"冻结权重 2 bytes/param + 少量可训练参数开销"。

**③ 能合并吗？**
可以：$W\leftarrow W_0+\frac{\alpha}{r}BA$，合并后**推理零额外延迟**——这是 LoRA 相对 Adapter/Prefix 的最大优势。注意：合并进**量化**主干会有精度损失（QLoRA 场景通常保持不合并，或反量化到 fp16 再合并）。

**④ 多 LoRA 服务**
不合并即可**动态切换/组合**多个 LoRA（S-LoRA、vLLM 的 multi-LoRA），一份基座服务多个业务，是很有说服力的工程答案。

### 3.3 LoRA 的局限与变体

| 问题 | 变体 |
|------|------|
| 效果略逊全参（尤其继续预训练/大幅度能力迁移） | 增大 $r$、加满模块、LoRA+（给 $B$ 更大 lr）、ReLoRA（多次重启合并） |
| 各层需要的秩不同 | **AdaLoRA**（按重要性动态分配秩） |
| 只调幅度不调方向 | **DoRA**（分解为幅度 + 方向分别调，效果更接近全参） |
| 初始化浪费前期训练 | **PiSSA**（用 $W_0$ 的主奇异分量初始化 A/B） |
| 高秩时 $\alpha/r$ 缩放不当 | **rsLoRA**（改用 $\alpha/\sqrt{r}$ 缩放） |
| 长上下文微调成本 | **LongLoRA**（shifted sparse attention + 可训 norm/embed） |

---

## 4. QLoRA（Dettmers et al. 2023）

三个关键技术：

1. **NF4（4-bit NormalFloat）**：基于"预训练权重近似正态分布"设计的**信息论最优**分位量化数据类型，逐块（block-wise）量化；
2. **双重量化（Double Quantization）**：把量化常数本身再量化一次，平均每参数再省约 0.37 bit；
3. **Paged Optimizer**：用 NVIDIA 统一内存把优化器状态在显存不足时分页到 CPU，避免长序列时的 OOM 峰值。

**效果**：单张 48GB 卡微调 65B 模型，且论文报告在多个基准上**接近 16-bit 全参微调**质量。计算时把 NF4 权重**反量化到 bf16** 再做矩阵乘（所以省显存不省算力，速度通常比 LoRA 慢）。

> 面试高频：**QLoRA 为什么不损失太多效果？** → ① NF4 匹配权重的正态分布，量化误差小；② LoRA 分支保持 bf16 高精度，可"补偿"量化误差；③ 加满所有线性层的 LoRA 提高了容量。

---

## 5. 灾难性遗忘（Catastrophic Forgetting）

**现象**：在领域数据上微调后，通用能力（多轮对话、指令遵循、其它语言、代码）明显退化。

**原因**：新任务梯度覆盖了原有参数中承载通用能力的方向；数据分布单一时尤其严重。

**解法（按实用度排序）**：

1. **混入通用数据**（replay/rehearsal）：领域数据 : 通用数据常用 1:1 ~ 1:4，是最有效也最常用的手段；
2. **PEFT**：LoRA 冻结主干，天然限制漂移幅度（$r$ 小 = 容量小 = 遗忘少）；
3. **更小 lr + 更少 epoch**：1~2 epoch、lr 降一档；
4. **正则约束**：对参考模型输出加 KL 惩罚、EWC（按 Fisher 信息加权保护重要参数）、L2-SP；
5. **蒸馏自我保持**：用原模型对通用数据产出软标签，训练时一起拟合；
6. **模型融合**：训后与原模型做权重插值（model soup / task arithmetic），在能力与保持之间取平衡；
7. **评测护栏**：每次微调后跑通用 benchmark 回归（MMLU/IFEval/多轮对话）确认没退化。

---

## 6. 手撕代码：LoRA 线性层

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    """在冻结的 base 线性层旁挂低秩分支，可 merge 回主干"""
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                    # 冻结主干
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B 初始化为 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        """训练后合并，推理零额外开销"""
        self.base.weight += (self.B @ self.A) * self.scaling
        self.merged = True
```

---

## 7. 面试高频问题速查

1. **SFT 为什么只对回答算 loss？** → 目标是学"如何回答"；对 prompt 建模浪费容量、削弱指令遵循。
2. **SFT 的典型超参？** → 全参 lr 1e-5~2e-5、LoRA 1e-4~2e-4；1~3 epoch；cosine + warmup；bf16 + grad clip 1.0。
3. **SFT 数据要多少？** → LIMA 表明 1k 条高质量数据即可显著对齐；质量与多样性 > 数量。
4. **LoRA 的核心假设？** → 权重更新具有低内在秩，可用 $BA$ 近似。
5. **为什么 B 初始化为 0？** → 保证训练起点等价于原模型，避免破坏预训练权重。
6. **$\alpha$ 和 $r$ 分别是什么？** → $r$ 是秩（容量），$\alpha/r$ 是缩放系数，使不同 $r$ 下有效步长可比；常取 $\alpha=2r$。
7. **LoRA 加在哪些层？** → 建议所有线性层（q,k,v,o + FFN），QLoRA 论文指出这对逼近全参很关键。
8. **LoRA 省的是什么显存？** → 优化器状态与梯度；激活仍要存，故不是"只占 1% 显存"。
9. **LoRA 能合并吗？有什么好处？** → 能，$W_0+\frac\alpha r BA$，推理零延迟；不合并则可多 LoRA 动态切换。
10. **QLoRA 的三大技术？** → NF4 量化、双重量化、分页优化器。
11. **QLoRA 比 LoRA 慢吗？** → 通常慢，因为每次前向要反量化 NF4 到 bf16；换来的是显存大幅下降。
12. **灾难性遗忘怎么缓解？** → 混通用数据（最有效）、用 LoRA、小 lr 少 epoch、KL/EWC 正则、模型融合、回归评测。
13. **什么时候必须全参微调？** → 需要大幅改变分布/注入大量知识（继续预训练）、或对效果极限有要求且算力充足时。

---

## 参考

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314
- Zhou et al., *LIMA: Less Is More for Alignment*, arXiv:2305.11206
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, arXiv:2402.09353
- Zhang et al., *AdaLoRA*, arXiv:2303.10512
- Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, arXiv:2311.03285
