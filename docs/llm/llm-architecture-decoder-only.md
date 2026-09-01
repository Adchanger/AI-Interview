# 架构选型：为什么现在都是 Decoder-only（LLM 八股 14）

> **更新时间**：2026-08-31

> **标签**：Decoder-only、Encoder-only、BERT、GPT、架构选型、面试八股

> **一句话**：Encoder-only（BERT）擅长理解但不能生成，Encoder-Decoder（T5）适合有明确输入输出的转换任务，**Decoder-only（GPT/LLaMA）**因为训练目标信息密度高、结构简单可极致 scale、KV Cache 与 in-context learning 友好，成为通用大模型的唯一主流。

> **关联阅读**：[[/docs/llm/transformer-principle.md]]、[[/docs/llm/pretraining-and-scaling-law.md]]、[[/docs/llm/kv-cache.md]]

---

## 1. 三种架构对照

| 维度 | Encoder-only | Encoder-Decoder | **Decoder-only** |
|------|--------------|-----------------|------------------|
| 代表 | BERT、RoBERTa、DeBERTa、BGE | T5、BART、mT5、原始 Transformer | GPT 系、LLaMA、Qwen、DeepSeek |
| 注意力掩码 | 双向（全可见） | Encoder 双向 + Decoder 因果 + Cross-Attn | **因果（下三角）** |
| 预训练目标 | MLM（掩码 15%） | Span corruption / denoising | **Next-token prediction（CLM）** |
| 能否生成 | 不能（非自回归） | 能 | 能 |
| 典型任务 | 分类、NER、句向量、检索 | 翻译、摘要、结构化转换 | 通用对话、代码、推理、几乎一切 |
| 参数效率 | 高（理解类小模型够用） | 参数分两部分，同规模下"有效深度"较浅 | 全部参数服务同一目标 |

---

## 2. 为什么 Decoder-only 胜出（必背五点）

1. **训练信号密度最高**
   MLM 只对约 15% 的被掩码位置计算 loss；CLM 对**每个位置**都预测下一个 token，一条长度 $n$ 的序列提供 $n$ 个训练信号 → 相同语料下样本效率更高。

2. **不存在预训练-微调不一致**
   BERT 训练时见 `[MASK]`、下游没有 `[MASK]`（XLNet 指出的 pretrain-finetune discrepancy）；CLM 训练与推理形式完全一致。

3. **结构最简单，最好 scale**
   只有一种 block，没有 cross-attention、没有 encoder/decoder 参数分配问题 → 张量并行/流水并行切分简单，Scaling Law 曲线干净可预测，超参更易迁移。

4. **推理效率与 KV Cache**
   因果掩码使每个 token 的 K/V 一次算完永久可用 → KV Cache 天然成立、增量解码高效；双向注意力在生成场景每加一个 token 就要重算全部表示。

5. **In-context learning / 统一接口**
   一切任务都能写成"续写"：few-shot、CoT、工具调用、多轮对话统统是同一个接口，无需为每个任务改结构。这一条是 GPT-3 之后范式确立的关键。

**补充证据**：Google 的 *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?*（arXiv:2204.05832）系统对比后发现，**causal decoder-only + 自回归目标**在零样本泛化上最好；而先用 CLM 预训练再做少量适配可迁移到其它设置。这条论文级证据能显著加分。

> 面试高频：**Decoder-only 的注意力矩阵是下三角，会不会浪费一半算力？** → 会有理论上的一半浪费，但 FlashAttention 等实现只计算需要的块，实际开销接近一半而非全量；同时因果结构换来的 KV Cache 与训练目标优势远超这点损失。

---

## 3. 那 Encoder-only / Enc-Dec 还有用吗

**仍在广泛使用**，只是不做通用大模型：

| 场景 | 首选 | 原因 |
|------|------|------|
| 句向量 / 检索 / rerank | Encoder-only（BGE、E5、GTE）或基于 LLM 的 embedding | 需要双向语义压缩成定长向量、延迟低、成本低 |
| 文本分类 / NER / 审核 | Encoder-only（小模型可 CPU 部署） | 成本远低于 LLM，精度足够 |
| 机器翻译 / 语音识别 | Encoder-Decoder（Whisper、NLLB） | 输入输出模态/语言明确分离，cross-attn 天然合适 |
| 多模态 | 视觉 encoder + LLM decoder（LLaVA/Qwen-VL 范式） | 见 [[/docs/llm/vlm-evolution.md]] |
| 稠密 embedding + 生成一体 | Decoder-only 抽取最后隐状态 / 加对比训练 | 近年 LLM-based embedding（如 E5-Mistral）在榜单上更强 |

> 面试高频：**为什么检索还在用 Encoder（BERT 系）？** → ① 双向注意力对"整句压缩为一个向量"更自然；② 参数小、延迟低、成本可控（要为亿级文档建索引）；③ 对比学习训练成熟。但近两年 decoder-only 微调出的 embedding 模型在 MTEB 上已明显领先，趋势是两条路线并存。

---

## 4. 掩码与目标的组合空间（进阶）

| 组合 | 说明 | 代表 |
|------|------|------|
| Causal Decoder | 全因果掩码 + CLM | GPT、LLaMA |
| **Prefix Decoder**（非因果解码器） | prompt 部分双向可见、生成部分因果 | GLM、U-PaLM、部分多模态模型 |
| Encoder-Decoder | 双向编码 + 因果解码 + cross-attn | T5、BART |
| MLM | 双向 + 掩码重建 | BERT |
| **GLM 的自回归空白填充** | 打乱 span 顺序 + 自回归填空，兼顾理解与生成 | ChatGLM |
| **FIM（Fill-in-the-Middle）** | 重排为 prefix-suffix-middle 后仍用 CLM 训练 | 代码模型（StarCoder、DeepSeek-Coder），补全必备 |

**Prefix LM 的取舍**：prompt 双向可见理论上理解更强，但会**破坏 prompt 部分 KV Cache 的可复用性**（前缀不可增量扩展），且训练目标信息密度低于纯 CLM，因此主流仍选纯因果。

---

## 5. GPT / BERT / T5 演进要点速记

- **BERT (2018)**：MLM + NSP；NSP 后被 RoBERTa 证明无用而去掉；开启"预训练 + 微调"范式；
- **GPT-2 (2019)**：更大 + zero-shot 初现；
- **GPT-3 (2020)**：175B，**in-context learning / few-shot** 成立，范式从"微调"转向"提示"；
- **T5 (2019)**：把所有 NLP 任务统一成 text-to-text，span corruption 目标；
- **InstructGPT (2022)**：SFT + RLHF，把"能力"变成"可用"；
- **LLaMA (2023)**：Pre-RMSNorm + SwiGLU + RoPE + GQA(2 代大模型)，开源标准配方；
- **o1/R1 (2024–2025)**：训练重心从预训练扩展到**后训练 RL + 推理时算力**，见 [[/docs/llm/reasoning-and-test-time-scaling.md]]；
- **2025–2026**：稀疏 MoE 常态化、原生多模态、长上下文与 Agent 化成为标配。

---

## 6. 面试高频问题速查

1. **三种架构分别适合什么任务？** → Encoder-only 理解/检索；Enc-Dec 明确的序列转换；Decoder-only 通用生成。
2. **为什么主流是 Decoder-only？** → 训练信号密度高、无 pretrain-finetune 不一致、结构简单好 scale、KV Cache 友好、in-context learning 统一接口。
3. **MLM 与 CLM 的信号量差多少？** → MLM 每序列约 15% 位置产生 loss，CLM 每个位置都产生 loss。
4. **BERT 为什么不能直接做生成？** → 双向掩码 + 非自回归目标，没有"逐步生成"的机制；强行迭代填空效率与质量都差。
5. **Prefix LM 是什么？为什么不是主流？** → prompt 双向、生成因果；破坏前缀 Cache 复用、训练信号密度较低。
6. **Encoder-Decoder 的 cross-attention 作用？** → 解码端用 Q 去查询编码端的 K/V，实现"条件生成"。
7. **T5 的预训练目标？** → span corruption（掩掉连续片段并用 sentinel token 还原），text-to-text 统一。
8. **NSP 有用吗？** → RoBERTa 证明去掉 NSP、用更大数据/更长训练更好。
9. **FIM 是干什么的？** → 让 decoder-only 模型具备中间填空能力，代码补全必需。
10. **检索/分类现在该用什么？** → 成本敏感且规模大用 Encoder-only 小模型；追求效果可用 LLM-based embedding；两者并存。
11. **未来还会回到 Encoder-Decoder 吗？** → 多模态与语音领域仍大量使用（编码非文本模态），纯文本通用模型短期看不到回归动力。

---

## 参考

- Devlin et al., *BERT*, arXiv:1810.04805
- Brown et al., *Language Models are Few-Shot Learners (GPT-3)*, arXiv:2005.14165
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, arXiv:1910.10683
- Wang et al., *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?*, arXiv:2204.05832
- Du et al., *GLM: General Language Model Pretraining with Autoregressive Blank Infilling*, arXiv:2103.10360
- Bavarian et al., *Efficient Training of Language Models to Fill in the Middle*, arXiv:2207.14255
