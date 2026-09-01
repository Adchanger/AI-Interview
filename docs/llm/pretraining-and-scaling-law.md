# 预训练与 Scaling Law（LLM 八股 15）

> **更新时间**：2026-08-31

> **标签**：预训练、ScalingLaw、Chinchilla、数据配比、涌现、面试八股

> **一句话**：Scaling Law 说损失随参数量、数据量、算力呈幂律下降；Kaplan(2020) 偏向"把参数做大"，Chinchilla(2022) 修正为"参数与数据同比放大、约 20 tokens/param"，而 LLaMA-3 之后行业实际按**推理成本最优**远超 Chinchilla 比例地喂数据（15T token 训 8B）。

> **关联阅读**：[[/docs/llm/llm-architecture-decoder-only.md]]、[[/docs/llm/moe-mixture-of-experts.md]]、[[/docs/engineering/distributed-training.md]]

---

## 1. 预训练做什么

**目标**：因果语言建模，最大化 $\sum_t \log P(x_t\mid x_{<t})$；评价用 **困惑度** $\mathrm{PPL}=e^{\mathcal{L}}$（每 token 平均交叉熵的指数）。

**主要工程环节**：
1. **数据**：网页（CommonCrawl → 精洗，如 RefinedWeb / FineWeb）、代码（GitHub）、书籍论文、百科、多语言、合成数据；
2. **清洗**：语言识别、质量分类器/规则过滤、**去重**（MinHash/SimHash 近重复 + 精确子串去重）、去毒、去个人信息、**去测试集污染**；
3. **配比与课程**：多阶段（通用 → 高质量/代码/数学 → 长上下文），近年常见"退火阶段（annealing）"专门喂高质量与合成数据；
4. **打包**：把短文本 pack 成固定长度（注意跨样本注意力隔离，否则会串味）；
5. **训练**：bf16/fp8 混合精度 + AdamW + warmup-cosine/WSD + 梯度裁剪 + 多维并行；
6. **监控**：loss spike、grad norm、各 benchmark 中途评测、数据健康度。

> 面试高频：**去重为什么重要？** → 重复数据会导致记忆化（隐私/版权风险）、有效数据量虚高、验证集污染，实验显示去重能明显改善下游效果与样本效率。

---

## 2. Scaling Law

### 2.1 Kaplan et al. 2020（OpenAI）

损失与三者呈**幂律**（在其它两者不成瓶颈时）：

$$L(N)\propto N^{-\alpha_N},\quad L(D)\propto D^{-\alpha_D},\quad L(C)\propto C^{-\alpha_C}$$

论文拟合出的指数很小（$\alpha_N\approx0.076$ 量级）——意味着**要 10 倍参数才能换来相对有限的 loss 下降**，但曲线极其平滑可预测。结论倾向于：给定算力增长，**参数量应比数据量增长得更快**。GPT-3（175B / 300B token）就是这个思路的产物。

### 2.2 Chinchilla（Hoffmann et al. 2022，DeepMind）

用更严谨的实验（覆盖不同 lr 调度与 400+ 模型）修正了上述结论：

- 在固定训练算力 $C\approx 6ND$ 下，**参数 $N$ 与数据 $D$ 应等比例放大**（$N\propto C^{0.5}$，$D\propto C^{0.5}$）；
- 经验法则：**每个参数约配 20 个训练 token**；
- 证据：**Chinchilla 70B / 1.4T token** 在同等算力下全面优于 Gopher 280B / 300T… 即"GPT-3 那代模型都训练不足（under-trained）"。

**算力估算公式（必背）**：

$$C_{\text{train}} \approx 6\,N\,D \ \text{FLOPs}$$

（前向 2ND + 反向 4ND；MoE 用激活参数量代入）

> 面试高频：**给定 $10^{22}$ FLOPs，怎么分配？** → $6ND=10^{22}$ 且 $D=20N$ → $120N^2=10^{22}$ → $N\approx 9\times10^{9}$（约 9B 参数），$D\approx1.8\times10^{11}$（约 180B token）。会现场算这道题非常加分。

### 2.3 训练最优 ≠ 部署最优（现代实践）

Chinchilla 只优化**训练**算力，忽略了**推理**成本。真实产品要服务海量请求，于是行业转向"**小模型 + 超量数据**"：

| 模型 | 参数 | 训练 token | tokens/param |
|------|------|-----------|--------------|
| GPT-3 | 175B | 300B | ~1.7（严重不足） |
| Chinchilla | 70B | 1.4T | 20（训练最优） |
| LLaMA-1 7B | 7B | 1T | ~143 |
| LLaMA-3 8B | 8B | **15T** | ~1875 |

结论：**远超 Chinchilla 比例继续喂数据，loss 仍在下降**（收益递减但对推理成本极其划算）。这就是"**inference-optimal / over-training**"思路。

### 2.4 其他重要 Scaling 结论

- **数据受限时的 Scaling**（Muennighoff et al., *Scaling Data-Constrained Language Models*）：数据不够时重复 epoch 是可行的，约 **4 个 epoch 内**收益接近新数据，之后迅速退化；
- **MoE Scaling**：固定训练 FLOPs 下 MoE 通常给出更低 loss（更高效算力利用）；
- **蒸馏 Scaling / 合成数据**：小模型用大模型的输出训练，可在小算力下逼近大模型能力；
- **后训练与推理时 Scaling**：2024 年后新增两条曲线——RL 后训练算力与推理时算力（见 [[/docs/llm/reasoning-and-test-time-scaling.md]]）；
- **超参 Scaling（µP / µTransfer）**：用小模型搜到的超参迁移到大模型，节省搜参算力。

---

## 3. 涌现能力（Emergent Abilities）的争议

- **原始主张**（Wei et al. 2022）：某些能力（多步算术、CoT、指令遵循）在模型规模跨过阈值后"突然出现"；
- **反驳**（Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*, NeurIPS 2023）：很多"涌现"是**指标选择的产物**——用严格的 exact-match 精确匹配指标会呈阶跃，换成连续指标（token 级编辑距离、log-prob）后曲线是平滑的；
- **稳妥答法**：Scaling 带来的能力提升是真实的，但"是否存在真正的相变式涌现"取决于指标定义，学界仍有争论。这样答既准确又显示读过文献。

---

## 4. 面试常问的工程数字

- **模型参数量估算**（Decoder-only，忽略 bias/norm）：
  每层 ≈ 注意力 $4d^2$（Q/K/V/O，MHA）+ FFN $3\times d\times d_{ff}$（SwiGLU，$d_{ff}\approx\frac83d$）≈ $4d^2+8d^2=12d^2$；
  总计 ≈ $12\,L\,d^2$ + embedding $V\!\times\!d$（+ 输出头，若不绑定权重）。
  代入 LLaMA-7B（$L{=}32,d{=}4096$）：$12\times32\times4096^2\approx 6.4$B，加 embedding≈0.26B → 约 6.7B ✓
- **训练显存**（全参、混合精度 + AdamW）≈ 16 bytes/param（fp16 参数 2 + fp16 梯度 2 + fp32 主参数 4 + Adam m/v 各 4）+ 激活；详见 [[/docs/engineering/distributed-training.md]]；
- **训练时间**：$T=\frac{6ND}{\text{GPU 数}\times\text{单卡有效 FLOPS}}$，有效算力 = 峰值 × MFU（大规模训练 MFU 常在 35%~55%）。

---

## 5. 面试高频问题速查

1. **Scaling Law 的基本形式？** → loss 对参数/数据/算力呈幂律下降，且在多个数量级上高度可预测。
2. **Chinchilla 的核心结论？** → 固定算力下参数与数据等比放大，约 20 tokens/param；GPT-3 那代普遍训练不足。
3. **训练算力公式？** → $C\approx6ND$（MoE 用激活参数）。
4. **给定算力如何分配参数与数据？** → 解 $6ND=C,\ D=20N$；要能现场算。
5. **为什么 LLaMA-3 8B 要训 15T token？** → 优化的是**推理成本**：过训练的小模型部署更便宜，尽管训练算力上不是最优。
6. **数据不够怎么办？** → 有限重复（约 4 epoch 内收益接近新数据）、合成数据、更强清洗与配比、蒸馏。
7. **PPL 与 loss 的关系？** → $\mathrm{PPL}=e^{\mathcal{L}}$；跨 tokenizer 比较 PPL 无意义（token 粒度不同）。
8. **涌现能力真的存在吗？** → 能力提升真实，但阶跃形态很大程度由不连续指标造成，学界有争论。
9. **预训练数据清洗有哪些关键步骤？** → 语言识别、质量过滤、去重、去毒/隐私、去污染、配比与课程。
10. **loss spike 怎么处理？** → 梯度裁剪、跳过异常数据、回滚 checkpoint、降 $\beta_2$、加 z-loss / QK-Norm。
11. **MFU 是什么？** → 模型算力利用率 = 实际有效 FLOPs / 硬件峰值，衡量训练效率的核心指标。
12. **Scaling Law 会失效吗？** → 高质量数据接近耗尽是公认瓶颈；行业转向合成数据、后训练 RL 与推理时算力三条新曲线。

---

## 参考

- Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv:2001.08361
- Hoffmann et al., *Training Compute-Optimal Large Language Models (Chinchilla)*, arXiv:2203.15556
- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264
- Wei et al., *Emergent Abilities of Large Language Models*, arXiv:2206.07682
- Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*, arXiv:2304.15004
- Dubey et al., *The Llama 3 Herd of Models*, arXiv:2407.21783
