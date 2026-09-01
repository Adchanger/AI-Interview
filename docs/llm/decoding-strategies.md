# 解码策略：greedy / beam / top-k / top-p / temperature（LLM 八股 17）

> **更新时间**：2026-08-31

> **标签**：解码策略、采样、温度、top-p、beam search、面试八股

> **一句话**：解码是把每步的概率分布变成一个 token 的过程——确定性方法（greedy/beam）适合有唯一答案的任务，随机采样（temperature + top-k/top-p）适合开放生成；实际推理栈的作用顺序是 **logits 处理（惩罚/掩码）→ temperature → top-k → top-p → 采样**。

> **关联阅读**：[[/docs/llm/kv-cache.md]]、[[/docs/llm/reasoning-and-test-time-scaling.md]]、[[/docs/engineering/inference-serving-optimization.md]]

---

## 1. 确定性解码

### 1.1 Greedy

每步取 $\arg\max P(x_t\mid x_{<t})$。

- 优点：快、可复现；
- 缺点：**局部最优不等于全局最优**，容易陷入重复循环（"the the the"）、文本乏味。

### 1.2 Beam Search

维护 $k$ 条候选序列，每步扩展并保留总分最高的 $k$ 条。打分通常用长度归一化的对数概率：

$$\text{score}(y) = \frac{\log P(y\mid x)}{\mathrm{lp}(|y|)},\qquad \mathrm{lp}(t)=\Big(\frac{5+t}{6}\Big)^{\alpha}$$

- **为什么要长度惩罚？** 对数概率随长度单调下降，不归一化会**系统性偏好短句**；
- 适用：机器翻译、摘要、语音识别等**有相对唯一正确答案**的任务；
- **不适合开放生成**：Beam 越大文本越"安全乏味"（likelihood trap，Holtzman et al. 指出高概率序列往往退化重复）；
- 成本：显存与算力约 ×k，且与 KV Cache/批调度实现耦合复杂，因此**现代 LLM 服务默认不开 beam**。
- 变体：diverse beam search、group beam search（提升多样性）。

---

## 2. 随机采样

### 2.1 Temperature

$$P_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

| $T$ | 效果 |
|-----|------|
| $T\to0$ | 趋近 greedy（分布变尖） |
| $T=1$ | 原始分布 |
| $T>1$ | 分布变平，更随机、更"有创意"也更容易胡说 |

工程：$T=0$ 一般被实现为直接 greedy（避免除零）。

### 2.2 Top-k 采样

只保留概率最高的 $k$ 个 token，重新归一化后采样。

- 问题：$k$ 固定，**不适应分布形状**。分布很尖时（下一个词几乎确定）$k=50$ 会引入垃圾候选；分布很平时 $k=50$ 又砍掉了合理选项。

### 2.3 Top-p / Nucleus 采样（Holtzman et al. 2019）

取**累积概率刚好超过 $p$** 的最小 token 集合（核，nucleus），再归一化采样。

- **动态截断**：尖分布时集合小、平分布时集合大 → 比 top-k 更合理；
- 典型值 $p=0.9\sim0.95$，是当前默认。

### 2.4 其他采样与惩罚

| 方法 | 作用 |
|------|------|
| **Min-p** | 阈值取 $p_{\max}\times p_{\text{scale}}$，比 top-p 在高温下更稳健 |
| Typical / Eta / Epsilon sampling | 用局部信息量（typicality）筛选候选 |
| **repetition_penalty** | 对已出现 token 的 logits 做除法/减法惩罚（乘性，>1 生效） |
| **presence / frequency penalty** | OpenAI 风格：出现过就减固定值 / 按出现次数线性减 |
| **no_repeat_ngram_size** | 硬性禁止重复 n-gram（摘要常用，但可能损害正常复述） |
| Contrastive search | 兼顾概率与与历史表示的差异度，抑制重复 |
| **约束解码** | JSON Schema / 正则 / 语法（GBNF）→ 用 FSM 掩码非法 token，保证结构化输出（Outlines、XGrammar、SGLang 的结构化输出） |
| Logit bias | 直接对特定 token 加减 bias（强制/禁止某些词） |

> 面试高频：**temperature 与 top-p 谁先作用？** → 主流实现（HF `transformers`、vLLM）先做 logits 处理器（重复惩罚、bias、坏词掩码），**再按 temperature 缩放 logits**，然后 top-k、top-p 截断，最后 softmax 采样。顺序不同结果不同，这是很能考细节的点。

### 2.5 参数怎么选（场景表）

| 场景 | 推荐 |
|------|------|
| 代码生成 / 数学 / 抽取 / 分类 | `T=0`（greedy）或 `T≤0.2`，要可复现 |
| 通用问答 | `T=0.6~0.8, top_p=0.9` |
| 创意写作 | `T=0.9~1.1, top_p=0.95` |
| 多样本自一致（self-consistency） | `T=0.6~0.8` 采多条再投票 |
| 结构化输出 | 低温 + 约束解码/JSON mode |
| 推理模型（o1/R1 类） | 官方多建议中等温度（如 R1 建议 0.5–0.7），过低会导致重复、过高破坏推理链 |

---

## 3. 与推理加速的关系

| 技术 | 与解码策略的关系 |
|------|------------------|
| **投机解码** | 草稿模型提议 + 目标模型验证；**保证输出分布与目标模型一致**（用修正后的接受-拒绝采样），因此不改变生成质量 |
| **MTP**（多 token 预测） | 训练时多头预测未来若干 token，推理时可作为自投机草稿，见 [[/docs/llm/mtp-multi-token-prediction.md]] |
| Continuous batching | 与采样参数无关，但不同请求参数不同 → 需 per-request sampling 参数支持 |
| Prefix caching | 与解码策略无关，只影响 prefill |
| Beam search | 与分页 KV/连续批处理配合复杂，是现代服务栈默认关闭 beam 的工程原因之一 |

> 面试高频：**投机解码会改变输出质量吗？** → 标准（无损）投机解码不会：验证阶段用的是修正采样，理论上输出分布等同于目标模型直接采样；有损变体（如放宽接受阈值、Medusa 的 typical acceptance）会改变分布，需权衡。

---

## 4. 手撕代码：完整采样管线

```python
import torch

def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0,
                repetition_penalty=1.0, prev_ids=None):
    """logits: (V,) 单条序列的下一步 logits。演示标准作用顺序。"""
    logits = logits.clone().float()

    # 1) 重复惩罚（作用在温度缩放之前）
    if repetition_penalty != 1.0 and prev_ids is not None:
        uniq = torch.unique(prev_ids)
        pos = logits[uniq] > 0
        logits[uniq] = torch.where(pos, logits[uniq] / repetition_penalty,
                                   logits[uniq] * repetition_penalty)

    # 2) temperature
    if temperature <= 0:                       # T=0 → greedy
        return logits.argmax()
    logits = logits / temperature

    # 3) top-k
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits[logits < kth] = -float("inf")

    # 4) top-p（nucleus）
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p           # 保证至少保留 1 个 token
        logits[sorted_idx[remove]] = -float("inf")

    # 5) 采样
    return torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(-1)
```

---

## 5. 面试高频问题速查

1. **greedy 的问题？** → 局部最优、易重复、缺乏多样性。
2. **beam search 为什么需要长度归一化？** → 对数概率随长度递减，否则偏好短序列。
3. **为什么开放生成不用 beam？** → 高概率序列反而退化重复乏味；且成本高、与分页 KV/连续批处理配合复杂。
4. **temperature 的数学作用？** → 缩放 logits，改变 softmax 尖锐度；$T\to0$ 等价 greedy。
5. **top-k 与 top-p 的区别？** → 固定个数 vs 动态累积概率阈值；top-p 自适应分布形状，是当前默认。
6. **采样管线的作用顺序？** → logits 处理（惩罚/bias/掩码）→ temperature → top-k → top-p → softmax 采样。
7. **repetition_penalty 与 frequency_penalty 区别？** → 前者乘性、只看是否出现过；后者按出现次数线性减，可叠加 presence penalty。
8. **怎么保证输出严格是 JSON？** → 约束解码（用 JSON Schema 编译 FSM 掩码非法 token），比"提示词请求 JSON"可靠得多。
9. **T=0 一定完全可复现吗？** → 逻辑上是，但实际受 batch 组成、算子非确定性、并行归约顺序、KV 量化等影响，可能仍有细微差异。
10. **投机解码影响质量吗？** → 无损版本不影响（修正采样保证分布一致）；有损变体需评测。
11. **推理模型该用什么参数？** → 中等温度（如 0.5–0.7）+ top-p 0.95，配合足够的最大生成长度；温度过低易重复、过高破坏思维链。
12. **多样本投票为什么要提高温度？** → 需要候选之间有差异，$T=0$ 时多次采样结果相同，投票无意义。

---

## 参考

- Holtzman et al., *The Curious Case of Neural Text Degeneration (Nucleus Sampling)*, arXiv:1904.09751
- Fan et al., *Hierarchical Neural Story Generation (Top-k)*, arXiv:1805.04833
- Wu et al., *Google's NMT System*（长度惩罚公式）, arXiv:1609.08144
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192
- Willard & Louf, *Efficient Guided Generation for LLMs (Outlines)*, arXiv:2307.09702
