# Tokenizer：BPE / BBPE / WordPiece / SentencePiece（LLM 八股 13）

> **更新时间**：2026-08-31

> **标签**：Tokenizer、BPE、词表、SentencePiece、面试八股

> **一句话**：分词器把文本切成模型词表里的 token，子词（subword）方案在"词表大小"与"OOV/序列长度"之间取平衡；BPE 按频率合并、WordPiece 按似然增益合并、BBPE 在字节层面做 BPE 从而**永不 OOV**，SentencePiece 是无需预分词的工程实现。

> **关联阅读**：[[/docs/llm/llm-architecture-decoder-only.md]]、[[/docs/llm/pretraining-and-scaling-law.md]]

---

## 1. 为什么需要子词

| 粒度 | 词表 | 序列长度 | 问题 |
|------|------|----------|------|
| 词级（word） | 极大（10⁵~10⁶） | 短 | **OOV**、长尾词学不好、embedding 表巨大 |
| 字符级（char） | 极小（几百） | **极长** | 序列长导致注意力 $O(n^2)$ 爆炸、语义粒度太细 |
| **子词（subword）** | 3 万~30 万 | 适中 | 折中方案，现代标配 |

子词让"unhappiness"拆成 `un + happi + ness`，既复用词根又避免 OOV。

---

## 2. 主流算法

### 2.1 BPE（Byte Pair Encoding）

**训练**（贪心合并，频率驱动）：
1. 初始化词表为所有字符；
2. 统计相邻符号对频率，把**最高频的一对**合并为新符号，记入 merge 表；
3. 重复直到词表达到目标大小。

**编码**：按 merge 表的**顺序**依次应用合并规则（因此 BPE 的分词结果依赖 merge 顺序，是确定性的）。

代表：GPT-2/3/4、LLaMA（BPE 变体）、RoBERTa。

### 2.2 BBPE（Byte-level BPE）

在 **UTF-8 字节**上做 BPE，基础词表固定为 256 个字节：

- **永不 OOV**：任何字符最坏情况被拆成若干字节；
- 不需要 `<unk>`，天然支持 emoji、生僻字、任意二进制文本；
- 代价：**非拉丁语言（如中文）吃亏**——一个汉字 UTF-8 占 3 字节，若词表没学到该字的合并，就要用 3 个 token 表示 1 个字；
- GPT-2 起成为主流（GPT-4 的 `cl100k_base`/`o200k_base` 也是 BBPE）。

> 面试高频：**中文为什么比英文更"费 token"？** → ① BBPE 下汉字按 3 字节起步；② 训练语料以英文为主，中文合并规则学得少；③ 早期模型词表中中文覆盖不足。国产模型（Qwen、GLM、DeepSeek）会扩充中文词表，把中文压缩率做到约 1.5–1.8 字/token 量级，显著优于原始 GPT-2 词表。

### 2.3 WordPiece

与 BPE 的唯一本质差别在**合并准则**：不是选频率最高的对，而是选让**语言模型似然增益最大**的对：

$$\text{score}(a,b) = \frac{P(ab)}{P(a)P(b)} \approx \frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$$

即偏好"共现远高于独立出现"的组合。代表：BERT（用 `##` 标记词内后续子词）。

### 2.4 Unigram LM（SentencePiece 默认）

反向思路：先给一个大候选集，用 EM 迭代**删除**对总似然损害最小的子词，直到目标词表大小。编码时用 Viterbi 找最大似然切分（可给出多种切分概率 → 支持 **subword regularization** 做数据增强）。代表：ALBERT、T5、XLNet、多数多语言模型。

### 2.5 SentencePiece（工程实现，不是算法）

- **无需预分词**（language-independent），直接在原始文本上训练，适合中日泰等无空格语言；
- 用 `▁`（U+2581）显式表示空格 → **可完全可逆还原**原文（lossless detokenization）；
- 支持 BPE 与 Unigram 两种算法、支持 byte fallback；
- LLaMA-1/2、Qwen（早期）、ChatGLM 等大量模型使用。

### 2.6 对比表

| 维度 | BPE | BBPE | WordPiece | Unigram |
|------|-----|------|-----------|---------|
| 合并/剪枝准则 | 频率最高对 | 同 BPE（字节级） | 似然增益最大对 | 删除损失最小的子词 |
| 方向 | 自底向上合并 | 自底向上 | 自底向上 | **自顶向下剪枝** |
| OOV | 可能有 | **无** | `[UNK]` | 有 byte fallback 时无 |
| 多切分/概率 | 否（确定性） | 否 | 否 | **是**（可做正则化） |
| 代表 | GPT-2、RoBERTa | GPT-3/4、LLaMA-3 | BERT | T5、ALBERT |

---

## 3. 词表大小的权衡（高频设计题）

| 词表变大 | 影响 |
|----------|------|
| ✅ 序列变短 | 相同文本 token 更少 → 注意力/FFN 计算量下降、上下文能装更多内容、推理更快 |
| ✅ 语义更完整 | 常用词整体成 token，减少切碎带来的歧义 |
| ❌ Embedding + 输出头参数量线性增长 | $2\times V\times d$，$V$=200k、$d$=4096 时约 1.6B 参数（fp16 约 3.3GB） |
| ❌ Softmax/logits 计算与显存变大 | 训练时 logits 张量 $b\times s\times V$ 常是显存尖峰所在 |
| ❌ 长尾 token 训练不足 | 罕见 token 的 embedding 学不好（"glitch token"如 SolidGoldMagikarp 现象） |

**常见取值**：BERT 30k → LLaMA-2 32k → LLaMA-3 128k → GPT-4o `o200k` 200k → 部分多语言模型 25 万+。趋势是**词表变大**，因为序列变短带来的算力节省超过 embedding 变大的代价（尤其长上下文场景）。

**大词表的工程缓解**：
- 输出层与 embedding **权重绑定**（weight tying）省一半；
- 分块计算 logits / fused cross-entropy（如 Liger-Kernel 类实现），避免物化巨大 logits；
- 词表并行（vocab parallel embedding + 并行 CE）。

---

## 4. 特殊 token 与 Chat 模板

- 常见特殊 token：`<bos>`、`<eos>`、`<pad>`、`<unk>`、`<|im_start|>`/`<|im_end|>`（ChatML）、工具调用标记、`<think>`/`</think>`（推理模型思维链标记）；
- **Chat Template** 决定多轮对话如何拼接成一个字符串，训练与推理**必须完全一致**，否则效果断崖式下降 —— 这是实际工作中最高频的 bug 来源之一；
- 训练细节：SFT 时 prompt 部分 label 置 -100、`padding_side` 在解码时应为 **left**（右 padding 会让最后一个有效 token 不在末位而使位置/生成错乱）；
- **词表扩充**（加中文/领域词）后必须：调整 embedding 矩阵（新行用均值或已有子词均值初始化）+ 继续预训练，否则新 token 是随机向量，性能反而下降。

---

## 5. 手撕代码：BPE 训练与编码（最小实现）

```python
from collections import Counter

def train_bpe(corpus, vocab_size=100):
    """corpus: List[str]；返回 merge 规则列表（按顺序应用）"""
    words = Counter(w for line in corpus for w in line.split())
    splits = {w: list(w) + ["</w>"] for w in words}
    merges = []
    base = {c for s in splits.values() for c in s}
    while len(base) + len(merges) < vocab_size:
        pairs = Counter()
        for w, freq in words.items():
            s = splits[w]
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]          # 频率最高的相邻对
        merges.append(best)
        for w in splits:                            # 应用该合并
            s, out, i = splits[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(s[i] + s[i + 1]); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out
    return merges

def encode_word(word, merges):
    s = list(word) + ["</w>"]
    for a, b in merges:                             # 必须按训练顺序应用
        out, i = [], 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s
```

---

## 6. 面试高频问题速查

1. **为什么要用子词而不是词/字符？** → 词级有 OOV 与巨大词表，字符级序列过长；子词折中。
2. **BPE 的训练过程？** → 从字符出发，反复合并最高频相邻对，直到词表达标；编码按 merge 顺序应用。
3. **BPE 与 WordPiece 的区别？** → 合并准则：频率 vs 似然增益 $\frac{P(ab)}{P(a)P(b)}$。
4. **Unigram 与 BPE 的方向差别？** → Unigram 自顶向下按似然剪枝，可给出多种切分概率；BPE 自底向上确定性合并。
5. **BBPE 的最大优点？** → 字节级基础词表，永不 OOV，支持任意 Unicode/emoji。
6. **SentencePiece 是算法吗？** → 不是，是实现库（可用 BPE 或 Unigram），特点是无需预分词、用 `▁` 表示空格、可逆还原。
7. **词表大小怎么权衡？** → 大词表缩短序列、省注意力算力，但增大 embedding/输出层参数与 logits 显存，长尾 token 训练不足。
8. **中文为什么更耗 token？** → UTF-8 3 字节 + 英文主导的合并规则；扩充中文词表可显著改善。
9. **词表扩充要注意什么？** → 新 embedding 需合理初始化 + 继续预训练；tokenizer 与模型必须版本一致。
10. **为什么 Chat Template 不能改？** → 训练/推理拼接格式必须一致；特殊 token 不匹配会显著掉点。
11. **推理为什么用 left padding？** → 生成从最后一个位置继续，右 padding 会让 pad 成为"最后 token"，破坏位置与生成逻辑。
12. **tokenizer 会影响模型能力吗？** → 会。数字切分方式影响算术能力、代码空格/缩进的切分影响代码能力，这也是各家不断改词表的原因。

---

## 参考

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units (BPE)*, arXiv:1508.07909
- Wu et al., *Google's Neural Machine Translation System (WordPiece)*, arXiv:1609.08144
- Kudo, *Subword Regularization (Unigram LM)*, arXiv:1804.10959
- Kudo & Richardson, *SentencePiece*, arXiv:1808.06226
- Radford et al., *Language Models are Unsupervised Multitask Learners (GPT-2, BBPE)*, 2019
