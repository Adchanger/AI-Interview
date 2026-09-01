# 手撕代码题清单（面试八股 02）

> **更新时间**：2026-08-31

> **标签**：手撕代码、白板编程、多头注意力、面试八股

> **一句话**：算法岗手撕高频集中在四类——**注意力/Transformer 组件**、**经典机器学习算法**、**评估指标与数值稳定实现**、**LLM 相关工程实现（KV Cache、采样、LoRA、beam search）**；本文按"能否 10 分钟内白板写出"的标准给出可直接背的实现。

> **关联阅读**：[[/docs/interview/bagu-knowledge-map.md]]、[[/docs/llm/transformer-principle.md]]、[[/docs/llm/attention-variants-mha-mqa-gqa.md]]

---

## 0. 白板编程通用注意事项

1. **先说思路再写**：输入输出形状、复杂度、边界；
2. **写形状注释**：`# (B, h, T, dk)` 这类注释是加分项，也避免自己写错；
3. **数值稳定**：softmax 减最大值、log-sum-exp、除法加 eps、`sigmoid` 分支实现；
4. **边界条件**：空输入、单类别（AUC）、除零、mask 与 padding；
5. **写完自测**：给一个小例子口算验证；
6. 不确定 API 时**说明并写伪代码**，别卡死在函数名上。

---

## 1. 多头自注意力（最高频，必须默写）

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, causal=False):
        B, T, D = x.shape
        # (B,T,D) -> (B,h,T,dk)
        q = self.wq(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dk).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)      # (B,h,T,T)
        if causal:
            cm = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(cm, float("-inf"))
        if mask is not None:                                       # padding mask: (B,1,1,T)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.drop(scores.softmax(-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)           # 合并多头
        return self.wo(out)
```

**必答追问**：为什么除 $\sqrt{d_k}$（点积方差随维度增长，防 softmax 饱和）；mask 用 `-inf` 而不是 0（softmax 后才为 0）；复杂度 $O(T^2 d)$。

---

## 2. RoPE 与 RMSNorm（LLM 组件）

```python
def build_rope_cache(seq_len, dim, base=10000.0, device="cpu"):
    """dim 为每个头的维度（需为偶数）"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                    # (T, dim/2)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """x: (B, h, T, dim)；把相邻两维看作复数做旋转"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight
```

---

## 3. Softmax / 交叉熵 / LayerNorm（数值稳定）

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)             # 防溢出
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def logsumexp(x, axis=-1, keepdims=False):
    m = x.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)

def cross_entropy(logits, target):
    """logits (N,K)，target (N,) 类别索引"""
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -logp[np.arange(len(target)), target].mean()

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
```

---

## 4. 经典机器学习

### 4.1 KMeans（含 KMeans++）
见 [[/docs/ml/unsupervised-clustering-and-dimensionality.md]] 的完整实现。

### 4.2 逻辑回归 / 线性回归
见 [[/docs/ml/logistic-regression.md]]。

### 4.3 AUC（秩公式）
见 [[/docs/ml/model-evaluation-metrics.md]]。

### 4.4 KNN

```python
def knn_predict(X_train, y_train, X_test, k=5):
    # (n_test, n_train) 距离矩阵；大数据要分块避免 OOM
    d = ((X_test[:, None, :] - X_train[None]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]          # O(n) 选 topk
    votes = y_train[idx]
    return np.array([np.bincount(v).argmax() for v in votes])
```

### 4.5 决策树的基尼/信息增益

```python
def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()

def best_split(X, y):
    best = (None, None, -1)
    base = gini(y)
    for f in range(X.shape[1]):
        for thr in np.unique(X[:, f]):
            left = X[:, f] <= thr
            if left.sum() == 0 or left.sum() == len(y):
                continue
            g = (left.mean() * gini(y[left]) +
                 (1 - left.mean()) * gini(y[~left]))
            if base - g > best[2]:
                best = (f, thr, base - g)
    return best                                        # (特征, 阈值, 增益)
```

---

## 5. 目标检测 / CV 高频

```python
def nms(boxes, scores, iou_thr=0.5):
    """boxes: (N,4) [x1,y1,x2,y2]，按分数降序贪心抑制"""
    idx = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(idx) > 0:
        i = idx[0]; keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[idx[1:]] - inter)
        idx = idx[1:][iou <= iou_thr]
    return keep

def conv2d_naive(x, w, stride=1, pad=0):
    """x:(H,W) w:(k,k)，用于口述卷积实现与输出尺寸"""
    x = np.pad(x, pad)
    k = w.shape[0]
    H = (x.shape[0] - k) // stride + 1
    W = (x.shape[1] - k) // stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = (x[i*stride:i*stride+k, j*stride:j*stride+k] * w).sum()
    return out
```

---

## 6. LLM 工程实现

### 6.1 采样管线（temperature / top-k / top-p）
见 [[/docs/llm/decoding-strategies.md]]。

### 6.2 带 KV Cache 的生成
见 [[/docs/llm/kv-cache.md]]。

### 6.3 Beam Search（简版）

```python
import math

def beam_search(step_fn, bos, eos, beam=4, max_len=32, alpha=0.7):
    """step_fn(seq) -> log_probs over vocab（一维）"""
    beams = [([bos], 0.0)]                  # (序列, 累积 logprob)
    done = []
    for _ in range(max_len):
        cands = []
        for seq, score in beams:
            lp = step_fn(seq)
            topv, topi = lp.topk(beam)
            for v, i in zip(topv.tolist(), topi.tolist()):
                ns, nsc = seq + [i], score + v
                if i == eos:
                    done.append((ns, nsc / (len(ns) ** alpha)))   # 长度归一化
                else:
                    cands.append((ns, nsc))
        if not cands:
            break
        beams = sorted(cands, key=lambda x: -x[1])[:beam]
    if not done:
        done = [(s, sc / (len(s) ** alpha)) for s, sc in beams]
    return max(done, key=lambda x: x[1])[0]
```

### 6.4 LoRA 层
见 [[/docs/llm/sft-lora-peft.md]]。

### 6.5 DPO 损失
见 [[/docs/llm/rlhf-ppo-dpo.md]]。

### 6.6 KV Cache 显存计算器（口算题也常考）

```python
def kv_cache_bytes(layers, n_kv_heads, head_dim, batch, seq, dtype_bytes=2):
    """KV Cache = 2(K,V) × L × n_kv × d_head × b × s × bytes"""
    return 2 * layers * n_kv_heads * head_dim * batch * seq * dtype_bytes

# LLaMA-2 7B, 4k 上下文, batch 16
# kv_cache_bytes(32, 32, 128, 16, 4096) / 1e9 ≈ 34.4 GB
```

---

## 7. 其他常考小题

| 题目 | 关键点 |
|------|--------|
| Dropout 前向/反向 | inverted dropout：训练时 `mask/(1-p)`，推理直通 |
| BatchNorm 前向 + running stats | 训练/推理分支、有偏/无偏方差，见 [[/docs/dl/normalization-bn-ln-rmsnorm.md]] |
| Focal Loss | $-\alpha_t(1-p_t)^\gamma\log p_t$，见 [[/docs/dl/loss-functions.md]] |
| InfoNCE / CLIP 损失 | 对称交叉熵 + 温度，见 [[/docs/dl/loss-functions.md]] |
| Positional Encoding（正余弦） | 偶数维 sin、奇数维 cos，见 [[/docs/llm/positional-encoding.md]] |
| BPE 训练与编码 | 见 [[/docs/llm/tokenizer-bpe.md]] |
| RRF 融合 | 见 [[/docs/rag/retrieval-optimization-and-graphrag.md]] |
| HNSW 单层搜索 | 见 [[/docs/rag/vector-index-and-database.md]] |
| Top-k MoE 路由 + 负载均衡损失 | 见 [[/docs/llm/moe-mixture-of-experts.md]] |
| Adam / AdamW | 偏差修正 + 解耦 weight decay，见 [[/docs/dl/optimizers-and-lr-schedule.md]] |
| 分组量化/反量化 | 见 [[/docs/llm/quantization.md]] |
| LSTM 单元 | 一次矩阵乘算四门 + forget bias=1，见 [[/docs/dl/cnn-rnn-lstm-vs-transformer.md]] |

**LeetCode 侧**：算法岗仍会考中等难度题，重点是二分、双指针、DFS/BFS、动态规划（编辑距离/最长上升子序列）、堆（TopK）、滑动窗口、前缀和、字符串处理。建议按"高频 100 题 + 手撕 ML 组件"两条线准备。

---

## 8. 面试高频问题速查

1. **默写多头注意力** → 见 §1，注意形状变换与 `-inf` mask。
2. **为什么除 $\sqrt{d_k}$** → 点积方差随维度线性增长，缩放防 softmax 进饱和区。
3. **softmax 怎么防溢出** → 减去最大值；交叉熵用 log-sum-exp。
4. **手写 BN 要注意什么** → 训练/推理分支、running stats 用无偏方差、归一化用有偏方差。
5. **手写 AUC** → 秩公式，处理并列取平均秩，单类别返回 NaN。
6. **手写 NMS** → 按分数排序贪心 + IoU 抑制，注意面积与交集为 0 的处理。
7. **手写 KMeans** → KMeans++ 初始化 + 空簇处理 + SSE 收敛判据。
8. **手写 LoRA** → A 高斯 B 零初始化、$\alpha/r$ 缩放、支持 merge。
9. **手写采样管线** → 惩罚 → temperature → top-k → top-p → multinomial。
10. **手算 KV Cache** → $2Ln_{kv}d_{head}bs\cdot\text{bytes}$。
11. **手写 DPO loss** → $-\log\sigma(\beta(\Delta\log\pi - \Delta\log\pi_{ref}))$，注意只统计 answer token。
12. **写代码时怎么体现工程素养** → 形状注释、数值稳定、边界处理、复杂度说明、可测试的小例子。

---

## 参考

- 各条目对应的原理文档见本仓库 [[/docs/interview/bagu-knowledge-map.md]]
- PyTorch 官方文档（`scaled_dot_product_attention`、`clip_grad_norm_` 等）
- 《动手学深度学习》相关章节实现
