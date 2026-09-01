# 量化：PTQ / QAT、GPTQ / AWQ / SmoothQuant 与 FP8（LLM 八股 21）

> **更新时间**：2026-08-31

> **标签**：量化、GPTQ、AWQ、SmoothQuant、FP8、面试八股

> **一句话**：量化用低比特表示权重/激活/KV 来省显存和带宽；**权重好量化、激活难量化**（离群值），于是有 W4A16 路线（GPTQ/AWQ，主打显存与解码带宽）与 W8A8 路线（SmoothQuant/FP8，主打算力吞吐），KV Cache 量化则是长上下文时代的新重点。

> **关联阅读**：[[/docs/llm/kv-cache.md]]、[[/docs/engineering/inference-serving-optimization.md]]、[[/docs/llm/sft-lora-peft.md]]

---

## 1. 基础：量化的数学

**均匀（affine）量化**：

$$q = \mathrm{round}\!\Big(\frac{x}{s}\Big) + z,\qquad \hat x = s\,(q - z)$$

- **对称量化**（$z=0$）：$s=\frac{\max|x|}{2^{b-1}-1}$，实现简单、乘法友好，权重常用；
- **非对称量化**：有 zero-point，能更好利用范围（激活/有偏分布常用）；
- **粒度**：per-tensor（最粗、最快）→ **per-channel/per-column**（权重常用）→ **per-group**（如 group_size=128，4bit 主流）→ per-token（激活动态量化常用）。粒度越细精度越好、开销越大。

**显存收益**：fp16 → int8 减半，→ int4 减到 1/4。7B 模型：fp16 ≈ 14GB，int4 ≈ 3.5–4GB（含量化元数据）。

### 1.1 为什么权重能 4bit，激活不行

| | 权重 | 激活 |
|--|------|------|
| 分布 | 接近正态、范围稳定 | 存在**系统性离群值**（某些通道幅值大几十倍） |
| 可否离线校准 | ✅ 静态、可精细优化 | 动态随输入变化 |
| 结论 | 4bit 可行（配 group-wise + 补偿） | 8bit 已需技巧，4bit 极难 |

**离群值现象**（Dettmers, LLM.int8()）：模型超过 ~6.7B 后，少数隐藏维度出现极大激活值，且对性能至关重要；直接 per-tensor int8 量化会崩。

---

## 2. PTQ vs QAT

| | **PTQ**（训练后量化） | **QAT**（量化感知训练） |
|--|----------------------|------------------------|
| 成本 | 少量校准数据（128~512 条）、几分钟到几小时 | 需要重新训练/微调，昂贵 |
| 精度 | 8bit 几乎无损；4bit 需好算法 | 更好，尤其极低比特（2–3bit） |
| LLM 现状 | **绝对主流**（GPTQ/AWQ/SmoothQuant/FP8） | 少用；QLoRA 属于"量化 + 适配器微调"，介于两者之间 |
| 关键技巧 | 逐层误差补偿、离群值处理、混合精度 | STE（直通估计器）绕过 round 不可导 |

---

## 3. 权重量化算法（W4A16 路线）

### 3.1 GPTQ（Frantar et al. 2022）

- 基于 **OBQ/OBS 二阶信息**：逐列量化权重，并用 Hessian（由校准数据的输入相关矩阵 $H=2XX^\top$ 近似）**把量化误差补偿到尚未量化的列上**；
- 用 Cholesky 分解 + 惰性批更新做到大模型可行（175B 约几小时）；
- 精度好，是最早让 175B int4 单机推理可行的方案；
- 对**校准集有一定敏感性**，group_size 常取 128，可选 act-order（按激活重要性排序）。

### 3.2 AWQ（Lin et al. 2023，MLSys 2024 最佳论文）

- 洞见：**只有约 1% 的"显著权重"决定精度**，而显著性由**激活幅值**（不是权重幅值）决定；
- 做法：不做混合精度（对硬件不友好），而是**按激活分布对权重做逐通道缩放**（等价数学变换：$W\cdot s$ 与 $x/s$），把重要通道"放大"后再量化，降低其相对量化误差；
- 优点：**不依赖反向/二阶信息，无需重训**，对校准集更鲁棒，泛化到指令模型与多模态更稳，推理 kernel 高效；
- 现已是开源部署最常用的 W4A16 方案之一。

### 3.3 其他

| 方案 | 特点 |
|------|------|
| **GGUF / llama.cpp（Q4_K_M 等）** | k-quants 混合方案，CPU/端侧首选，块内多级 scale |
| **bitsandbytes NF4** | QLoRA 用的 4bit 数据类型，微调场景 |
| **HQQ** | 无需校准数据的快速量化 |
| **SpQR / QuIP# / AQLM** | 极低比特（2–3bit）研究方向，用稀疏离群保留、格量化、加性量化 |
| **EXL2/EXL3** | 混合比特、按层分配，追求单卡极致 |

> 面试高频：**GPTQ 和 AWQ 怎么选？** → 追求极致精度且能接受校准敏感性 → GPTQ（尤其 act-order）；追求鲁棒、部署稳定、指令/多模态模型 → AWQ。实际以目标硬件的 kernel 支持与实测精度为准。

---

## 4. 激活量化与 W8A8 路线

### 4.1 SmoothQuant（Xiao et al. 2022）

难点在激活离群值。核心是**难度迁移**：用逐通道缩放 $s$ 做等价变换

$$Y = (X\,\mathrm{diag}(s)^{-1})\cdot(\mathrm{diag}(s)\,W)$$

把激活的离群幅值"搬"到权重侧（权重更耐量化），$s_j=\frac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}$（$\alpha$ 常取 0.5 平衡两侧难度）。缩放可**融合进前一层的 LayerNorm/线性层**，推理零额外开销 → 实现 **W8A8** 且几乎无损。

### 4.2 LLM.int8()

对离群维度保留 fp16、其余 int8（混合精度分解）。精度好但因需拆分矩阵乘，**速度往往不如 fp16**，现多用于教学/兜底。

### 4.3 FP8 与 FP4（硬件原生低精度）

| 格式 | 说明 |
|------|------|
| **FP8（E4M3 / E5M2）** | Hopper/Ada 起原生支持。E4M3 精度高（前向/权重激活），E5M2 动态范围大（梯度）。相比 INT8 **动态范围大得多**，通常无需复杂离群处理，per-tensor scale 即可 |
| **FP8 训练** | DeepSeek-V3 大规模 FP8 混合精度训练（细粒度 tile/block scaling + 关键路径保高精度），是重要工程里程碑 |
| **FP4 / NVFP4 / MXFP4** | Blackwell 起支持，带块级 scale（如每 16/32 元素一个 scale）；2025–2026 推理侧快速铺开，精度需按模型实测 |

![FP8 量化的细粒度缩放策略](../images/fp8-quantization-01.png)

图1：细粒度 FP8 量化（tile/block-wise scaling）（来源：DeepSeek-V3 技术报告，arXiv:2412.19437）

> 面试高频：**FP8 与 INT8 谁更好？** → 同为 8bit，FP8 动态范围大、对离群值宽容、量化流程简单（常可免校准），且新硬件有原生 tensor core 支持；INT8 在老硬件（A100/T4）与端侧生态更广。趋势是新卡走 FP8/FP4。

---

## 5. KV Cache 量化（长上下文时代的重点）

- 长上下文下 KV Cache 常**超过权重显存**，故 KV int8/fp8 已成推理引擎标配（vLLM/SGLang/TensorRT-LLM 均支持）；
- 关键细节：K 与 V 的分布特性不同（K 通道方向离群更明显）→ 常 **K 按通道、V 按 token** 量化；
- int4 KV 可行但需分组 + 保留少量高精度 token（如首尾/sink token），长上下文检索类任务掉点更明显，必须实测；
- 与 GQA/MLA 叠加使用，收益乘性。

---

## 6. 精度损失评估与实践路线

**评估不能只看 PPL**：PPL 变化 0.05 可能对应 benchmark 掉几个点，尤其**数学/代码/长上下文/多语言**更敏感。应跑：MMLU 类知识、GSM8K/MATH 推理、HumanEval 代码、长上下文 NIAH、以及**业务自建评测集**。

**推荐决策路线**：
1. 显存够 → **不量化**（bf16），最稳；
2. 新卡（H100/H200/B200）追吞吐 → **FP8 W8A8（+FP8 KV）**；
3. 老卡/显存紧、要装大模型 → **W4A16（AWQ/GPTQ）+ int8 KV**；
4. 端侧/CPU → **GGUF Q4_K_M / Q5_K_M**；
5. MoE 模型 → 按**总参数**估显存，专家权重量化收益最大，路由器保高精度；
6. 微调场景显存不足 → **QLoRA（NF4）**，见 [[/docs/llm/sft-lora-peft.md]]。

---

## 7. 手撕代码：对称/分组量化与反量化

```python
import torch

def quantize_per_group(w: torch.Tensor, bits=4, group_size=128, symmetric=True):
    """w: (out, in)，沿输入维分组量化，返回 (q, scale, zero)"""
    out, cin = w.shape
    assert cin % group_size == 0
    wg = w.reshape(out, cin // group_size, group_size)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(-1, keepdim=True) / qmax
        zero = torch.zeros_like(scale)
        q = (wg / scale).round().clamp(-qmax - 1, qmax)
    else:
        qmax = 2 ** bits - 1
        mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-8) / qmax
        zero = (-mn / scale).round()
        q = (wg / scale + zero).round().clamp(0, qmax)
    return q.to(torch.int8), scale, zero

def dequantize(q, scale, zero, shape):
    return ((q.float() - zero) * scale).reshape(shape)

# 量化误差评估：相对 Frobenius 误差
def q_error(w, bits=4, group_size=128):
    q, s, z = quantize_per_group(w, bits, group_size)
    w_hat = dequantize(q, s, z, w.shape)
    return (w - w_hat).norm() / w.norm()
```

---

## 8. 面试高频问题速查

1. **量化的收益是什么？** → 省显存（装更大模型/更多并发）+ 省带宽（decode 提速），部分格式还能用低精度 tensor core 提算力。
2. **对称与非对称量化区别？** → 有无 zero-point；对称实现快，非对称更贴合偏斜分布。
3. **量化粒度有哪些？** → per-tensor / per-channel / per-group / per-token，越细越准越慢。
4. **为什么权重能 4bit 而激活不行？** → 权重分布稳定可离线优化；激活有动态离群值，需迁移或更高比特。
5. **GPTQ 的核心思想？** → 基于二阶（Hessian）信息逐列量化并把误差补偿到未量化列。
6. **AWQ 的核心思想？** → 显著权重由激活幅值决定，用逐通道缩放保护重要通道，无需二阶信息、更鲁棒。
7. **SmoothQuant 解决什么？** → 用等价缩放把激活离群难度迁移到权重，实现 W8A8 近无损，缩放可融合进前层。
8. **FP8 相比 INT8 的优势？** → 动态范围大、对离群宽容、流程简单、新硬件原生支持；且可用于训练。
9. **W4A16 与 W8A8 的适用差异？** → W4A16 主打显存与 decode 带宽（低并发/单卡大模型）；W8A8 主打算力吞吐（高并发、prefill 重）。
10. **KV Cache 量化注意什么？** → K/V 分布不同需不同粒度；长上下文精度更敏感；与 GQA/MLA 叠加。
11. **QAT 在 LLM 上常用吗？** → 不常用（成本高）；QLoRA 是"量化底座 + 高精度 LoRA"的折中。
12. **怎么验证量化没掉点？** → PPL 只做初筛，必须跑推理/代码/长上下文与业务评测集，关注最差子任务。
13. **MoE 量化有什么特别之处？** → 显存按总参数算，专家权重量化收益最大；路由器/norm 保持高精度。

---

## 参考

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, arXiv:2210.17323
- Lin et al., *AWQ: Activation-aware Weight Quantization*, arXiv:2306.00978（MLSys 2024 最佳论文）
- Xiao et al., *SmoothQuant*, arXiv:2211.10438
- Dettmers et al., *LLM.int8()*, arXiv:2208.07339
- Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433
- DeepSeek-AI, *DeepSeek-V3 Technical Report*（FP8 训练）, arXiv:2412.19437
