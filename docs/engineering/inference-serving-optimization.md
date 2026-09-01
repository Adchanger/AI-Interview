# 推理服务优化：吞吐、延迟与调度（工程八股 02）

> **更新时间**：2026-08-31

> **标签**：推理优化、ContinuousBatching、投机解码、PD分离、TTFT、面试八股

> **一句话**：LLM 推理优化围绕"**prefill 受算力限、decode 受带宽限**"这一条主线展开——continuous batching 提利用率、PagedAttention 省显存、prefix cache 免重算、投机解码把多步变一步、量化降带宽，最后用 PD 分离与调度策略在 TTFT / TPOT / 吞吐之间取舍。

> **关联阅读**：[[/docs/llm/kv-cache.md]]、[[/docs/llm/sglang-vs-vllm.md]]、[[/docs/llm/quantization.md]]

---

## 1. 指标体系（先把话说清楚）

| 指标 | 含义 | 主要影响因素 |
|------|------|-------------|
| **TTFT**（Time To First Token） | 首 token 延迟 | prompt 长度、prefill 算力、排队时间、prefix cache 命中 |
| **TPOT / ITL** | 每个后续 token 的延迟 | 显存带宽、batch 大小、KV Cache 大小、投机解码 |
| **端到端延迟** | TTFT + TPOT × 输出长度 | 上述全部 + 输出长度 |
| **吞吐** | tokens/s 或 requests/s | batch 并发度（受 KV 显存限制）、调度效率 |
| **Goodput** | 满足 SLO 前提下的有效吞吐 | 调度策略、优先级 |
| MFU / 显存利用率 | 硬件效率 | kernel 效率、碎片、batch |

**核心矛盾**：**吞吐与延迟不可兼得**——增大 batch 提吞吐但每个请求排队与单步时间变长。工程上按业务定 SLO（如 TTFT < 500ms、TPOT < 50ms），在满足 SLO 下最大化吞吐。

---

## 2. 批处理与调度

### 2.1 Static batching 的问题

传统批处理要等一批全部完成才释放：短请求被长请求"拖着"，GPU 大量空转（padding 与等待）。

### 2.2 Continuous Batching（连续批处理）

**迭代级调度**：每完成一个 decode step 就检查——有请求结束就移出、有新请求就插入，batch 组成动态变化。

- 由 **Orca** 提出，vLLM/TGI/TensorRT-LLM 都实现；
- 相比 static batching 吞吐可提升数倍（尤其输出长度差异大时）；
- 与 PagedAttention 配合才能真正做到：因为动态进出需要**灵活的 KV 显存分配**。

### 2.3 Chunked Prefill 与调度策略

- **问题**：长 prompt 的 prefill 会占满 GPU 一段时间，让正在 decode 的请求"卡住"（TPOT 抖动）；
- **Chunked prefill**：把长 prefill 切块，与 decode 步交错执行，平滑延迟（Sarathi-Serve 的核心思想）；
- **调度策略**：FCFS（公平但长请求拖尾）、优先级/SLO 感知、抢占（显存不足时把低优先级请求的 KV **换出/重算**，vLLM 的 preemption）、按输出长度预测排序（减少 head-of-line blocking）。

### 2.4 PD 分离（Prefill/Decode Disaggregation）

把 prefill 与 decode 放**不同实例/不同卡池**：
- prefill 集群算力型（大 batch 提高 MFU）、decode 集群带宽型（大并发摊薄权重读取）；
- 二者互不干扰 → TTFT 与 TPOT 都更稳；
- 代价：需把 KV Cache 从 prefill 传到 decode（走 NVLink/RDMA），实现复杂；
- 2025–2026 年成为大规模服务的主流架构（DeepSeek、vLLM/SGLang 均支持）。

---

## 3. 显存与缓存

| 技术 | 收益 |
|------|------|
| **PagedAttention** | KV 分块 + 页表，消除碎片，显存利用率接近 100%，支持共享与 copy-on-write |
| **Prefix Caching / RadixAttention** | 相同前缀（系统提示、few-shot、多轮历史）只 prefill 一次，命中时 TTFT 骤降；SGLang 用 Radix 树自动管理，见 [[/docs/llm/sglang-vs-vllm.md]] |
| **KV 量化（int8/fp8）** | Cache 减半/减四分之一 → 并发提升 |
| **GQA / MLA** | 架构级减小 Cache，收益最大且与上面全部可叠加 |
| CPU/NVMe offload | 容量换带宽，仅适合离线/低 SLO 场景 |
| 语义缓存 | 相似 query 直接返回缓存答案（需谨慎，语义相似≠答案相同） |

![RadixAttention 的前缀共享](../images/sglang-vllm-radixattention-01.png)

图1：RadixAttention 用基数树自动复用共享前缀的 KV（来源：SGLang，arXiv:2312.07104）

---

## 4. 解码加速

### 4.1 投机解码（Speculative Decoding）

**流程**：草稿模型（小模型）一次生成 $\gamma$ 个候选 token → 目标模型**一次前向并行验证**这 $\gamma+1$ 个位置 → 用修正的接受-拒绝采样保留前缀。

- **为什么有效**：decode 是带宽受限的，一次前向验证多个 token 几乎不增加时间成本 → 用"闲置算力"换步数；
- **无损性**：标准算法保证输出分布与目标模型一致；
- **加速比**取决于**接受率**（草稿与目标的一致程度）与草稿成本；简单文本/代码接受率高，收益大（常 1.5–3×）。

**变体**：
| 方案 | 草稿来源 |
|------|----------|
| 小模型草稿 | 同族小模型（如 1B 草稿 + 70B 目标） |
| **Medusa** | 在目标模型上加多个预测头，无需独立小模型 |
| **EAGLE / EAGLE-2/3** | 在特征层做自回归草稿，接受率更高，当前主流之一 |
| **MTP** | 训练时就学多 token 预测，天然自投机，见 [[/docs/llm/mtp-multi-token-prediction.md]] |
| **Lookahead / Prompt lookup** | 从上下文里直接找 n-gram 当草稿（RAG/长文摘要场景极有效，零成本） |

### 4.2 其他

- **算子融合与高效 kernel**：FlashAttention/FlashDecoding、fused RMSNorm+残差、fused SwiGLU、CUDA Graph 消除 launch 开销（小 batch 收益大）；
- **张量并行推理**：多卡切分降低单卡权重读取量，也降低单卡 KV 压力；
- **批内长度分桶**：把长度相近的请求组 batch，减少 padding 浪费（连续批处理下影响变小但仍存在）；
- **结构化输出加速**：约束解码可跳过确定性 token（如 JSON 的固定字段名）——jump-forward 解码。

---

## 5. 框架与部署选择

| 框架 | 定位 |
|------|------|
| **vLLM** | 通用性最好、生态最广，PagedAttention 起家，功能全面（LoRA、量化、PD 分离、多模态） |
| **SGLang** | RadixAttention 前缀复用 + 结构化输出/复杂调用编排见长，高并发共享前缀场景优势明显 |
| **TensorRT-LLM** | NVIDIA 官方，kernel 极致优化，latency 敏感场景强，但编译/部署较重 |
| llama.cpp / MLC | 端侧/CPU/苹果芯片 |
| Hugging Face TGI | 生态成熟，工程化早 |

**选型经验**：先看是否有大量共享前缀（→SGLang/前缀缓存）、是否极致低延迟（→TensorRT-LLM）、是否需要快速迭代与广泛模型支持（→vLLM）。

---

## 6. 容量规划（面试实战题）

给定 **H100 80GB、Qwen 7B（bf16）、上下文 8k、目标并发 32**：

1. 权重：7B × 2B ≈ 14 GB；
2. 框架/激活/碎片预留：约 6–10 GB；
3. 可用 KV 预算：80 − 14 − 8 ≈ 58 GB；
4. 单序列 KV（假设 GQA，$n_{kv}=4,\ d_{head}=128,\ L=28$）：每 token = $2\times4\times128\times2\times28 \approx 57$ KB → 8k token ≈ 0.46 GB；
5. 可支持并发 ≈ 58 / 0.46 ≈ **126** → 满足 32 并发有余，可考虑提高 batch 或上更长上下文；
6. 若换 MHA（$n_{kv}=28$）→ 单序列约 3.2 GB → 并发仅 ~18，**不达标** → 说明 GQA 的关键作用。

**答题模板**：算权重 → 扣预留 → 算单序列 KV → 除法得并发 → 用 GQA/量化/分页调节。

---

## 7. 面试高频问题速查

1. **prefill 与 decode 的瓶颈差异？** → prefill 算力受限（大矩阵乘），decode 带宽受限（读权重 + KV）。
2. **TTFT 和 TPOT 分别怎么优化？** → TTFT：prefix cache、chunked prefill、减少排队、TP；TPOT：减小 KV（GQA/量化）、增大 batch、投机解码、CUDA Graph。
3. **continuous batching 是什么？** → 迭代级调度，逐 step 动态加入/移出请求，避免等齐一批。
4. **它为什么需要 PagedAttention？** → 动态进出要求非连续、按需的 KV 分配。
5. **chunked prefill 解决什么？** → 长 prompt 的 prefill 阻塞 decode 造成的 TPOT 抖动。
6. **PD 分离的动机与代价？** → 两阶段资源特性不同，分离后互不干扰；代价是 KV 跨实例传输与架构复杂度。
7. **投机解码原理与是否无损？** → 小模型提议 + 大模型并行验证；标准算法无损，接受率决定加速比。
8. **Medusa / EAGLE / MTP 的区别？** → 多头预测 / 特征层自回归草稿 / 训练即多 token 预测；都免去独立草稿模型。
9. **prefix caching 什么场景收益最大？** → 长系统提示、few-shot、多轮对话、批量同模板请求。
10. **吞吐与延迟怎么权衡？** → 定 SLO 后在满足前提下最大化 batch；用 goodput 而非裸吞吐衡量。
11. **显存不够时框架怎么处理？** → 抢占：把低优先级请求 KV 换出或丢弃后重算（recompute/swap）。
12. **怎么做容量规划？** → 权重 + 预留 + 单序列 KV → 并发上限；用 GQA/量化/分页调节。
13. **vLLM 与 SGLang 怎么选？** → 通用与生态选 vLLM；大量共享前缀、复杂结构化输出选 SGLang；极致低延迟考虑 TensorRT-LLM。

---

## 参考

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022（continuous batching）
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention (vLLM)*, arXiv:2309.06180
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, arXiv:2403.02310
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192
- Cai et al., *Medusa*, arXiv:2401.10774；Li et al., *EAGLE*, arXiv:2401.15077
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding*, arXiv:2401.09670
