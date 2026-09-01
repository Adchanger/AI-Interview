# 算法八股知识点总清单（面试地图）

> **更新时间**：2026-08-31

> **标签**：面试八股、知识地图、机器学习、深度学习、大模型

> **一句话**：把「经典机器学习 + 深度学习基础 + 大模型（原理 / 训练对齐 / 推理部署）+ RAG / Agent + 手撕代码」六大板块的高频八股知识点收敛成一张可勾选的清单，每个知识点对应一篇可背诵的详解文档。

---

## 0. 怎么用这张清单

1. **先按岗位裁剪**：算法岗（ML/DL/LLM 原理 + 训练对齐权重高）、应用开发岗（RAG + Agent + 推理部署权重高）、Infra 岗（分布式训练 + 推理服务 + 量化权重高）。
2. **再按「能不能讲 3 分钟」自测**：每个知识点问自己三层——**是什么 → 为什么这样设计 → 换成别的方案会怎样**。答不出第二三层的，就是要补的洞。
3. **最后刷「面试高频问题速查」**：每篇文档末尾都有 8–12 题速查表，考前只看这一块。

> 清单来源：2026 年各家大厂面经与公开题库的交集（卡码笔记大模型面经、JavaGuide RAG/Agent 题库、LLMInterviewQuestions 115 题、后训练算法八股 152 问 Checklist 等），并按「原理可考、答案可背」二次筛选。

---

## 1. 经典机器学习（ML）

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 1.1 | 偏差-方差分解、过拟合/欠拟合、L1/L2 正则 | 「过拟合怎么判断怎么解决」「L1 为什么产生稀疏」 | [[/docs/ml/bias-variance-and-regularization.md]] |
| 1.2 | 逻辑回归 | 「为什么用交叉熵不用 MSE」「LR 为什么要归一化」 | [[/docs/ml/logistic-regression.md]] |
| 1.3 | SVM | 「为什么要转对偶」「核函数怎么选」「hinge loss」 | [[/docs/ml/svm.md]] |
| 1.4 | 决策树 / RF / GBDT / XGBoost / LightGBM | 「XGB 与 GBDT 区别」「LGBM 为什么快」 | [[/docs/ml/tree-ensemble-gbdt-xgboost.md]] |
| 1.5 | 评估指标：P/R/F1、ROC-AUC、PR-AUC、KS | 「AUC 的概率含义」「AUC 与 PR 曲线怎么选」 | [[/docs/ml/model-evaluation-metrics.md]] |
| 1.6 | 特征工程、归一化、类别不平衡 | 「哪些模型需要归一化」「样本不平衡怎么办」 | [[/docs/ml/feature-engineering-and-imbalance.md]] |
| 1.7 | 无监督：KMeans、GMM+EM、PCA/SVD、t-SNE | 「KMeans 缺陷」「EM 为什么收敛」「PCA 与 SVD 关系」 | [[/docs/ml/unsupervised-clustering-and-dimensionality.md]] |

## 2. 深度学习基础（DL）

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 2.1 | BN / LN / RMSNorm / GN、Pre-LN vs Post-LN | 「NLP 为什么不用 BN」「RMSNorm 省了什么」 | [[/docs/dl/normalization-bn-ln-rmsnorm.md]] |
| 2.2 | 激活函数：Sigmoid→ReLU→GELU→SwiGLU | 「ReLU 为什么缓解梯度消失」「SwiGLU 为什么 8/3 倍宽」 | [[/docs/dl/activation-functions.md]] |
| 2.3 | 优化器与学习率：SGDM / Adam / AdamW、warmup、cosine | 「Adam 与 AdamW 区别」「为什么必须 warmup」 | [[/docs/dl/optimizers-and-lr-schedule.md]] |
| 2.4 | 梯度消失/爆炸、残差连接、初始化、梯度裁剪 | 「残差为什么能训很深」「Xavier vs He」 | [[/docs/dl/gradient-vanishing-exploding-residual.md]] |
| 2.5 | CNN / RNN / LSTM / GRU 与 Transformer 对比 | 「感受野与参数量计算」「LSTM 三门作用」 | [[/docs/dl/cnn-rnn-lstm-vs-transformer.md]] |
| 2.6 | 损失函数：CE、MSE、Focal、Label Smoothing、对比损失 | 「Focal Loss 两个超参」「标签平滑为什么有效」 | [[/docs/dl/loss-functions.md]] |

## 3. 大模型原理（LLM Core）

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 3.1 | Transformer 与自注意力 | 「为什么除 √d_k」「复杂度」 | [[/docs/llm/transformer-principle.md]] |
| 3.2 | 位置编码：正余弦 / 可学习 / RoPE / ALiBi | 「RoPE 为什么能外推」 | [[/docs/llm/positional-encoding.md]] |
| 3.3 | 注意力变体：MHA / MQA / GQA / MLA | 「GQA 折中了什么」「KV 头数怎么选」 | [[/docs/llm/attention-variants-mha-mqa-gqa.md]] |
| 3.4 | KV Cache：原理、显存公式、prefill/decode | 「手算 KV Cache 大小」「为什么 decode 是带宽瓶颈」 | [[/docs/llm/kv-cache.md]] |
| 3.5 | MLA（低秩 KV 压缩） | 「MLA 与 GQA 的差别」 | [[/docs/llm/mla-multi-head-latent-attention.md]] |
| 3.6 | MoE：路由、Top-k、负载均衡、专家塌陷 | 「MoE 为什么省算力不省显存」 | [[/docs/llm/moe-mixture-of-experts.md]] |
| 3.7 | Tokenizer：BPE / BBPE / WordPiece / SentencePiece | 「词表大小怎么权衡」「中文为什么吃亏」 | [[/docs/llm/tokenizer-bpe.md]] |
| 3.8 | 架构选型：Encoder-only / Decoder-only / Enc-Dec | 「为什么现在都是 Decoder-only」 | [[/docs/llm/llm-architecture-decoder-only.md]] |
| 3.9 | 预训练与 Scaling Law（Chinchilla、数据配比、涌现） | 「给定算力怎么分参数和数据」 | [[/docs/llm/pretraining-and-scaling-law.md]] |
| 3.10 | 长上下文：外推、YaRN、稀疏/线性注意力、FlashAttention | 「FlashAttention 为什么快」「Lost in the Middle」 | [[/docs/llm/long-context-and-flashattention.md]] |
| 3.11 | 解码策略：greedy / beam / top-k / top-p / temperature | 「temperature 与 top-p 谁先作用」 | [[/docs/llm/decoding-strategies.md]] |
| 3.12 | MTP 多 token 预测 | 「MTP 与投机解码的关系」 | [[/docs/llm/mtp-multi-token-prediction.md]] |

## 4. 训练与对齐（Post-training）

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 4.1 | SFT + PEFT：LoRA / QLoRA、灾难性遗忘 | 「LoRA 的 r 和 alpha」「为什么能 merge」 | [[/docs/llm/sft-lora-peft.md]] |
| 4.2 | RLHF：RM、PPO、DPO、reward hacking、PRM/ORM | 「DPO 为什么不用 RM」「KL 惩罚作用」 | [[/docs/llm/rlhf-ppo-dpo.md]] |
| 4.3 | GRPO | 「GRPO 去掉 critic 靠什么估 baseline」 | [[/docs/llm/grpo-group-relative-policy-optimization.md]] |
| 4.4 | 推理模型与 Test-Time Scaling（o1/R1、RLVR、CoT） | 「长思维链怎么训出来」「self-consistency」 | [[/docs/llm/reasoning-and-test-time-scaling.md]] |
| 4.5 | 量化：PTQ/QAT、GPTQ/AWQ/SmoothQuant、FP8、W4A16 | 「为什么权重能 4bit 激活不能」 | [[/docs/llm/quantization.md]] |
| 4.6 | 幻觉与评测：成因、缓解、LLM-as-Judge、数据污染 | 「怎么评一个对话模型」 | [[/docs/llm/hallucination-and-evaluation.md]] |

## 5. RAG 与 Agent（应用层）

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 5.1 | RAG 全链路、与微调/长上下文取舍 | 「RAG vs 微调怎么选」「RAG 会不会幻觉」 | [[/docs/rag/rag-basics.md]] |
| 5.2 | Embedding、相似度度量、HNSW/IVF-PQ、向量库 | 「HNSW 与 IVF 区别」「ef_search 怎么调」 | [[/docs/rag/vector-index-and-database.md]] |
| 5.3 | 检索优化：chunking、hybrid、rerank、query rewrite、GraphRAG | 「召回率低怎么排查」 | [[/docs/rag/retrieval-optimization-and-graphrag.md]] |
| 5.4 | Agent 基础：ReAct、Plan-Execute、Function Calling、记忆 | 「ReAct 循环怎么终止」 | [[/docs/agent/agent-fundamentals.md]] |
| 5.5 | MCP / 多 Agent / 上下文工程 / 可观测性 | 「MCP 解决什么问题」「上下文怎么压缩」 | [[/docs/agent/mcp-multi-agent-context-engineering.md]] |

## 6. 工程与 Infra

| # | 知识点 | 高频考法 | 文档 |
|---|--------|----------|------|
| 6.1 | 分布式训练：DP/DDP/FSDP、ZeRO、TP/PP/EP、混合精度、显存估算 | 「7B 全参微调要多少显存」 | [[/docs/engineering/distributed-training.md]] |
| 6.2 | 推理服务：continuous batching、PagedAttention、投机解码、TTFT/TPOT | 「吞吐和延迟怎么权衡」 | [[/docs/engineering/inference-serving-optimization.md]] |
| 6.3 | 推理框架对比：vLLM / SGLang | 「RadixAttention 是什么」 | [[/docs/llm/sglang-vs-vllm.md]] |
| 6.4 | 手撕代码：Attention、MHA、BN、KMeans、AUC、beam search、NMS | 「10 分钟白板写多头注意力」 | [[/docs/interview/coding-must-write.md]] |

## 7. 多模态与其他方向

| # | 知识点 | 文档 |
|---|--------|------|
| 7.1 | VLM 演进（CLIP → BLIP-2 → LLaVA → Qwen-VL） | [[/docs/llm/vlm-evolution.md]] |
| 7.2 | DeepSeek 系列技术全景 | [[/docs/llm/deepseek-family.md]] |
| 7.3 | DeepSeek V4 vs V3/R1 | [[/docs/llm/deepseek-v4-vs-v3-r1.md]] |

---

## 8. 面试高频问题速查（跨模块 20 问）

1. **过拟合怎么判断、怎么解决？** → 训练集好测试集差；正则/数据增强/早停/dropout/简化模型 → §1.1
2. **L1 为什么稀疏、L2 为什么不稀疏？** → L1 等值线是菱形，最优解易落在坐标轴顶点 → §1.1
3. **AUC 的概率含义是什么？** → 随机正样本得分高于随机负样本的概率 → §1.5
4. **XGBoost 与 GBDT 的核心区别？** → 二阶泰勒 + 正则项 + 稀疏感知分裂 + 列采样 + 并行分桶 → §1.4
5. **NLP 为什么不用 BatchNorm？** → 变长序列 + 小 batch 统计量不稳；LN 按样本内特征维归一化 → §2.1
6. **Adam 与 AdamW 差别？** → AdamW 把 weight decay 从梯度里剥离为解耦权重衰减 → §2.3
7. **为什么大模型训练一定要 warmup？** → 初期二阶矩估计不准，大 lr 直接炸；线性升温稳住前几千步 → §2.3
8. **自注意力为什么除 √d_k？** → 点积方差随 d_k 线性增长，缩放后 softmax 不进饱和区 → §3.1
9. **RoPE 为什么比可学习 PE 更能外推？** → 旋转使点积只依赖相对位置，可配合 NTK/YaRN 插值 → §3.2
10. **GQA 相比 MHA/MQA 折中了什么？** → KV 头数从 h→g，KV Cache 降 h/g 倍，质量损失远小于 MQA → §3.3
11. **怎么手算 KV Cache 显存？** → `2 × L × n_kv × d_head × b × s × dtype_bytes` → §3.4
12. **MoE 为什么省算力但不省显存？** → 只激活 Top-k 专家（FLOPs 降），但全部专家权重都要驻留 → §3.6
13. **为什么主流是 Decoder-only？** → 训练目标信息密度高、结构简单可极致 scale、KV Cache 友好、zero-shot 强 → §3.8
14. **给定算力怎么分参数量和数据量？** → Chinchilla：参数与 token 同比放大，约 20 tokens/param → §3.9
15. **FlashAttention 为什么快？** → 分块 + online softmax，避免物化 n×n 注意力矩阵，IO-aware → §3.10
16. **DPO 相比 RLHF-PPO 省了什么？** → 省掉显式 RM 与在线采样，把偏好优化写成闭式分类损失 → §4.2
17. **为什么权重能量化到 4bit，激活不行？** → 权重分布集中、离线可校准；激活有动态离群值，需 SmoothQuant 等迁移 → §4.5
18. **RAG 召回不准怎么排查？** → 分层定位：解析→chunk→embedding→索引→召回→rerank→上下文→生成 → §5.3
19. **ReAct 循环怎么防死循环？** → 步数/预算上限 + 终止条件 + 重复动作检测 + 工具失败兜底 → §5.4
20. **7B 模型全参微调大概要多少显存？** → 参数+梯度+Adam 双状态 ≈ 16 bytes/param ≈ 112 GB（再加激活）→ §6.1

---

## 参考

- 卡码笔记《2026 最全大模型面经汇总》：<https://notes.kamacoder.com/interview/llm/>
- JavaGuide《RAG 面试题总结》：<https://javaguide.cn/ai/interview-questions/rag-interview-questions.html>
- LLMInterviewQuestions（100+ 题，Google/NVIDIA/Meta 等）：<https://github.com/llmgenai/LLMInterviewQuestions>
- 《LLM 后训练算法面试八股》：<https://tcsnyy.github.io/llm-interview-notes/>
