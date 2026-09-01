# 分布式训练：并行策略与显存估算（工程八股 01）

> **更新时间**：2026-08-31

> **标签**：分布式训练、ZeRO、张量并行、流水线并行、显存估算、面试八股

> **一句话**：单卡装不下就切——**数据并行**切样本、**ZeRO/FSDP** 切优化器状态与参数、**张量并行**切矩阵（通信最重，只在机内）、**流水线并行**切层（有 bubble）、**专家并行**切专家；显存则按"参数 + 梯度 + 优化器状态 + 激活"四项估算，混合精度 AdamW 约 **16 bytes/param**。

> **关联阅读**：[[/docs/dl/optimizers-and-lr-schedule.md]]、[[/docs/llm/moe-mixture-of-experts.md]]、[[/docs/llm/pretraining-and-scaling-law.md]]

---

## 1. 显存去哪了（必背估算）

以 7B 模型、混合精度（bf16 计算 + fp32 主参数）+ AdamW 为例，**每个参数**：

| 项 | 精度 | 字节/参数 |
|----|------|----------|
| 参数（bf16 副本） | bf16 | 2 |
| 梯度（bf16） | bf16 | 2 |
| **fp32 主参数** | fp32 | 4 |
| Adam 一阶动量 $m$ | fp32 | 4 |
| Adam 二阶动量 $v$ | fp32 | 4 |
| **合计** | | **16 bytes/param** |

→ 7B 全参训练：$7\times10^9\times16 \approx 112$ GB（**还没算激活**）→ 单张 80GB 卡装不下，必须切分。

**推理**只需参数：bf16 约 **2 bytes/param**（7B ≈ 14GB）+ KV Cache，见 [[/docs/llm/kv-cache.md]]。

### 1.1 激活显存

激活量正比于 $b\times s\times d\times L$，且与实现细节（是否重计算、是否 FlashAttention）强相关。控制手段：

| 手段 | 效果 |
|------|------|
| **梯度检查点 / 激活重计算** | 只存每层边界，反向重算，显存降到约 $O(\sqrt L)$ 或按层线性下降；**多约 30% 计算量** |
| **FlashAttention** | 不物化 $n^2$ 注意力矩阵，长序列激活大降 |
| 梯度累积 | 降低 micro-batch 从而降激活（不改变有效 batch） |
| 序列并行（SP） | 把 LayerNorm/dropout 等非 TP 部分的激活也沿序列维切分 |
| 激活 offload | 换到 CPU，省显存但吃 PCIe 带宽 |

---

## 2. 并行策略五种

### 2.1 数据并行（DP / DDP）

每卡一份完整模型副本，各喂不同数据，反向后 **All-Reduce 梯度**（通信量 ≈ 2×参数量，Ring All-Reduce）。

- `DataParallel`（单进程多线程）已废弃 → 用 **DDP**（多进程，每卡一进程，梯度桶重叠通信与计算）；
- 优点：实现简单、扩展性好；缺点：**显存不省**（每卡都要全套 16 bytes/param）。

### 2.2 ZeRO / FSDP（分片数据并行）

把 DP 中冗余的状态切开（DeepSpeed ZeRO，PyTorch FSDP 同思想）：

| 阶段 | 切分对象 | 显存（N 卡） | 额外通信 |
|------|----------|-------------|----------|
| **ZeRO-1** | 优化器状态 | 4 + 12/N | 与 DDP 相当（+ReduceScatter/AllGather 参数更新） |
| **ZeRO-2** | + 梯度 | 2 + 14/N | 约同 DDP |
| **ZeRO-3 / FSDP-FULL_SHARD** | + **参数** | 16/N | 前向/反向都要 All-Gather 参数，通信约 1.5× |
| ZeRO-Offload / Infinity | 状态放 CPU / NVMe | 更省 | 带宽受限，慢 |

> 面试高频：**ZeRO-3 与张量并行的区别？** → ZeRO-3 是**按数据并行维度切参数**，计算时临时 All-Gather 回完整权重再算（数学上等价于 DP）；TP 是**把单个矩阵乘切开**，每卡只算一部分并在层内同步激活。ZeRO-3 通信在参数（可预取重叠），TP 通信在激活（每层多次、延迟敏感）。

### 2.3 张量并行（TP，Megatron-LM）

把权重矩阵切开，层内并行：

- **列并行**（$W=[W_1,W_2]$）：$XW_i$ 各算一半 → 输出拼接；
- **行并行**（按行切）：输入切分，输出需 **All-Reduce** 相加；
- Megatron 的经典组合：FFN 的第一层列并行、第二层行并行 → **每个 Transformer 层前向 2 次 All-Reduce、反向 2 次**；注意力按头切分（天然并行）。

**特点**：每层都要通信 → **必须在单机内用 NVLink**（TP 通常 ≤ 8，等于机内卡数）；能有效降低单卡参数与激活。

**序列并行（SP）**常与 TP 搭配：把 TP 不覆盖的 LayerNorm/Dropout 部分沿序列维切，进一步省激活。

### 2.4 流水线并行（PP）

按层切成 stage，不同卡持有不同层，数据以 **micro-batch** 流过。

- **Bubble（气泡）**：朴素方案空闲比例 $\frac{p-1}{m+p-1}$（$p$ stage 数、$m$ micro-batch 数）→ 增大 $m$ 可摊薄；
- 调度：GPipe（先全前向再全反向，激活显存大）→ **1F1B**（交错前向反向，显存更省）→ **Interleaved 1F1B**（虚拟 stage 进一步减 bubble）→ **Zero Bubble / DualPipe**（DeepSeek-V3：前向与反向计算通信重叠，近零气泡）；
- 通信量小（只传 stage 边界激活），适合跨机；
- 缺点：负载均衡难（层不等大）、实现复杂、对 batch 结构敏感。

![DualPipe 的双向流水调度](../images/dualpipe-scheduling-01.png)

图1：DeepSeek-V3 的 DualPipe 调度（前向/反向计算与通信重叠，近零气泡）（来源：DeepSeek-V3 技术报告，arXiv:2412.19437）

### 2.5 专家并行（EP）

MoE 专用：专家分布到不同设备，每层两次 **All-to-All**，见 [[/docs/llm/moe-mixture-of-experts.md]]。

### 2.6 组合：3D / 4D 并行

典型大模型训练配置（如 512 卡训 70B）：

```
TP=8（机内 NVLink） × PP=8（跨机） × DP=8（数据并行，配 ZeRO-1）
[MoE 模型再叠加 EP，长序列再叠加 CP/SP]
```

**排布原则**（很能体现经验）：
1. **TP 放机内**（通信最频繁、最需要 NVLink 带宽）；
2. **PP 跨机**（通信量最小，只传边界激活）；
3. **DP 最外层**（All-Reduce 可与反向计算重叠）；
4. TP×PP 先满足"单卡能装下"，剩余卡数给 DP 提吞吐；
5. 长序列加 **CP（Context/Sequence Parallel，如 Ring Attention）** 切序列维。

---

## 3. 混合精度与数值稳定

| 精度 | 说明 |
|------|------|
| **fp16 + loss scaling** | 动态范围小易下溢，必须放大 loss 再反向；已逐渐被 bf16 取代 |
| **bf16** | 指数位与 fp32 相同、动态范围大，**无需 loss scaling**，是当前默认 |
| **fp8**（H100+） | 前向/权重用 E4M3、梯度用 E5M2；需细粒度 scaling 与关键路径保高精度（DeepSeek-V3 大规模验证） |
| 保持 fp32 的部分 | 主参数副本、优化器状态、LayerNorm/softmax 归约、loss 计算 |

**稳定性工具箱**：梯度裁剪 1.0、warmup、z-loss、QK-Norm、跳过异常 batch、定期 checkpoint 便于回滚。

---

## 4. 通信与效率

- **集合通信原语**：All-Reduce（= ReduceScatter + AllGather）、All-Gather、ReduceScatter、All-to-All（MoE）、Broadcast；
- **NCCL** 是 GPU 上的实现；拓扑上机内 NVLink/NVSwitch（数百 GB/s~TB/s），跨机 IB/RoCE（数十~数百 Gb/s）；
- **计算通信重叠**是关键优化：DDP 梯度分桶、FSDP 参数预取、DualPipe 的双向重叠；
- **MFU（Model FLOPs Utilization）** = 有效 FLOPs / 峰值算力，大规模训练常 35%~55%；训练时间估算：
  $$T = \frac{6ND}{\text{GPU 数}\times\text{单卡峰值}\times \text{MFU}}$$

---

## 5. 微调场景的显存账（实用）

| 方案 | 7B 显存量级 | 说明 |
|------|------------|------|
| 全参 + AdamW | ~112 GB + 激活 | 需多卡 + ZeRO |
| 全参 + ZeRO-3（8 卡） | ~14 GB/卡 + 激活 | 通信换显存 |
| **LoRA**（bf16 底座） | ~14 GB + 少量 + 激活 | 单卡 A100 可行 |
| **QLoRA**（NF4 底座） | ~5–6 GB + 激活 | 单卡 24GB 消费卡可行 |
| + 梯度检查点 | 激活大幅下降 | 换 ~30% 计算 |

估算口诀：**全参 16、LoRA ≈ 冻结权重 2、QLoRA ≈ 0.5–0.6（bytes/param）**，再加激活与碎片。

---

## 6. 面试高频问题速查

1. **训练显存由哪几部分组成？** → 参数、梯度、优化器状态、激活（+ 碎片与通信缓冲）。
2. **为什么是 16 bytes/param？** → bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + Adam m/v 各 4。
3. **7B 全参微调需要多少显存？** → 约 112GB 状态 + 激活，单卡不可行，需 ZeRO/多卡。
4. **DDP 与 DP 的区别？** → DDP 多进程 + 梯度桶重叠通信，DP 单进程多线程已废弃。
5. **ZeRO 三个阶段切什么？** → 1 切优化器状态、2 加梯度、3 加参数；显存趋近 16/N。
6. **ZeRO-3 与 TP 有何本质不同？** → ZeRO-3 临时聚合完整权重（等价 DP），TP 真正切分矩阵乘且每层通信激活。
7. **张量并行为什么只在机内？** → 每层多次 All-Reduce，延迟与带宽敏感，必须 NVLink。
8. **Megatron 每层几次 All-Reduce？** → FFN 与注意力各一次，前向 2 次、反向 2 次。
9. **流水线的 bubble 怎么算、怎么减？** → $\frac{p-1}{m+p-1}$；增大 micro-batch 数、用 1F1B/Interleaved/Zero-Bubble(DualPipe)。
10. **3D 并行怎么排布？** → TP 机内、PP 跨机、DP 最外层；先保证装得下再提吞吐。
11. **bf16 与 fp16 怎么选？** → bf16 动态范围大、无需 loss scaling，是默认；fp16 需动态损失缩放。
12. **梯度检查点的代价？** → 约多 30% 计算换取激活显存大幅下降。
13. **MFU 是什么？典型值？** → 模型算力利用率，大规模训练常 35%~55%。
14. **MoE 训练要加哪种并行？** → 专家并行（EP），主要开销是两次 All-to-All，需与计算重叠。

---

## 参考

- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, arXiv:1910.02054
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, arXiv:1909.08053
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters*, arXiv:2104.04473
- Huang et al., *GPipe*, arXiv:1811.06965；Qi et al., *Zero Bubble Pipeline Parallelism*, arXiv:2401.10241
- Zhao et al., *PyTorch FSDP*, arXiv:2304.11277
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*（梯度检查点）, arXiv:1604.06174
- DeepSeek-AI, *DeepSeek-V3 Technical Report*（FP8 训练 + DualPipe）, arXiv:2412.19437
