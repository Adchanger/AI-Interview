# CNN / RNN / LSTM 与 Transformer 对比（DL 八股 05）

> **更新时间**：2026-08-31

> **标签**：CNN、RNN、LSTM、GRU、Transformer、面试八股

> **一句话**：CNN 靠局部连接 + 权值共享获得平移等变与高效率，RNN/LSTM 靠时间递归建模序列但无法并行且长依赖弱，Transformer 用自注意力一步连接任意两位置、可完全并行，代价是 $O(n^2)$ 复杂度与需要显式位置编码。

> **关联阅读**：[[/docs/llm/transformer-principle.md]]、[[/docs/dl/gradient-vanishing-exploding-residual.md]]、[[/docs/llm/long-context-and-flashattention.md]]

---

## 1. CNN 核心考点

### 1.1 三大归纳偏置

1. **局部连接**（局部感受野）：像素相关性随距离衰减；
2. **权值共享**：同一卷积核扫全图 → 参数量与输入尺寸解耦、具备**平移等变性**（translation equivariance；配合 pooling 才近似平移不变性）；
3. **层次化**：浅层边缘纹理 → 深层语义部件。

### 1.2 必背公式

**输出尺寸**：

$$H_{out} = \left\lfloor\frac{H_{in} + 2p - k_{\text{eff}}}{s}\right\rfloor + 1,\qquad k_{\text{eff}} = d(k-1)+1$$

（$p$ padding、$s$ stride、$d$ dilation 空洞率）

**参数量**（含 bias）：$k\times k\times C_{in}\times C_{out} + C_{out}$

**FLOPs**（乘加计 1）：$H_{out}\times W_{out}\times k^2\times C_{in}\times C_{out}$

**感受野递推**：$RF_{l} = RF_{l-1} + (k_l-1)\prod_{i<l}s_i$

> 面试高频：**两层 3×3 与一层 5×5 感受野相同，为什么用两层 3×3？** → 参数量 $2\times9C^2=18C^2 < 25C^2$、FLOPs 更少、中间多一次非线性表达力更强（VGG 的核心论点）。

### 1.3 卷积变体

| 变体 | 作用 |
|------|------|
| 1×1 卷积 | 通道升降维、跨通道信息融合、极省参数（bottleneck 结构） |
| **深度可分离卷积** | Depthwise + Pointwise，计算量约降到 $\frac{1}{C_{out}}+\frac{1}{k^2}$，MobileNet 核心 |
| 空洞卷积 | 不增参数扩大感受野，语义分割常用 |
| 转置卷积 | 上采样（棋盘效应需注意，可用插值+conv 替代） |
| 分组卷积 | ResNeXt / ShuffleNet，降算力 |
| 可变形卷积 | 采样点可学偏移，适应形变 |

### 1.4 池化与经典网络演进

- **Max pooling**：保留最强响应、平移鲁棒；**Average pooling**：保留整体统计；**Global Average Pooling** 替代全连接大幅减参（NIN/ResNet）；
- 演进主线：LeNet → AlexNet（ReLU+Dropout+GPU）→ VGG（3×3 堆叠）→ Inception（多尺度+1×1）→ **ResNet（残差）** → DenseNet → SENet（通道注意力）→ MobileNet/EfficientNet（轻量与缩放法则）→ ConvNeXt（借鉴 Transformer 设计的现代 CNN）。

---

## 2. RNN / LSTM / GRU

### 2.1 朴素 RNN 及其问题

$$h_t = \tanh(W_hh_{t-1}+W_xx_t+b)$$

- **无法并行**：$h_t$ 依赖 $h_{t-1}$，训练时间随序列长度线性增长；
- **长依赖失效**：梯度沿时间连乘 → 消失/爆炸（见 [[/docs/dl/gradient-vanishing-exploding-residual.md]]）；
- 信息瓶颈：所有历史压进一个固定维度的隐状态。

### 2.2 LSTM 三门

$$f_t=\sigma(W_f[h_{t-1},x_t]),\quad i_t=\sigma(W_i[\cdot]),\quad o_t=\sigma(W_o[\cdot]),\quad \tilde c_t=\tanh(W_c[\cdot])$$

$$c_t = f_t\odot c_{t-1} + i_t\odot \tilde c_t,\qquad h_t = o_t\odot\tanh(c_t)$$

| 门 | 作用 |
|----|------|
| 遗忘门 $f$ | 决定丢弃多少旧记忆（**最关键**，bias 常初始化为 1） |
| 输入门 $i$ | 决定写入多少新信息 |
| 输出门 $o$ | 决定暴露多少记忆给下游 |

**为什么能缓解梯度消失**：$c_t$ 是**加性**更新，$\partial c_t/\partial c_{t-1}=f_t$ 可接近 1，形成常数误差流。

### 2.3 GRU

合并为**更新门 $z$ + 重置门 $r$**，去掉单独的 cell state：

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

参数比 LSTM 少约 1/4（3 组权重 vs 4 组），小数据/短序列上常与 LSTM 相当且更快；大数据长序列 LSTM 略稳。

### 2.4 其他序列结构

- **BiLSTM**：双向拼接，仅适用于非自回归任务（分类、NER）；
- **Seq2Seq + Attention**（Bahdanau/Luong）：注意力最初就是为解决 encoder 定长向量瓶颈提出的，是 Transformer 的前身；
- **TCN**：因果空洞卷积做序列建模，可并行、感受野可控。

---

## 3. 三者系统对比

| 维度 | CNN | RNN / LSTM | Transformer |
|------|-----|------------|-------------|
| 任意两位置的路径长度 | $O(\log_k n)$（层堆叠） | $O(n)$ | **$O(1)$** |
| 每层计算复杂度 | $O(k\cdot n\cdot d^2)$ | $O(n\cdot d^2)$ | $O(n^2 d)$（注意力）+ $O(nd^2)$（FFN） |
| 训练并行度 | 高 | **低（时间步串行）** | 高 |
| 归纳偏置 | 局部性、平移等变 | 时序递归、马尔可夫式 | **弱**（几乎无，靠数据与规模） |
| 位置信息 | 隐含在卷积结构 | 隐含在递归顺序 | **必须显式注入**（位置编码） |
| 长依赖 | 需堆很深或空洞卷积 | 弱（LSTM 缓解） | 强 |
| 显存/推理 | 小 | 状态固定，流式友好 | KV Cache 随长度线性增长 |
| 数据需求 | 中（偏置强，小数据也行） | 中 | **大**（偏置弱，需大数据/预训练） |

> 面试高频：**Transformer 相比 LSTM 的优势？** → ① 全并行训练（利用 GPU）；② 任意两 token 一步可达，长依赖强；③ 可堆叠极深并良好 scale（Scaling Law 有效）；④ 更适合迁移/预训练。代价：$O(n^2)$ 复杂度、需要位置编码、小数据易过拟合。

> 面试高频：**为什么 CNN 在 ViT 之后仍没被淘汰？** → 小数据/小算力下 CNN 的归纳偏置带来更好样本效率；边端部署上卷积算子更成熟高效；ConvNeXt 等现代 CNN 与 ViT 在 ImageNet 上可比。混合结构（卷积 stem + Transformer）也很常见。

---

## 4. 手撕代码

```python
import torch, torch.nn as nn

def conv_out_size(h, k, s=1, p=0, d=1):
    """卷积/池化输出尺寸，面试口算题必背"""
    k_eff = d * (k - 1) + 1
    return (h + 2 * p - k_eff) // s + 1

class LSTMCellManual(nn.Module):
    """手写 LSTM 单元：一次矩阵乘算四个门"""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.w = nn.Linear(d_in + d_h, 4 * d_h)
        nn.init.zeros_(self.w.bias)
        nn.init.ones_(self.w.bias[d_h:2 * d_h])      # forget gate bias = 1

    def forward(self, x, state):
        h, c = state
        gates = self.w(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                            # 加性更新 → 常数误差流
        h = o * torch.tanh(c)
        return h, (h, c)
```

---

## 5. 面试高频问题速查

1. **卷积输出尺寸怎么算？** → $\lfloor (H+2p-k_{eff})/s\rfloor+1$，$k_{eff}=d(k-1)+1$。
2. **卷积参数量与 FLOPs？** → $k^2C_{in}C_{out}(+C_{out})$；FLOPs 再乘 $H_{out}W_{out}$。
3. **1×1 卷积有什么用？** → 通道变换/降维、跨通道融合、构造 bottleneck、几乎不增算力。
4. **为什么用两层 3×3 代替 5×5？** → 参数与算力更少、非线性更多。
5. **深度可分离卷积省多少？** → 约 $\frac{1}{C_{out}}+\frac{1}{k^2}$，3×3 时约 1/8~1/9。
6. **感受野怎么算？** → $RF_l=RF_{l-1}+(k_l-1)\prod_{i<l}s_i$。
7. **CNN 的平移不变性从哪来？** → 权值共享给等变性，pooling/下采样给近似不变性；严格不变性并不成立。
8. **LSTM 三门各自作用？** → 遗忘门丢旧、输入门写新、输出门控暴露；核心是加性 cell state。
9. **GRU 与 LSTM 区别？** → GRU 两门、无独立 cell、参数少更快；效果多数任务相当。
10. **RNN 为什么不能并行？** → 隐状态时间依赖；推理时 Transformer 也是逐 token，但训练可并行。
11. **Transformer 的复杂度与瓶颈？** → 注意力 $O(n^2d)$，长序列瓶颈；解法见 FlashAttention/稀疏/线性注意力。
12. **Transformer 为什么需要位置编码而 CNN/RNN 不需要？** → 自注意力是置换不变的，结构本身不含顺序信息。

---

## 参考

- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762（表 1 的复杂度对比）
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder (GRU)*, arXiv:1406.1078
- Simonyan & Zisserman, *VGG*, arXiv:1409.1556
- Howard et al., *MobileNets*, arXiv:1704.04861
- Liu et al., *A ConvNet for the 2020s (ConvNeXt)*, arXiv:2201.03545
