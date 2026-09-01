# 梯度消失/爆炸、残差连接与初始化（DL 八股 04）

> **更新时间**：2026-08-31

> **标签**：梯度消失、残差连接、初始化、梯度裁剪、面试八股

> **一句话**：深层网络的梯度是各层雅可比的连乘，乘积小于 1 就消失、大于 1 就爆炸；解决办法是**改激活（ReLU 系）+ 归一化（BN/LN）+ 残差连接（提供恒等梯度通路）+ 合适初始化（Xavier/He）+ 梯度裁剪**。

> **关联阅读**：[[/docs/dl/normalization-bn-ln-rmsnorm.md]]、[[/docs/dl/activation-functions.md]]、[[/docs/dl/cnn-rnn-lstm-vs-transformer.md]]

---

## 1. 问题的数学根源

对 $L$ 层网络，损失对第 $l$ 层的梯度：

$$\frac{\partial \mathcal{L}}{\partial h_l} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{k=l+1}^{L}\frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{k=l+1}^{L} \big(W_k^\top \mathrm{diag}(\phi'(z_k))\big)$$

**连乘结构**决定一切：
- 每项谱范数 < 1 → 指数衰减 → **梯度消失**（底层学不动，等价于只训练了顶部几层）；
- 每项谱范数 > 1 → 指数放大 → **梯度爆炸**（loss 变 NaN、参数发散）。

### 1.1 典型诱因

| 现象 | 诱因 |
|------|------|
| 梯度消失 | Sigmoid/Tanh 饱和（导数 ≤0.25）、权重初始化过小、网络过深无残差、RNN 长序列 |
| 梯度爆炸 | 权重初始化过大、RNN 循环矩阵谱半径 >1、学习率过大、数据中的异常样本 |

### 1.2 怎么诊断

- 打印每层 `grad_norm`（`p.grad.norm()`）：底层比顶层小几个数量级 → 消失；出现 inf/NaN → 爆炸；
- 观察参数更新量与权重量级之比；
- loss 长期不动（消失）vs loss 突然飙升/NaN（爆炸）。

---

## 2. 解决方案全景

| 方案 | 机制 |
|------|------|
| **ReLU 系激活** | 正区间导数恒为 1，切断"每层乘 <1"的链条 |
| **归一化（BN/LN/RMSNorm）** | 控制每层输入尺度，间接控制雅可比范数，平滑损失曲面 |
| **残差连接** | 提供导数为 1 的恒等通路（见下节） |
| **合适初始化** | 让前向/反向的方差在层间守恒 |
| **梯度裁剪** | 直接给梯度范数上界，专治爆炸 |
| **LSTM/GRU 门控** | 用加性状态更新 + 门控让 cell state 的梯度接近 1（缓解 RNN 梯度消失） |
| **架构层面** | DenseNet 密集连接、Highway Network、DeepNorm、Pre-LN、更短的有效深度 |

---

## 3. 残差连接（ResNet 的核心）

$$h_{l+1}=h_l + F(h_l) \;\Longrightarrow\; \frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}$$

**关键就是这个 $I$**：即使 $\partial F/\partial h$ 很小，梯度也能以 ≈1 的系数直通浅层；展开后梯度是"多条路径之和"而不是"单一路径之积"，其中包含一条完全恒等的路径。

> 面试高频：**残差解决的是梯度消失还是退化问题？** → 论文的动机是**退化问题（degradation）**：56 层比 20 层的**训练误差**更高，说明不是过拟合而是难优化。残差让"学习恒等映射"变得平凡（$F\to0$ 即可），使深层至少不比浅层差；顺带极大改善了梯度流。两点都要说。

**补充要点**：
- **Identity mapping 很重要**：He 等在 *Identity Mappings in Deep Residual Networks* 中给出 Pre-activation ResNet（BN-ReLU-Conv 顺序），让恒等路径完全干净，1001 层可训；
- **维度不匹配**时用 1×1 conv 投影（projection shortcut）；
- Transformer 的每个子层都是 `x + Sublayer(LN(x))`（Pre-LN），同一思想；
- 残差不是免费的：ResNet 可视作"浅网络的集成"（Veit et al.），有效路径长度远小于名义深度。

---

## 4. 参数初始化

目标：让每层激活的方差在前向传播中保持稳定、梯度方差在反向中保持稳定。

| 初始化 | 方差 | 适用 |
|--------|------|------|
| **Xavier / Glorot** | $\mathrm{Var}(W)=\frac{2}{n_{in}+n_{out}}$ | Tanh / Sigmoid（对称、线性区增益 1） |
| **He / Kaiming** | $\mathrm{Var}(W)=\frac{2}{n_{in}}$ | **ReLU 系**（ReLU 砍掉一半方差，故乘 2） |
| 正交初始化 | $W$ 正交，谱范数 1 | RNN 循环矩阵、深层线性堆叠 |
| 全 0 | ✗ | 对称性无法破除，所有神经元学到相同东西 |
| 过大随机 | ✗ | 直接饱和/爆炸 |
| **LLM 常用** | $\mathcal{N}(0, 0.02^2)$，并对残差分支输出层再除 $\sqrt{2N_{layer}}$ | GPT-2 起的做法，抑制残差累积导致的激活范数增长 |

> 面试高频：**为什么 He 初始化的方差是 Xavier 的两倍？** → ReLU 让约一半输入置零，输出方差约减半；补偿因子 2 才能维持方差守恒。

**偏置**一般初始化为 0；LSTM 的 forget gate bias 常初始化为 1（让初始状态倾向"记住"）。

---

## 5. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- **按范数裁剪**（推荐）：$g\leftarrow g\cdot\min(1, \frac{\text{max\_norm}}{\|g\|})$，保持方向不变；
- **按值裁剪**：逐元素 clamp，会改变梯度方向，慎用；
- 混合精度训练时要**先 unscale 再裁剪**（`scaler.unscale_(optimizer)`），否则裁的是被放大后的梯度；
- LLM 预训练默认 `max_norm=1.0`，是抑制 loss spike 的第一道防线。

---

## 6. RNN 的特例：为什么 LSTM 能缓解梯度消失

朴素 RNN：$h_t=\phi(W_hh_{t-1}+W_xx_t)$，梯度沿时间反传含 $\prod W_h^\top\mathrm{diag}(\phi')$ → 谱半径 <1 消失、>1 爆炸，长依赖学不到。

LSTM 的 cell state 更新是**加性**的：$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$，则 $\frac{\partial c_t}{\partial c_{t-1}}=f_t$（对角、可接近 1）→ 形成"**常数误差流（CEC）**"通道，梯度可跨很多时间步传播。注意这只是**缓解**而非消除：$f_t<1$ 累乘仍会衰减，且梯度爆炸仍需裁剪。

---

## 7. 面试高频问题速查

1. **梯度消失/爆炸的根本原因？** → 反向传播是各层雅可比的连乘，谱范数偏离 1 就指数衰减/放大。
2. **怎么判断遇到了哪一个？** → 看逐层 grad_norm 与 loss 曲线：底层梯度极小/loss 不动 = 消失；NaN/骤升 = 爆炸。
3. **残差为什么能训很深？** → 雅可比中出现 $I$，梯度有恒等通路；同时把"学恒等映射"变简单，解决退化问题。
4. **ResNet 解决的是过拟合吗？** → 不是，解决的是深层网络**训练误差反而上升**的退化/优化问题。
5. **Xavier 与 He 的区别？** → 分母 $\frac{n_{in}+n_{out}}2$ vs $n_{in}$，He 多乘 2 补偿 ReLU 砍掉的一半方差。
6. **权重能全初始化为 0 吗？** → 不能，对称性无法破除；bias 可以为 0。
7. **BN 如何帮助梯度传播？** → 约束每层输入分布与尺度，使雅可比范数可控、损失曲面更平滑，允许更大 lr。
8. **LSTM 为什么缓解梯度消失？** → cell state 加性更新使 $\partial c_t/\partial c_{t-1}=f_t$，构成常数误差流。
9. **梯度裁剪按范数还是按值？** → 按范数，保持方向；混合精度需先 unscale。
10. **Transformer 里对应的手段有哪些？** → Pre-LN + 残差 + RMSNorm + 小初始化（0.02 且按层数缩放）+ 梯度裁剪 + warmup。
11. **深层网络还有哪些优化难点？** → 不是局部极小值（高维中多为鞍点），而是鞍点/平坦区、尖锐极小点与 loss spike。

---

## 参考

- He et al., *Deep Residual Learning for Image Recognition*, arXiv:1512.03385
- He et al., *Identity Mappings in Deep Residual Networks*, arXiv:1603.05027
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010
- He et al., *Delving Deep into Rectifiers (He init)*, arXiv:1502.01852
- Pascanu et al., *On the difficulty of training Recurrent Neural Networks*, arXiv:1211.5063
- Veit et al., *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*, arXiv:1605.06431
