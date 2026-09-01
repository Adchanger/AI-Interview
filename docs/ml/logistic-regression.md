# 逻辑回归（ML 八股 02）

> **更新时间**：2026-08-31

> **标签**：逻辑回归、交叉熵、最大似然、广义线性模型、面试八股

> **一句话**：逻辑回归是用 sigmoid 把线性打分压到 (0,1) 当概率、用交叉熵（等价于伯努利最大似然）训练的**线性分类器**，是所有"为什么不用 MSE""为什么要归一化""怎么处理多分类"类问题的原型题。

> **关联阅读**：[[/docs/ml/bias-variance-and-regularization.md]]、[[/docs/ml/svm.md]]、[[/docs/dl/loss-functions.md]]

---

## 1. 模型定义

$$z = w^{\top}x + b, \qquad \hat y = \sigma(z) = \frac{1}{1+e^{-z}}$$

把 $\hat y$ 解释为 $P(y=1\mid x)$，决策边界是 $z=0$（一个超平面）→ **LR 是线性模型**，它的"非线性"只体现在把打分映射成概率上。

### 1.1 对数几率（logit）解释

$$\log \frac{P(y=1\mid x)}{P(y=0\mid x)} = w^{\top}x + b$$

即"**对数几率是特征的线性函数**"，这也是名字 logistic / logit regression 的来源。由此可得**系数可解释性**：$w_j$ 增加 1 单位，几率（odds）乘以 $e^{w_j}$——金融风控、医疗领域偏爱 LR 的核心原因。

---

## 2. 损失函数：为什么是交叉熵而不是 MSE

伯努利似然：$P(y\mid x) = \hat y^{\,y}(1-\hat y)^{1-y}$，取负对数并对样本求平均：

$$L = -\frac1N\sum_i \big[y_i\log \hat y_i + (1-y_i)\log(1-\hat y_i)\big]$$

**梯度非常干净**：

$$\frac{\partial L}{\partial w} = \frac1N\sum_i (\hat y_i - y_i)\,x_i$$

> 面试高频：**为什么不用 MSE？** 三点，缺一不可：
> 1. **梯度会被 sigmoid 饱和杀掉**：MSE 的梯度含 $\sigma'(z)=\hat y(1-\hat y)$，预测严重错误（$z$ 很大但标签为 0）时 $\sigma'\to 0$，梯度消失、学不动；交叉熵的梯度是 $(\hat y - y)x$，错得越狠梯度越大。
> 2. **非凸 vs 凸**：MSE + sigmoid 对 $w$ 非凸，可能有多个局部极小；交叉熵 + sigmoid 对 $w$ 是凸的（Hessian 半正定），全局最优可达。
> 3. **概率建模不匹配**：分类标签是伯努利分布，MLE 天然给出交叉熵；MSE 隐含高斯噪声假设。

---

## 3. 求解与优化

- 无闭式解，用**梯度下降 / 拟牛顿（L-BFGS）/ IRLS（牛顿法）**；
- Hessian：$X^{\top}SX$，$S=\mathrm{diag}(\hat y_i(1-\hat y_i))$，正半定 → 凸优化；
- 大规模稀疏场景（CTR）常用 **FTRL**（在线学习 + L1 稀疏）。

### 3.1 线性可分时会发生什么

若数据线性可分，无正则的 LR 权重会**发散到无穷**（不断增大 $\|w\|$ 让似然趋近 1，log-loss 趋近 0）。所以工业实现（sklearn）**默认自带 L2**。这是一道很能区分水平的追问。

---

## 4. 常见追问

### 4.1 LR 需要特征归一化吗

- **理论上不需要**：LR 不受量纲影响，缩放特征会等比缩放对应权重，决策边界不变；
- **实践上需要**：
  1. 用梯度下降时，量纲差异大 → 损失面呈狭长椭圆 → 收敛慢、学习率难选；
  2. 加了 **L2 正则后就不再等价**：正则对所有权重同等惩罚，大量纲特征的权重被压得更狠 → 归一化影响结果；
  3. 特征离散化 / 分箱后天然同尺度。

### 4.2 LR 与线性回归的关系

| 维度 | 线性回归 | 逻辑回归 |
|------|----------|----------|
| 输出 | 实数 | (0,1) 概率 |
| 链接函数 | 恒等 | logit |
| 噪声/似然假设 | 高斯 | 伯努利 |
| 损失 | MSE | 交叉熵 |
| 解 | 有闭式解（正规方程） | 无闭式解，迭代求解 |

两者都是**广义线性模型（GLM）**的特例（指数族 + 链接函数）。

### 4.3 LR vs SVM

| 维度 | LR | 线性 SVM |
|------|----|----------|
| 损失 | log loss（所有样本都参与） | hinge loss（只有支持向量与违反间隔的样本参与） |
| 输出 | 概率（可直接排序/定阈值） | 距离，需 Platt scaling 才有概率 |
| 对离群点 | 较敏感（远点仍贡献梯度） | 较稳健（间隔外样本 0 损失） |
| 非线性 | 需手工特征交叉 | 核技巧直接上 |
| 大规模稀疏 | 更常用（易并行、易在线学习） | 核 SVM 不适合大样本 |

### 4.4 多分类怎么做

1. **Softmax 回归（多项 LR）**：$P(y=k\mid x)=\frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}$，类别互斥时首选；
2. **OvR（one-vs-rest）**：K 个二分类器，概率需归一化，实现简单；
3. **OvO**：$K(K-1)/2$ 个分类器，训练量大但每个更小；
4. **多标签**（非互斥）：K 个独立 sigmoid + BCE，不要用 softmax。

> softmax 有**平移不变性**（所有 logits 加常数结果不变），因此参数有冗余；二分类 softmax 与 sigmoid 等价。

### 4.5 类别不平衡怎么办

`class_weight='balanced'`（给少数类加权，等价于代价敏感学习）、下采样 + Bagging、上采样/SMOTE、只调阈值而不改模型；注意**采样会改变先验，输出概率需要校准**（做概率校正或用 log odds 修正）。详见 [[/docs/ml/feature-engineering-and-imbalance.md]]。

### 4.6 LR 为什么在工业界（CTR）长期主导

可解释、训练/上线极快、支持超高维稀疏特征、易做在线学习与 A/B、天然输出概率便于计费/排序。经典组合是 **特征交叉/分箱 + LR**，以及 Facebook 的 **GBDT + LR**（GBDT 做特征离散化，LR 做在线更新）。

---

## 5. 手撕代码：从零实现 LR

```python
import numpy as np

def sigmoid(z):
    # 数值稳定写法，避免 exp 溢出
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def train_lr(X, y, lr=0.1, epochs=1000, l2=1e-3):
    """X: (N, D) 已加偏置列; y: (N,) 取值 0/1"""
    N, D = X.shape
    w = np.zeros(D)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / N + l2 * w      # 交叉熵梯度 + L2
        w -= lr * grad
    return w

def predict_proba(X, w):
    return sigmoid(X @ w)
```

要点：sigmoid 必须写数值稳定版本；梯度就是 $X^\top(\hat p - y)/N$；不要对偏置列做 weight decay（严格实现时把 bias 排除）。

---

## 6. 面试高频问题速查

1. **LR 是线性还是非线性模型？** → 线性（决策边界是超平面），sigmoid 只是把打分映射为概率。
2. **为什么用交叉熵不用 MSE？** → 梯度不被 sigmoid 饱和杀死、损失对 $w$ 是凸的、符合伯努利 MLE。
3. **损失的梯度是什么？** → $\frac1N X^\top(\hat p - y)$，形式与线性回归一致，非常好记。
4. **LR 要不要归一化？** → 理论不需要，但用梯度下降或带 L2 时需要。
5. **线性可分时 LR 有什么问题？** → 权重发散，必须加正则。
6. **LR 输出的概率可信吗？** → 未采样、无强正则时较好；采样/加权后需概率校准（Platt / Isotonic）。
7. **LR 与 SVM 区别？** → log loss vs hinge loss；概率输出 vs 间隔；对离群点敏感度不同。
8. **LR 怎么做非线性？** → 特征交叉、多项式特征、分箱离散化、GBDT 编码后接 LR、核 LR。
9. **多分类怎么做？** → softmax 回归（互斥）/ OvR / 多标签用多个 sigmoid。
10. **LR 能处理缺失值吗？** → 不能，需先填补或分箱建"缺失"桶；树模型（XGBoost）可原生处理。

---

## 参考

- 周志华《机器学习》第 3 章
- Bishop, *Pattern Recognition and Machine Learning*, Ch. 4
- He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook*（GBDT + LR）
