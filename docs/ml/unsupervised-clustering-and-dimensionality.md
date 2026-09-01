# 聚类与降维：KMeans / GMM-EM / PCA / t-SNE（ML 八股 07）

> **更新时间**：2026-08-31

> **标签**：KMeans、EM算法、GMM、PCA、降维、面试八股

> **一句话**：KMeans 是硬划分的"距离最近就归你"，GMM 用 EM 做软划分并能建模椭圆簇；PCA 沿方差最大方向做正交投影（本质是协方差矩阵特征分解 / 数据矩阵 SVD），t-SNE/UMAP 只用于可视化、不用于建模特征。

> **关联阅读**：[[/docs/ml/feature-engineering-and-imbalance.md]]、[[/docs/ml/model-evaluation-metrics.md]]

---

## 1. KMeans

### 1.1 目标与算法

最小化簇内平方误差（SSE / inertia）：

$$J = \sum_{k=1}^{K}\sum_{x\in C_k}\|x-\mu_k\|^2$$

迭代两步（本质是**坐标下降**，也是 EM 的硬指派特例）：
1. **指派**：每个样本归到最近质心；
2. **更新**：质心 = 簇内样本均值。

单调不增 → 一定收敛，但**只收敛到局部最优**。

### 1.2 关键细节（高频追问）

- **初始化**：随机初始化差异极大 → 用 **KMeans++**（按距离平方为概率依次选远的初始点），或多次重启取最小 SSE（sklearn `n_init`）；
- **K 怎么选**：肘部法（SSE 拐点）、**轮廓系数**（$s=\frac{b-a}{\max(a,b)}$，越接近 1 越好）、Gap Statistic、业务约束；
- **必须归一化**：基于欧氏距离，量纲主导结果；
- **只适合凸形、大小相近、密度相近的簇**：环形/月牙形/长条形会失败 → 换 DBSCAN、谱聚类、GMM；
- **对离群点敏感**：均值被拉偏 → 用 **K-medoids / K-medians**；
- **复杂度** $O(N\cdot K\cdot d\cdot T)$，大规模用 **MiniBatchKMeans**；
- **距离度量**：KMeans 与"均值是最优中心"绑定在平方欧氏距离上，换成余弦距离要用球面 KMeans（先 L2 归一化）。

### 1.3 其他聚类方法速览

| 方法 | 核心思想 | 优势 | 局限 |
|------|----------|------|------|
| **DBSCAN** | 密度可达（`eps`, `min_samples`） | 任意形状、自动定簇数、识别噪声点 | 高维失效、密度不均时难调参 |
| **层次聚类** | 自底向上合并（Ward/average/complete） | 出树状图、无需预设 K | $O(N^2)$ 以上，不适合大数据 |
| **谱聚类** | 相似图的拉普拉斯特征向量 + KMeans | 处理非凸结构 | 需构相似图，$N$ 大时贵 |
| **GMM** | 概率生成模型 + EM | 软划分、椭圆簇、给概率 | 需定 K、可能奇异解 |
| **HDBSCAN** | DBSCAN 的层次化版本 | 密度不均更稳、少调参 | 实现较重 |

---

## 2. EM 算法与 GMM

### 2.1 GMM 模型

$$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid \mu_k,\Sigma_k),\qquad \sum_k \pi_k = 1$$

隐变量 $z$ 表示样本来自哪个高斯，直接对含隐变量的对数似然求导困难（log 里有求和）→ 用 EM。

### 2.2 EM 两步

- **E 步**：用当前参数算隐变量后验（责任度）
  $$\gamma_{ik}=\frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$$
- **M 步**：用责任度加权更新参数
  $$\mu_k=\frac{\sum_i\gamma_{ik}x_i}{\sum_i\gamma_{ik}},\quad \Sigma_k=\frac{\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{\sum_i\gamma_{ik}},\quad \pi_k=\frac{\sum_i\gamma_{ik}}{N}$$

> 面试高频：**EM 为什么收敛？** → E 步构造对数似然的**下界**（Jensen 不等式，ELBO），M 步最大化该下界；由于每步下界不减且下界在当前参数处与似然相切，所以似然单调不减，收敛到局部最优/鞍点。**不保证全局最优**。

### 2.3 KMeans 与 GMM 的关系

| 维度 | KMeans | GMM |
|------|--------|-----|
| 指派 | 硬（0/1） | 软（概率 $\gamma_{ik}$） |
| 簇形状 | 球形（等方差） | 椭圆（可学协方差） |
| 目标 | SSE | 似然 |
| 关系 | 当 $\Sigma_k=\sigma^2 I$ 且 $\sigma\to 0$ 时，GMM 的 EM 退化为 KMeans | |

工程注意：GMM 协方差可能奇异（某簇只有一两个点）→ 加 `reg_covar` 正则；协方差类型可选 `full/diag/tied/spherical` 控制参数量。

---

## 3. PCA

### 3.1 两种等价推导

1. **最大方差**：找单位方向 $w$ 使投影方差 $w^\top \Sigma w$ 最大 → 拉格朗日 → $\Sigma w = \lambda w$，即协方差矩阵的**特征向量**，特征值就是该方向方差；
2. **最小重构误差**：找 $k$ 维子空间使样本到子空间的平方距离和最小 → 同一组特征向量。

**流程**：中心化（必须！）→ 算协方差 $\Sigma=\frac1N X^\top X$ → 特征分解 → 取前 $k$ 个特征向量组成 $W$ → $Z=XW$。

### 3.2 与 SVD 的关系

对中心化后的 $X\in\mathbb{R}^{N\times d}$ 做 $X=U\Sigma_{s}V^\top$：
- $V$ 的列 = 主成分方向（$X^\top X$ 的特征向量）；
- 奇异值与特征值关系 $\lambda_i = \sigma_i^2/N$；
- **实际实现都用 SVD**（数值更稳、不用显式构造 $d\times d$ 协方差矩阵，$d$ 很大时尤其重要）。

### 3.3 常见追问

- **为什么必须中心化？** → 不中心化则"方差最大方向"会被均值偏移主导，第一主成分退化为指向数据均值的方向；
- **要不要标准化？** → 特征量纲不同时必须（否则大量纲特征霸占主成分）；同量纲（如像素）可只中心化；
- **怎么定 $k$？** → 累计解释方差比（85%/95%）、碎石图拐点、下游任务交叉验证；
- **PCA 会提升模型效果吗？** → 主要用于去相关/降噪/加速/可视化，**不保证提升精度**：方差大 ≠ 判别性强（可能丢掉小方差但强判别的方向）→ 有监督降维用 **LDA**；
- **PCA vs LDA**：无监督最大方差 vs 有监督最大化类间/类内散度比，LDA 降维上限 $K-1$；
- **PCA 是线性方法**：非线性结构用 Kernel PCA、AutoEncoder、UMAP。

### 3.4 白化（Whitening）

除以 $\sqrt{\lambda_i}$ 让各方向方差为 1，去相关 + 同尺度；代价是放大了小特征值方向的噪声（需加 $\epsilon$）。

---

## 4. t-SNE 与 UMAP（可视化专用）

| 维度 | t-SNE | UMAP |
|------|-------|------|
| 思想 | 高维用高斯、低维用 t 分布刻画邻域概率，最小化 KL | 基于模糊单纯集/图的拓扑近似，优化交叉熵 |
| 保留结构 | 局部结构好，**全局距离不可信** | 局部 + 一定的全局结构 |
| 速度 | 慢（Barnes-Hut $O(N\log N)$） | 更快，可 transform 新样本 |
| 关键超参 | perplexity（5–50） | `n_neighbors`、`min_dist` |

**必背结论**：t-SNE 的**簇间距离、簇大小都没有定量意义**，随机种子和 perplexity 会改变图形；它**不是特征降维工具**（无法自然映射新样本、非凸优化不稳定），只做可视化探索。

---

## 5. 手撕代码：KMeans（KMeans++ 初始化）

```python
import numpy as np

def kmeans(X, k, iters=100, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    n = len(X)
    # ---- KMeans++ 初始化 ----
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)

    prev = np.inf
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)   # (n, k)
        labels = d2.argmin(1)
        sse = d2[np.arange(n), labels].sum()
        for j in range(k):                              # 更新质心
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
            else:                                       # 空簇：重定位到最远点
                C[j] = X[d2.min(1).argmax()]
        if abs(prev - sse) < tol:
            break
        prev = sse
    return labels, C, sse
```

要点：KMeans++ 初始化、空簇处理、以 SSE 变化量为收敛判据。

---

## 6. 面试高频问题速查

1. **KMeans 的目标函数与收敛性？** → 最小化 SSE，交替优化单调不增，收敛到局部最优。
2. **KMeans 的缺点？** → 需预设 K、初始化敏感、只适合凸球形簇、对离群点敏感、必须归一化。
3. **K 怎么选？** → 肘部法、轮廓系数、Gap Statistic、业务需求。
4. **KMeans++ 做了什么？** → 按距离平方概率选初始中心，让初始点分散，显著降低陷入坏局部解的概率。
5. **KMeans 与 GMM 的关系？** → GMM 在球形等方差且方差趋 0 时退化为 KMeans；GMM 是软划分。
6. **EM 的 E 步 M 步在做什么？为什么收敛？** → E 求隐变量后验构造 ELBO 下界，M 最大化下界；似然单调不减。
7. **DBSCAN 与 KMeans 怎么选？** → 任意形状、有噪声、簇数未知用 DBSCAN；大规模、近球形簇用 KMeans。
8. **PCA 的两种推导？** → 最大投影方差 / 最小重构误差，殊途同归到协方差矩阵特征分解。
9. **PCA 为什么要中心化？** → 否则第一主成分被均值方向主导，方差定义失真。
10. **PCA 与 SVD 的关系？** → 中心化数据的右奇异向量即主成分方向，$\lambda=\sigma^2/N$；实现用 SVD 更稳。
11. **PCA 一定能提升下游效果吗？** → 不一定，方差大不等于判别性强；有监督场景可考虑 LDA。
12. **t-SNE 图能解读簇间距离吗？** → 不能，只反映局部邻域关系；且不适合作为建模特征。

---

## 参考

- 周志华《机器学习》第 9、10 章
- Bishop, *PRML*, Ch. 9（EM/GMM）与 Ch. 12（PCA）
- Arthur & Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, 2007
- van der Maaten & Hinton, *Visualizing Data using t-SNE*, JMLR 2008
- McInnes et al., *UMAP*, arXiv:1802.03426
