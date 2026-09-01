# 决策树与集成学习：RF / GBDT / XGBoost / LightGBM（ML 八股 04）

> **更新时间**：2026-08-31

> **标签**：决策树、随机森林、GBDT、XGBoost、LightGBM、面试八股

> **一句话**：决策树用特征阈值递归切分空间；Bagging（随机森林）并行建树**降方差**，Boosting（GBDT/XGBoost/LightGBM）串行拟合残差**降偏差**，XGBoost 的关键是二阶泰勒 + 显式正则，LightGBM 的关键是直方图 + 单边采样 + 互斥特征捆绑 + leaf-wise 生长。

> **关联阅读**：[[/docs/ml/bias-variance-and-regularization.md]]、[[/docs/ml/model-evaluation-metrics.md]]

---

## 1. 决策树基础

### 1.1 三种经典算法

| 算法 | 分裂准则 | 特征类型 | 树形 | 备注 |
|------|----------|----------|------|------|
| ID3 | 信息增益 | 离散 | 多叉 | 偏好取值多的特征 |
| C4.5 | 信息增益**率** | 离散+连续 | 多叉 | 增益率修正上述偏好；可处理缺失 |
| **CART** | 分类用**基尼指数**，回归用**平方误差** | 离散+连续 | **二叉** | sklearn / GBDT 系的基学习器 |

熵与基尼：

$$H(D) = -\sum_k p_k\log p_k, \qquad \mathrm{Gini}(D) = 1-\sum_k p_k^2$$

> 面试高频：**基尼和熵怎么选？** → 二者曲线形状接近（基尼是熵的一阶近似），效果差别很小；基尼**不用算 log**，计算更快，所以 CART 用它。

### 1.2 剪枝

- **预剪枝**：`max_depth`、`min_samples_split/leaf`、`min_impurity_decrease` —— 快，但可能欠拟合（贪心的短视）；
- **后剪枝**：先长满再剪（CART 的**代价复杂度剪枝** CCP：$R_\alpha(T)=R(T)+\alpha|T|$）—— 效果更好，代价是训练更慢。

### 1.3 决策树的性质（常考）

- **不需要归一化**：只比较阈值大小，单调变换不改变分裂；
- **对异常值较稳健**：异常值只影响自己所在的划分路径；
- **能处理混合类型特征**、可解释性强（if-then 规则）；
- **天生高方差**：数据小扰动可能让整棵树结构大变 → 需要集成；
- **难以表达线性/斜边界**：$x_1+x_2>0$ 这类边界需要很多阶梯去逼近。

---

## 2. 集成学习三大范式

| 范式 | 代表 | 基学习器 | 训练方式 | 主要降低 |
|------|------|----------|----------|----------|
| **Bagging** | 随机森林 | 强学习器（深树） | 并行、bootstrap 采样 | **方差** |
| **Boosting** | AdaBoost、GBDT、XGB、LGBM | 弱学习器（浅树） | 串行、拟合前序残差 | **偏差** |
| **Stacking** | 多模型 + 元学习器 | 异质模型 | 分层、用 out-of-fold 预测做特征 | 两者兼顾（易泄漏，要严格 OOF） |

### 2.1 随机森林（Random Forest）

两重随机：**样本 bootstrap** + **每次分裂随机选特征子集**（分类常用 $\sqrt{d}$，回归 $d/3$）。目的是**降低树之间的相关性**——因为 $\mathrm{Var}(\bar X)=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$，只有降低 $\rho$ 才能让平均真正减小方差。

- **OOB 估计**：每棵树约 36.8%（$1/e$）样本未被抽到，可当免费验证集；
- 天然并行、几乎不用调参、能给特征重要性（不纯度下降 / permutation importance）；
- 不容易过拟合（增加树的数量不会过拟合，只会饱和）。

> 面试高频：**RF 和 GBDT 的区别？** → ① 目标：降方差 vs 降偏差；② 基学习器：深树 vs 浅树（弱）；③ 训练：并行 vs 串行；④ 对异常值：RF 稳健，GBDT 更敏感（不断拟合残差会追噪声）；⑤ 调参：RF 少，GBDT 多（lr、树数、深度、采样）。

### 2.2 GBDT（梯度提升树）

模型是加法模型 $F_m(x)=F_{m-1}(x)+\nu\, h_m(x)$，其中 $h_m$ 用 CART 回归树拟合当前损失的**负梯度**（伪残差）：

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

平方损失时负梯度就是残差 $y-\hat y$，所以常说"GBDT 拟合残差"，严格说是**拟合负梯度**（这样才能推广到任意可导损失，如 logloss、Huber）。

- **学习率 $\nu$（shrinkage）**：每棵树只走一小步，$\nu$ 小 + 树多 → 泛化更好；
- **GBDT 分类**：用回归树拟合 logloss 的负梯度，最后 sigmoid/softmax 输出；基学习器**永远是回归树**；
- **Subsample < 1** 即 Stochastic Gradient Boosting，加随机性防过拟合。

---

## 3. XGBoost：GBDT 的工程与理论升级

### 3.1 二阶泰勒展开的目标函数

$$\mathcal{L}^{(t)} \simeq \sum_i \Big[g_i f_t(x_i) + \tfrac12 h_i f_t^2(x_i)\Big] + \Omega(f_t),\qquad \Omega(f)=\gamma T + \tfrac12\lambda\|w\|^2$$

其中 $g_i,h_i$ 是一阶、二阶导数，$T$ 是叶子数。对叶子 $j$（样本集合 $I_j$）求最优权重：

$$w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}, \qquad \mathcal{L}^* = -\frac12\sum_j \frac{(\sum_{i\in I_j}g_i)^2}{\sum_{i\in I_j}h_i+\lambda} + \gamma T$$

分裂增益：

$$\mathrm{Gain} = \tfrac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

$\gamma$ 相当于"分裂的门票"，增益小于 $\gamma$ 就不分裂（**预剪枝**）。

### 3.2 与 GBDT 的核心差异（必背）

| 维度 | GBDT | XGBoost |
|------|------|---------|
| 损失近似 | 一阶（负梯度） | **二阶泰勒**（收敛更快、更准） |
| 正则 | 靠树结构/学习率隐式 | 目标函数**显式**含 $\gamma T + \frac12\lambda\|w\|^2$ |
| 缺失值 | 需预处理 | **稀疏感知**：学习默认分裂方向 |
| 列采样 | 无 | 有（借鉴 RF，进一步防过拟合） |
| 分裂查找 | 精确贪心 | 精确 + **近似分位点（weighted quantile sketch）** |
| 工程 | 单线程为主 | 特征块并行、Cache-aware、out-of-core、支持分布式 |
| 收缩/采样 | shrinkage | shrinkage + 行列采样 |

> 注意：XGBoost 的"并行"是**特征维度上的并行**（预排序块、并行找分裂点），Boosting 的**树之间仍然串行**——这是高频陷阱题。

---

## 4. LightGBM：为什么更快更省内存

| 技术 | 做了什么 | 收益 |
|------|----------|------|
| **Histogram 直方图算法** | 连续特征分桶（默认 255 bin），用 bin 索引代替预排序值 | 内存降一个量级、分裂查找从 $O(N\cdot d)$ 降到 $O(\#bin\cdot d)$ |
| **直方图差加速** | 父直方图 − 兄弟直方图 = 另一子节点直方图 | 少算一半 |
| **Leaf-wise（best-first）生长** | 每次分裂全局增益最大的叶子，而非按层展开 | 同样叶子数下损失更低；但更易过拟合，需 `num_leaves` / `max_depth` / `min_data_in_leaf` 约束 |
| **GOSS**（单边梯度采样） | 保留大梯度样本，小梯度样本随机采样并放大权重 | 少样本训练、精度损失小 |
| **EFB**（互斥特征捆绑） | 把很少同时非零的稀疏特征打包成一个特征 | 有效特征数下降，适合 one-hot 后的高维稀疏 |
| **原生类别特征** | 按类别的梯度统计排序后寻找最优二分 | 不用 one-hot，避免高基数类别爆炸 |

> 面试高频：**LightGBM 为什么比 XGBoost 快？** → 直方图代替预排序（内存与计算都省）+ 直方图差 + GOSS 减样本 + EFB 减特征 + leaf-wise 更快降损失；代价是 leaf-wise 更易过拟合、小数据集上不如 XGB 稳。

### 4.1 CatBoost 一句话

用 **ordered target statistics**（有序目标编码）和 **ordered boosting** 解决类别特征编码带来的**目标泄漏 / 预测偏移**，类别特征多时表现好，默认对称树（oblivious tree）推理快。

---

## 5. 实战与追问

- **调参优先级**（XGB/LGBM）：`learning_rate` + `n_estimators`（配 early stopping）→ `max_depth`/`num_leaves` → `min_child_weight`/`min_data_in_leaf` → `subsample`/`colsample` → `lambda`/`alpha`/`gamma`；
- **树模型不需要归一化**，但**需要**处理高基数类别（目标编码需防泄漏，用 K 折 OOF）；
- **树模型 vs 神经网络**：中小规模**表格数据**上 GBDT 依然常胜（对异质特征、缺失、非光滑关系友好）；图像/文本/语音等高维同质信号用深度模型；
- **特征重要性坑**：`gain` 优于 `split count`；两者都对高基数特征有偏，严谨做法是 **permutation importance** 或 SHAP；
- **不能外推**：树模型预测值永远在训练目标范围内，时间趋势类任务要先做差分/去趋势。

---

## 6. 面试高频问题速查

1. **信息增益、增益率、基尼的区别？** → 增益偏好多取值特征；增益率除以特征熵做修正；基尼是熵的近似但不用 log，更快。
2. **Bagging 和 Boosting 的本质区别？** → 并行独立采样降方差 vs 串行拟合残差降偏差。
3. **随机森林两重随机是什么？为什么？** → 样本 bootstrap + 特征子集，为了降低树间相关性 $\rho$，让平均真正降方差。
4. **GBDT 拟合的是残差还是负梯度？** → 负梯度；平方损失时恰好等于残差。
5. **GBDT 用的是分类树还是回归树？** → 永远是回归树（拟合连续的负梯度）。
6. **XGBoost 相比 GBDT 的改进？** → 二阶泰勒、显式正则、缺失值稀疏感知、列采样、近似分位点、工程并行。
7. **XGBoost 的分裂增益公式？** → $\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}]-\gamma$。
8. **XGBoost 是并行训练树的吗？** → 不是，树间串行；并行发生在特征分裂点查找。
9. **LightGBM 的四大加速点？** → 直方图、leaf-wise、GOSS、EFB。
10. **LightGBM 为什么容易过拟合？** → leaf-wise 深度不受限，需 `num_leaves`、`min_data_in_leaf`、`max_depth` 一起约束。
11. **树模型要不要归一化？** → 不需要；但需要处理类别特征和缺失策略。
12. **为什么表格数据 GBDT 仍强于 DNN？** → 对异质/稀疏/缺失/非光滑特征鲁棒，超参少、样本效率高。

---

## 参考

- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017
- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS 2018
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001
- Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022
