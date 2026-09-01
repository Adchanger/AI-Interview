# 模型评估指标：P/R/F1、ROC-AUC、PR-AUC（ML 八股 05）

> **更新时间**：2026-08-31

> **标签**：评估指标、AUC、精确率召回率、类别不平衡、面试八股

> **一句话**：分类指标都从混淆矩阵长出来——精确率/召回率关注单一阈值下的正类质量，ROC-AUC 是"随机正样本得分高于随机负样本"的概率且与类别比例无关，PR-AUC 在极端不平衡下更敏感。

> **关联阅读**：[[/docs/ml/feature-engineering-and-imbalance.md]]、[[/docs/ml/logistic-regression.md]]

---

## 1. 混淆矩阵与基础指标

|  | 预测正 | 预测负 |
|--|--------|--------|
| **真实正** | TP | FN |
| **真实负** | FP | TN |

$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad \text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN}$$

$$F_1 = \frac{2PR}{P+R},\qquad F_\beta = \frac{(1+\beta^2)PR}{\beta^2 P + R}$$

- $\beta>1$（如 $F_2$）**偏重召回**（漏检代价高：疾病筛查、风控黑产）；
- $\beta<1$（如 $F_{0.5}$）**偏重精确率**（误报代价高：推送、封号）。

> 面试高频：**为什么不用准确率？** → 类别极不平衡时准确率被多数类主导。1% 正样本的场景，全预测负就有 99% 准确率，却毫无用处。

**其他派生指标**：
- TPR = Recall = 灵敏度（Sensitivity）；
- TNR = 特异度（Specificity）= $TN/(TN+FP)$；
- FPR = $1-\text{TNR} = FP/(FP+TN)$；
- 宏平均（macro，各类平均，重视小类）/ 微平均（micro，汇总 TP/FP，等于多分类 accuracy）/ 加权平均（weighted）。

---

## 2. ROC 与 AUC

**ROC 曲线**：以 FPR 为横轴、TPR 为纵轴，遍历所有阈值画出的曲线。左上角 (0,1) 是完美分类器，对角线是随机猜。

**AUC = ROC 曲线下面积**，三个必背性质：

1. **概率解释**：AUC = $P(\hat s_{\text{正}} > \hat s_{\text{负}})$，即随机取一正一负样本，模型给正样本更高分的概率。
2. **与 Wilcoxon-Mann-Whitney 统计量等价**，因此可用秩来算（见下方代码）：
   $$\mathrm{AUC} = \frac{\sum_{i\in \text{正}} \mathrm{rank}_i - \frac{M(M+1)}{2}}{M\cdot N}$$
   （$M$ 正样本数，$N$ 负样本数，rank 为按分数升序的秩，同分取平均秩）
3. **只关心排序，不关心分数绝对值**：分数做任何单调变换 AUC 不变 → 所以 **AUC 高不代表概率校准好**。

### 2.1 AUC 的关键特性

- **对类别不平衡不敏感**：TPR 只用正样本、FPR 只用负样本，改变正负比例不影响两者 → 这既是优点（跨数据集可比）也是缺点（掩盖了极端不平衡下 FP 的绝对数量巨大）。
- **阈值无关**：适合模型选型和排序类任务（CTR、召回、风控评分卡）。

---

## 3. PR 曲线与 PR-AUC

**PR 曲线**：横轴 Recall、纵轴 Precision。基线是正样本比例 $\pi$（随机模型的 Precision ≈ $\pi$）。

> 面试高频：**ROC 与 PR 怎么选？**
> - 关心**负样本占绝大多数、且 FP 代价高**（欺诈检测、疾病筛查、检索/推荐 TopK）→ 用 **PR-AUC / AP**：因为 Precision 分母含 FP，负样本一多，FPR 变化不明显但 Precision 会剧烈下降，PR 更能区分模型好坏；
> - 关心**整体排序能力、正负比例会变**（不同时间段、不同流量）→ 用 **ROC-AUC**，它与类别比例解耦。

**AP（Average Precision）**：$\mathrm{AP}=\sum_k (R_k - R_{k-1})P_k$，即 PR 曲线的阶梯积分，是目标检测 mAP 的基础。

---

## 4. 其他常见指标

| 场景 | 指标 | 说明 |
|------|------|------|
| 风控/评分卡 | **KS** = $\max(TPR-FPR)$ | 最大区分度；KS>0.3 可用（业务经验值） |
| 概率校准 | **Brier score**、可靠性图、ECE | 与 AUC 互补：AUC 看排序，校准看概率准不准 |
| 回归 | MAE / MSE / RMSE / MAPE / $R^2$ | MSE 对离群敏感；MAPE 在真值接近 0 时爆炸 |
| 排序/推荐 | Recall@K、Precision@K、**MRR**、**NDCG**、Hit Rate | NDCG 考虑位置折损与多级相关性 |
| 多分类 | macro/micro/weighted F1、混淆矩阵、Cohen's Kappa | 小类重要用 macro |
| 检测/分割 | IoU、mAP、Dice | Dice 与 F1 同形式 |
| 聚类 | 轮廓系数、CH 指数、ARI、NMI | 前两者无标签，后两者需真标签 |
| 生成/LLM | 见 [[/docs/llm/hallucination-and-evaluation.md]] | BLEU/ROUGE/Perplexity/LLM-as-Judge |

---

## 5. 阈值选择与业务落地

模型输出是分数，落地必须选阈值：

1. **按业务约束**：固定"每天最多人工审核 1000 单" → 取 TopN 对应的分数；
2. **按代价矩阵**：$\text{threshold}^* = \frac{C_{FP}}{C_{FP}+C_{FN}}$（在概率校准良好前提下）；
3. **按指标最优**：在验证集上扫阈值取 F1/F2 最大点（注意别在测试集上调）；
4. **多目标**：画 Precision-Recall-阈值三线图，与业务方一起定。

> 面试高频：**采样（下采样负样本）之后概率怎么修正？** → 采样改变了先验，需还原：
> $$p = \frac{p_s}{p_s + (1-p_s)/r}$$
> 其中 $p_s$ 是采样后模型输出，$r$ 是负样本采样率。或直接用 Platt/Isotonic 重新校准。

---

## 6. 手撕代码：用秩公式算 AUC

```python
import numpy as np

def auc_by_rank(y_true, y_score):
    """y_true: 0/1 数组; y_score: 预测分数。O(N log N)，同分取平均秩"""
    y_true = np.asarray(y_true)
    order = np.argsort(y_score, kind="mergesort")
    s = np.asarray(y_score)[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):                       # 处理并列：赋平均秩
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1  # 秩从 1 开始
        i = j + 1
    pos_rank_sum = ranks[y_true[order] == 1].sum()
    M = int(y_true.sum())
    N = len(y_true) - M
    if M == 0 or N == 0:
        return float("nan")                 # 单类别时 AUC 无定义
    return (pos_rank_sum - M * (M + 1) / 2) / (M * N)
```

---

## 7. 面试高频问题速查

1. **AUC 的物理含义？** → 随机正样本分数高于随机负样本的概率；等价于 Wilcoxon-Mann-Whitney 统计量。
2. **AUC 怎么快速计算？** → 秩公式 $\frac{\sum \mathrm{rank}_{pos} - M(M+1)/2}{MN}$。
3. **AUC 对不平衡敏感吗？** → 不敏感（TPR/FPR 分别只用一类样本），因此极端不平衡时要看 PR-AUC。
4. **ROC 与 PR 怎么选？** → 负样本极多且在意 FP → PR；关心通用排序能力、正负比例会漂移 → ROC。
5. **准确率什么时候失效？** → 类别不平衡、代价不对称时。
6. **精确率和召回率为什么矛盾？** → 同一分数排序下，降低阈值召回上升、精确率通常下降，是同一 PR 曲线上的移动。
7. **F1 为什么用调和平均？** → 调和平均对小值敏感，任何一项很低 F1 就低，避免"一头沉"。
8. **AUC=0.5 意味着什么？** → 与随机等价；小于 0.5 说明排序反了（可尝试取反）。
9. **AUC 高但线上效果差，可能原因？** → 概率未校准、离线/线上特征不一致、样本穿越泄漏、评估集与线上分布不同、指标与业务目标不匹配（应看 TopK 精确率）。
10. **多分类怎么算 AUC？** → OvR / OvO 后宏平均或加权平均（sklearn 的 `multi_class='ovr'/'ovo'`）。
11. **NDCG 与 MRR 区别？** → MRR 只看第一个正确结果的位置倒数；NDCG 考虑全部结果的分级相关性与位置折损。
12. **回归指标怎么选？** → 关注大误差用 RMSE，关注鲁棒用 MAE，跨量纲比较用 $R^2$/MAPE（真值近 0 慎用 MAPE）。

---

## 参考

- 周志华《机器学习》第 2 章
- Davis & Goadrich, *The Relationship Between Precision-Recall and ROC Curves*, ICML 2006
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot*, 2015
- Fawcett, *An Introduction to ROC Analysis*, 2006
