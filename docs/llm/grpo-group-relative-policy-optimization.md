# GRPO 算法详解（LLM 八股 17）

> **更新时间**：2026-08-20

> **标签**：GRPO、强化学习、PPO、去 critic、组内相对优势、面试八股

> **论文**：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv:2402.03300（Shao et al., 2024）

> **一句话**：GRPO（Group Relative Policy Optimization）用同一 prompt 的**一组 G 个采样回复的相对奖励**作为 baseline 来估计优势，**去掉了 PPO 中的 value model（critic）**，把策略网络和参考网络的 KL 散度作为约束项加在 loss 里——是 DeepSeek-R1、V4 训练推理能力的核心算法。

---

## 1. 背景：为什么需要 GRPO

### 1.1 强化学习在 LLM 中的地位

大模型对齐/训练推理能力主要靠 RL：

| 任务 | 算法 |
| --- | --- |
| 对齐人类偏好 | RLHF（PPO） |
| 提升数学/代码推理 | GRPO（DeepSeek-R1、o1 类） |
| 工具调用 Agent | GRPO、APO |
| 多模态对齐 | DPO、GRPO |

GRPO 是 PPO 的改良，专门为 LLM 的"生成-评估"训练范式设计。

### 1.2 PPO 的两大痛点

PPO（Proximal Policy Optimization）原本是 2017 年 OpenAI 提出的连续控制算法，2017 年 InstructGPT 把它用到 LLM RLHF：

PPO 的核心：

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中 `r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)` 是新旧策略比，`A_t` 是**优势函数**（advantage）。

PPO 在 LLM 上的两大痛点：

| 痛点 | 原因 | 代价 |
| --- | --- | --- |
| **需要 value model（critic）** | 优势 A_t 通常用 GAE 计算，需要 V(s_t) 的估计 | 训练 1 个 LLM 价值的"大"模型，内存+算力翻倍 |
| **绝对奖励值不可比** | 不同 prompt 的奖励绝对值差异大 | 训练不稳定 |

GRPO 的回应：

1. **去 critic**：用同一 prompt 的**一组采样**作为 baseline 估计优势，**不需要 value model**；
2. **组内归一化**：对每组 G 个样本的奖励做归一化，得到组内的相对优势，**避免绝对奖励的尺度问题**；
3. **KL 散度进 loss**：用 KL 无偏估计器把 KL 加到 loss 里，**确保策略不会偏离太远**。

> GRPO 不是替代 PPO 的"革命"，而是 PPO 的"工程化精简"——保留 PPO 的核心思想（clip + 优势 + KL 约束），但去掉了 value model。

---

## 2. 直觉：用同一道题做几次，看哪个更好

直觉类比：

- **PPO** = 老师**逐个**批改学生作业，参考"过去学生平均水平"（value model）打分；
- **GRPO** = 老师让一个学生**做同一道题 8 次**（采样 G 个回复），**用这 8 次的相对好坏**当"奖励"——8 次中最好的当正例，最差的当负例。

具体：
- 一道 prompt q → 让当前策略生成 G 个回复 `{o_1, o_2, ..., o_G}`；
- 用 reward model 给每个回复打分 `{r_1, r_2, ..., r_G}`；
- **归一化**：`A_i = (r_i - mean(r)) / std(r)`，得到"组内相对优势"；
- 用优势做策略更新。

> 面试高频：**"为什么去 critic 还能 work？"**——因为同组 G 个样本的奖励**分布**（mean、std）就是很好的 baseline。G 越大，组内归一化越准。

---

## 3. GRPO 算法流程

### 3.1 完整算法（伪代码）

参考 DeepSeekMath 论文 Algorithm 1（位于 p.14）：

```
Algorithm 1: Iterative Group Relative Policy Optimization
─────────────────────────────────────────────────
Input: initial policy model π_θ_init, reward models r_φ, task prompts D,
       hyperparameters ε, β, μ

1:  policy model π_θ ← π_θ_init
2:  for iteration = 1, ..., I do
3:      reference model π_ref ← π_θ
4:      for step = 1, ..., M do
5:          Sample a batch D_b from D
6:          Update the old policy model π_θ_old ← π_θ
7:          Sample G outputs {o_i}_{i=1}^G ~ π_θ_old(·|q) for each q in D_b
8:          Compute rewards {r_i}_{i=1}^G for each sampled output by running r_φ
9:          Compute Â_i for the t-th token of o_i through group relative advantage estimation
10:         for GRPO iteration = 1, ..., μ do
11:             Update the policy model π_θ by maximizing the GRPO objective (Eq. 21)
12:     Update r_φ through continuous training using a replay mechanism

Output: π_θ
```

### 3.2 三大步骤详细分解

**Step 1：组采样**  
对每个 prompt q 采样 G 个回复 `{o_i}_{i=1}^G ~ π_θ_old(·|q)`。论文 G=16。

**Step 2：组内相对优势估计**

$$
\hat{A}_{i,t} = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \ldots, r_G\})}{\mathrm{std}(\{r_1, r_2, \ldots, r_G\})}
$$

注意优势是**token 级别**——整个回复 o_i 的所有 token 共享同一个优势值（因为回复级的奖励是 outcome-level 的）。

**Step 3：GRPO 目标函数**

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \mathrm{clip}\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right) - \beta \, \mathbb{D}_{\mathrm{KL}}\left( \pi_\theta \mid\mid \pi_{\mathrm{ref}} \right) \right]
$$

其中：
- 第一个 min = PPO 的 clip 目标；
- `β` = KL 系数（论文 β=0.04）；
- `D_KL(π_θ || π_ref)` = 当前策略与参考策略的 KL 散度。

### 3.3 KL 散度的无偏估计

DeepSeekMath 用 Schulman 2020 的**无偏 KL 估计器**（避免 KL 散度的有偏估计）：

$$
\mathbb{D}_{\mathrm{KL}}\left( \pi_\theta \mid\mid \pi_{\mathrm{ref}} \right) = \frac{\pi_{\mathrm{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{\mathrm{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1
$$

**保证非负**（用 Schulman 2020 的 Trick 5 估计器，比直接估计 `log(π_θ/π_ref)` 稳定）。

### 3.4 三种 GRPO 变体

| 变体 | 监督信号 | 适用场景 | R1 用哪个 |
| --- | --- | --- | --- |
| **Outcome Supervision** | 整个回复的最终奖励 | 数学、代码（有确定答案） | **R1 推理 RL 阶段** |
| **Process Supervision** | 每一步推理的奖励 | 复杂数学（多步推理） | DeepSeekMath 论文 |
| **Iterative** | 不断用最新的策略生成新训练集 | 训练集耗尽时 | DeepSeekMath 论文 |

---

## 4. 与 PPO 的对比

| 维度 | PPO | GRPO |
| --- | --- | --- |
| **Value model** | 需要（与策略同尺寸的 critic） | **不需要** |
| **Baseline** | V(s) from critic | **同组 G 个样本的 mean** |
| **优势估计** | GAE（generalized advantage estimation） | **组内归一化** |
| **KL 约束** | 加在 reward 上 | **直接加在 loss 上** |
| **内存** | 2×（策略 + value） | **1×**（仅策略） |
| **训练稳定性** | 易受 reward 尺度影响 | **更稳定**（组内归一化） |
| **Reward Hacking 风险** | 中 | 低（rule-based reward） |

### 4.1 关键差异图示

```
PPO:
prompt → policy π → 1 个回复
                          ↓
                   reward model → 标量奖励 r
                          ↓
                value model V(s) → 优势 A = r - V(s)
                          ↓
                  PPO loss with clip + KL

GRPO:
prompt → policy π → G 个回复 {o_1, ..., o_G}
                          ↓
                   reward model → 标量奖励 {r_1, ..., r_G}
                          ↓
                组内归一化 A_i = (r_i - mean) / std
                          ↓
                  GRPO loss with clip + KL
```

> 关键 takeaway：**GRPO 把"baseline 估计"从"训练一个大 critic"变成"同组 G 个采样"**——用采样换模型，简单粗暴但有效。

---

## 5. 公式细节深入

### 5.1 Clip 目标

GRPO 沿用 PPO 的 clip 机制：

$$
\min\left( r_t(\theta) A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
$$

其中 `r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)` 是**重要性采样比**。

- 当 `A_t > 0`（好动作）且 `r_t(θ) > 1+ε` → clip 后用 ε 限制，避免策略过激更新；
- 当 `A_t < 0`（坏动作）且 `r_t(θ) < 1-ε` → clip 后用 -ε 限制。

> 论文 ε=0.2（与 PPO 默认一致）。

### 5.2 KL 散度作为 loss 项

把 KL 加到 loss 而不是 reward 上：

- **传统做法**（PPO）：`r_total = r - β * KL(π || π_ref)`；
- **GRPO**：直接 `loss = L_clip + β * KL(π_θ || π_ref)`。

GRPO 的好处：**KL 在 loss 里可以做梯度反向传播，约束更精确**。

### 5.3 重要超参

DeepSeek-R1 配置（来自 R1 论文）：

| 超参 | 值 | 含义 |
| --- | --- | --- |
| G | 16 | 每 prompt 采样数 |
| μ | 1 | 内层 epoch 数 |
| β | 0.04 | KL 系数 |
| ε | 0.2 | clip 范围 |
| lr | 1e-6 | 策略学习率 |
| KL estimator | 无偏 | 避免 KL 估计有偏 |

> R1 的 G=16 是工程折中：G 越大组内归一化越准，但**单步训练算力线性增加**。

---

## 6. GRPO 的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def grpo_loss(policy, ref_policy, prompts, gen_ids, rewards, gen_mask,
              clip_eps=0.2, beta=0.04, group_size=16):
    """GRPO loss（DeepSeek-R1 风格）

    Args:
        policy:  当前策略模型
        ref_policy: 参考策略（通常是初始化时的策略）
        prompts:  (B, T_prompt)
        gen_ids:  (B*G, T_gen) 生成的回复
        rewards:  (B*G,)       每个回复的奖励
        gen_mask: (B*G, T_gen) 生成 token 的 mask
        group_size: G
    """
    B = prompts.size(0) // group_size
    T_gen = gen_ids.size(1)

    # 1) 策略对数概率
    logits = policy(input_ids=torch.cat([prompts, gen_ids], dim=1)).logits
    log_probs = F.log_softmax(logits[:, -T_gen-1:-1], dim=-1)  # (B*G, T_gen, V)
    log_probs = log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B*G, T_gen)

    # 2) 参考策略对数概率
    with torch.no_grad():
        ref_logits = ref_policy(input_ids=torch.cat([prompts, gen_ids], dim=1)).logits
        ref_log_probs = F.log_softmax(ref_logits[:, -T_gen-1:-1], dim=-1)
        ref_log_probs = ref_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
        old_log_probs = log_probs.detach()  # π_old

    # 3) 组内归一化优势 (B, G)
    rewards_grouped = rewards.view(B, group_size)
    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    std = rewards_grouped.std(dim=-1, keepdim=True) + 1e-8
    advantages = ((rewards_grouped - mean) / std).view(-1)         # (B*G,)
    # token 级别展开
    advantages = advantages.unsqueeze(-1).expand(-1, T_gen)         # (B*G, T_gen)

    # 4) 重要性采样比 r_t
    ratio = (log_probs - old_log_probs).exp()                       # (B*G, T_gen)

    # 5) PPO clip 目标
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)                          # (B*G, T_gen)

    # 6) KL 散度（无偏估计器）
    # D_KL(π_θ || π_ref) ≈ exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    log_ratio = log_probs - ref_log_probs
    kl = (log_ratio.exp() - log_ratio - 1)                          # (B*G, T_gen)
    kl_loss = beta * kl

    # 7) 总 loss
    loss = (policy_loss + kl_loss) * gen_mask
    loss = loss.sum() / gen_mask.sum()

    # 8) 监控：KL 散度均值
    return loss, {
        'kl': kl.detach().mean().item(),
        'ratio': ratio.detach().mean().item(),
        'reward': rewards.mean().item(),
        'advantage_std': std.mean().item(),
    }


class RewardModel:
    """简单的标量 reward model（实际生产中用过程式规则或训练好的 RM）"""
    def __init__(self, fmt_reward_weight=0.1):
        self.fmt_weight = fmt_reward_weight

    def __call__(self, prompts, responses):
        # 假设 responses 是字符串列表
        rewards = []
        for r in responses:
            acc = 1.0 if '答案正确' in r else 0.0  # 假设规则判定
            fmt = 1.0 if '<think>' in r and '</think>' in r else 0.0
            rewards.append(acc + self.fmt_weight * fmt)
        return torch.tensor(rewards)
```

> 实际生产中 `policy` 和 `ref_policy` 是**同一个模型的两份副本**，ref_policy 在每次外层 iteration 后用 policy 替换。

---

## 7. GRPO 的现代演进

| 论文/方法 | 改进点 | 时间 |
| --- | --- | --- |
| DeepSeekMath | GRPO 首发 | 2024-02 |
| DeepSeek-R1 | R1 用 GRPO 训推理 | 2025-01 |
| DAPO（字节） | Clip-Higher、Token-level loss、Overlong reward Shaping | 2025 |
| GSPO（Qwen） | 序列级 importance sampling | 2025 |
| GRPO + Process Reward | 引入过程监督，缓解 reward hacking | 持续 |
| Online Preference Optimization（OPPO） | 在线偏好与 GRPO 结合 | 2025 |
| VinePPO | value baseline 替代组 baseline | 2024 |

### 7.1 DAPO（字节）

针对 GRPO 在长 CoT 上的几个问题做的改进：
- **Clip-Higher**：clip 上界 1+ε 提到更高的值（0.28），防止高质量样本被 clip；
- **Token-level loss**：对 token 级别做 loss（而不是 sequence-level mean）；
- **Overlong Reward Shaping**：惩罚过长回复；
- **Dynamic Sampling**：过滤优势全 0 或全 1 的组。

### 7.2 GSPO（Qwen）

序列级 importance sampling 替代 token 级：

$$
J_{\mathrm{GSPO}} = \mathbb{E}\left[ \min\left( \frac{\pi_\theta(o|q)}{\pi_{\theta_{\mathrm{old}}}(o|q)} \hat{A}, \mathrm{clip}\left( \frac{\pi_\theta(o|q)}{\pi_{\theta_{\mathrm{old}}}(o|q)}, 1-\varepsilon, 1+\varepsilon \right) \hat{A} \right) \right] - \beta \mathrm{KL}
$$

> 用**整序列的 ratio** 替代 token 级 ratio，**训练更稳定**（避免 token 级 ratio 的高方差）。

---

## 8. GRPO 的"奖励设计"哲学

GRPO 的强大在于：**配合 Rule-based reward，避开 reward hacking**。

### 8.1 R1 的奖励设计

DeepSeek-R1 用三类规则式 reward（**纯规则，无需训练 RM**）：

| 任务 | Accuracy Reward | Format Reward | Language Consistency |
| --- | --- | --- | --- |
| 数学 | 答案是否匹配 | `<think>...</think><answer>...</answer>` | — |
| 代码 | 单元测试通过 | 格式正确 | — |
| 通用 | 模型自评 | 格式 | 中英混杂比例 |

> 论文明确说**不用神经奖励模型**——因为"neural reward models are susceptible to reward hacking"。

### 8.2 为什么 rule-based reward 更安全

- **可解释**：规则明确，行为可追溯；
- **抗 hacking**：模型无法"骗"过规则；
- **低算力**：不需要训练和维护 reward model；
- **可组合**：多个规则可线性加权。

> 面试高频：**"GRPO 的奖励怎么设计？"**——优先用规则式（accuracy + format），不训练 RM。复杂任务用 LLM-as-judge 时要注意防 reward hacking。

---

## 9. GRPO 在 R1 / V4 中的应用

### 9.1 R1 的两阶段 GRPO

1. **推理导向 RL**：用 R1-Zero 的 base + GRPO 训练"长 CoT 推理能力"；
2. **全场景 RL**：在 R1 base 上继续 GRPO，**用 RM 而非纯规则**，加入 IF（指令遵循）和 helpfulness 奖励。

### 9.2 V4 的 Specialist Training

V4 沿用 GRPO 作为 Specialist 训练的核心：

```
多个 Specialist 专家（数学/代码/Agent/IF）
   │
   └── 每个专家独立 SFT + GRPO
        │
        └── Specialist 各自的奖励模型（规则式 or V3/R3 蒸馏）
```

详见 [[/docs/llm/deepseek-family.md]]（Part 5 · V4 训练与后训练）。

---

## 10. 面试高频问题速查

1. **GRPO 是什么？**
   Group Relative Policy Optimization，用**同一 prompt 的一组 G 个回复的相对奖励**做优势估计，**去掉了 PPO 的 value model**。DeepSeekMath 首发（arXiv:2402.03300）。

2. **GRPO 相对 PPO 的最大改进？**
   **去 critic**。用同组 G 个采样的 mean 作为 baseline，省掉一个 LLM 尺寸的 value model，**内存减半**。

3. **GRPO 的优势怎么算？**
   `A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})`，组内归一化。G 个回复共享同一 prompt，奖励可比。

4. **GRPO 怎么避免 reward hacking？**
   用**规则式奖励**（accuracy + format），不训练神经 RM。规则明确、可解释、抗 hacking。

5. **GRPO 的 KL 散度怎么算？**
   用 Schulman 2020 的**无偏估计器**：`D_KL ≈ exp(log_ratio) - log_ratio - 1`（其中 log_ratio = log(π_ref / π_θ)），保证非负。

6. **GRPO 的 G 一般多大？**
   DeepSeek-R1 用 G=16。G 越大组内归一化越准，但单步算力线性增加。**8-32 是常见范围**。

7. **GRPO 怎么和 DPO 区分？**
   - **DPO**：用偏好对 (preferred, rejected) 直接做监督学习，**无需在线采样**；
   - **GRPO**：在线采样 G 个回复，用相对奖励做 PPO 风格更新，**必须在线 rollout**。

8. **GRPO 的核心创新是？**
   **用组内归一化代替 value model**——简单、内存省、训练稳定。

9. **GRPO 适合哪些任务？**
   - 推理任务（数学、代码、逻辑）：rule-based reward 干净；
   - Agent 任务（工具调用）：任务成功率可验证；
   - 通用对齐：需要 RM 或 LLM-as-judge，**reward hacking 风险**。

10. **GRPO 的局限？**
    - **同组内奖励分布必须足够分散**：如果所有回复奖励相近（如都 0 或都 1），归一化无意义，训练无效；
    - **G 大了算力贵**，小了归一化不准；
    - **复杂任务**难以用规则式 reward，需要神经 RM。

11. **GRPO 与 DAPO 的关系？**
    DAPO 是字节 2025 年对 GRPO 在长 CoT 上的改进版：① Clip-Higher；② Token-level loss；③ Overlong Reward Shaping；④ Dynamic Sampling。

12. **GRPO 与 GSPO 的关系？**
    GSPO 是 Qwen 2025 年对 GRPO 的改进版：用**序列级 importance sampling** 替代 token 级，**训练更稳定**。

13. **R1 的两阶段 RL 用 GRPO 吗？**
    是的，**两阶段都用 GRPO**：
    - **阶段 1**（推理导向 RL）：rule-based reward（accuracy + format），训练长 CoT 推理；
    - **阶段 2**（全场景 RL）：RM-based reward（用 V3 蒸馏的 preference model），加入 IF 和 helpfulness。

14. **GRPO 与 Reinforce++ 的关系？**
    Reinforce++ 是 GRPO 的**前身**，同样去 critic，但用全 episode 奖励。GRPO 加了**组内归一化 + clip + KL**，更稳定。

15. **GRPO 的目标函数中有哪些可调超参？**
    - ε = 0.2（clip range）
    - β = 0.04（KL 系数，越大越保守）
    - G = 16（组大小）
    - μ = 1（内层 epoch 数）
    - lr = 1e-6（学习率）

16. **GRPO 在多轮 Agent 任务中怎么用？**
    把整条轨迹当成一个"回复"，**用最终任务成功率**作为奖励。G 个轨迹的相对成功率作为优势。详见 V4 Agent 训练。

17. **GRPO 的 KL 散度加在 reward 还是 loss？**
    **直接加在 loss**（不像 PPO 加在 reward）。好处是 KL 可以梯度反向传播，约束更精确。

18. **GRPO 与 PPO 的训练稳定性对比？**
    GRPO 更稳定，**因为组内归一化天然处理了 reward 尺度问题**。PPO 依赖 value model 的拟合质量。

19. **GRPO 的优势函数为什么 token 级相同？**
    outcome-level reward（如答案是否正确）下，整个回复 o_i 的所有 token 共享同一个 A_i。
    区别于 process-level reward：每步推理可以给不同奖励，需要额外标注。

20. **GRPO 的内存节省具体多少？**
    PPO 训练 671B 模型 + value model（671B）= 2×671B ≈ 1.3TB 主权重。GRPO 仅策略 = 671B 主权重。**主权重内存减半**（不含 activation/optimizer 状态）。

---

## 11. 一图流：GRPO 流程

```
prompt q
   │
   ▼
策略 π_θ_old 采样 G=16 个回复 {o_1, ..., o_G}
   │
   ▼
reward model 评估 {r_1, ..., r_G}
   │
   ▼
组内归一化：A_i = (r_i - mean(r)) / std(r)
   │
   ▼
GRPO loss（clip + KL）：
   L = -E[min(r_t·A, clip(r_t, 1-ε, 1+ε)·A) - β·D_KL(π_θ || π_ref)]
   │
   ▼
策略 π_θ 更新（多次内层 epoch，μ 次）
   │
   ▼
新策略 → 进入下一轮采样
```

---

## 12. 参考

- Shao et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv:2402.03300（GRPO 原始论文）
- Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347（PPO 原始）
- Schulman et al., Approximating KL Divergence, 2020（KL 无偏估计器）
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL, arXiv:2501.12948
- Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale, 2025
- Zheng et al., GSPO: Group Sequence Policy Optimization, 2025
- Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, NeurIPS 2023（DPO）
- 相关文章：
  - [[/docs/llm/deepseek-family.md]]（Part 3 · R1 / Part 5 · V4 Specialist / Part 1 · V3 后训练）
