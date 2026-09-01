# Agent 基础：ReAct、Function Calling 与记忆（Agent 八股 01）

> **更新时间**：2026-08-31

> **标签**：Agent、ReAct、FunctionCalling、记忆、规划、面试八股

> **一句话**：Agent = LLM（大脑）+ 工具（手）+ 记忆（经验）+ 规划与循环控制（意志），ReAct 用"思考-行动-观察"的循环把工具结果反馈回上下文，Function Calling 是让模型输出结构化调用参数的底层能力，而工程上真正难的是**循环终止、上下文治理、失败兜底与成本控制**。

> **关联阅读**：[[/docs/agent/mcp-multi-agent-context-engineering.md]]、[[/docs/rag/rag-basics.md]]、[[/docs/llm/reasoning-and-test-time-scaling.md]]

---

## 1. 为什么需要 Agent

单次 LLM 调用的局限：
1. **知识与时效**：不能查最新信息 → 需要检索/搜索工具；
2. **计算不可靠**：算术、日期、单位换算易错 → 需要代码/计算器；
3. **不能改变世界**：不能下单、发邮件、改数据库 → 需要动作执行；
4. **一次生成不能试错**：复杂任务需要"做一步看一步" → 需要循环与反馈。

**Agent 的本质**：把 LLM 从"一次问答"变成"**带反馈的多步决策循环**"。

---

## 2. 四大组件

| 组件 | 内容 |
|------|------|
| **规划（Planning）** | 任务分解、路径选择、反思与重规划（ReAct、Plan-and-Execute、ToT） |
| **工具（Tools）** | 检索、搜索、代码执行、数据库、API、浏览器、文件系统 |
| **记忆（Memory）** | 短期（上下文窗口/对话历史）、长期（向量库/结构化存储）、工作记忆（scratchpad）、程序性记忆（技能/SOP） |
| **执行/控制（Loop & Harness）** | 循环调度、终止条件、预算控制、错误重试、审批与安全边界 |

---

## 3. ReAct（Reasoning + Acting）

### 3.1 循环结构

```
Thought: 我需要先查一下 X
Action: search
Action Input: {"q": "X"}
Observation: <工具返回结果>
Thought: 结果说明 Y，还需要计算 Z
Action: python
Action Input: {"code": "..."}
Observation: 42
Thought: 已经足够回答
Final Answer: ...
```

**核心**：把 **CoT（推理）与工具调用（行动）交错**——推理决定下一步动作，观察结果又修正推理。相比纯 CoT，ReAct 有外部事实约束，显著降低幻觉；相比只调工具，有推理来决定"调什么、为什么"。

### 3.2 终止与防死循环（最高频工程题）

| 机制 | 说明 |
|------|------|
| 显式终止 | 模型输出 `Final Answer` / 调用 `finish` 工具 |
| **步数上限** | `max_iterations`（如 10–30），超出则强制总结现有信息作答 |
| **预算上限** | token / 费用 / 墙钟时间三重预算，任一触发即收尾 |
| **重复动作检测** | 相同 (tool, args) 连续出现 → 注入提示"该动作已执行且结果为…，请换思路"或直接中断 |
| **无进展检测** | 连续 N 步没有新增有效信息（观察为空/报错）→ 降级 |
| 工具失败兜底 | 重试（指数退避）→ 换备用工具 → 降级为"说明无法完成 + 已获信息" |
| 循环外看护 | 外层 watchdog 超时终止，避免线上请求悬挂 |

> 面试标准答法：「**三层保险**：模型侧给明确终止指令与 finish 工具；框架侧设步数/预算/重复检测；系统侧有超时与人工介入通道。任何情况下都必须能给用户一个可解释的结果，而不是转圈或空响应。」

---

## 4. Function Calling / Tool Use

### 4.1 实现机制

1. 开发者用 **JSON Schema** 描述工具（name、description、parameters）；
2. 工具描述被注入模型上下文（多数实现放在 system prompt 或专门的 tools 段）；
3. 模型输出**结构化调用**（`{"name": "get_weather", "arguments": {"city":"北京"}}`）；
4. **框架执行**函数（模型自己不执行！这是常见误解），把结果作为 `tool` 角色消息回填；
5. 模型基于结果继续推理或作答。

**为什么模型能输出合法 JSON**：训练阶段用大量工具调用数据做 SFT/RL；推理时可叠加**约束解码**（按 schema 编译 FSM 掩码非法 token）确保 100% 合法，见 [[/docs/llm/decoding-strategies.md]]。

### 4.2 工具设计的工程经验

| 原则 | 说明 |
|------|------|
| **描述即 prompt** | 工具的 description 决定模型会不会正确使用它，要写清用途、边界、参数示例 |
| **数量控制** | 工具太多（>20–30）会显著降低选择准确率 → 分组 + 二级路由（先选工具集再选工具） |
| **粒度适中** | 太细导致多轮调用爆炸，太粗导致参数复杂难填 |
| **返回精简** | 观察结果要裁剪（只返回必要字段），否则几步就把上下文撑爆 |
| **幂等与可重试** | 写操作要幂等键，避免重试造成重复下单 |
| **错误信息可行动** | 返回"参数 city 缺失，示例：北京"而不是 stack trace |
| **权限与确认** | 危险操作（删除、支付、发送）走人工审批或二次确认 |
| **并行调用** | 无依赖的工具可并行（现代 API 支持一次返回多个 tool call），降低延迟 |

---

## 5. 规划范式对比

| 范式 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **ReAct** | 边想边做，每步决定下一动作 | 灵活、可纠错、实现简单 | 无全局视野、可能绕路、步数不可控 |
| **Plan-and-Execute** | 先出完整计划，再逐步执行（可中途重规划） | 结构清晰、可并行、token 更省（规划一次） | 计划僵化，环境变化需 replan |
| **Reflexion / 自我反思** | 失败后写"经验"进记忆，重试 | 可跨轮改进 | 需可靠的失败信号；无外部反馈时收益有限 |
| **树搜索 / LATS** | 分支 + 评估 + 回溯 | 复杂任务成功率高 | 成本高、实现复杂 |
| **Graph / 工作流编排** | 用状态图定义节点与边（LangGraph 类） | 可控、可观测、可检查点/人工审批 | 灵活性下降，需预先设计 |

**工程实践**：**确定性流程用编排，不确定部分才交给 Agent 自主决策**。「能用工作流解决的不要用 Agent」是 2025–2026 年被反复验证的经验（Agent 的自由度越高，可控性、成本与可复现性越差）。

---

## 6. 记忆系统

| 类型 | 存储 | 用途 | 关键问题 |
|------|------|------|----------|
| 短期 / 工作记忆 | 上下文窗口、scratchpad | 当前任务状态 | 窗口有限 → 压缩/摘要/快照 |
| 长期语义记忆 | 向量库 + 元数据 | 用户偏好、历史事实 | 何时写入、如何检索、如何遗忘 |
| **情景记忆** | 轨迹日志（含成功/失败） | 复用过往解法 | 检索相似任务、避免误用 |
| **程序性记忆** | 技能/SOP/工具组合脚本 | 固化"怎么做" | 版本管理与失效 |

**上下文治理**（Agent 长任务的核心难题）：
- **压缩/摘要**：历史对话滚动摘要（auto-compact），保留关键决策与结论；
- **状态外置**：把任务状态写入结构化文件/待办列表，而不是全靠上下文记住；
- **观察裁剪**：工具返回只留必要部分，大结果落盘 + 只放引用；
- **注意力稀释与任务漂移**：上下文越长模型越容易忘记原始目标 → 定期"复述目标"、用结构化 todo 锚定。

---

## 7. 评估与可观测

- **轨迹级评测**：不只看最终答案，还要看工具调用是否正确、步数、是否有无效循环；
- **指标**：任务成功率、平均步数、平均 token/成本、工具调用成功率、p95 延迟、人工介入率；
- **Trace**：记录每步的 thought/action/observation、耗时、token、模型版本，便于回放定位；
- **回归集**：把线上失败案例固化为测试用例，每次改 prompt/工具都跑一遍（Agent 极易"改一处坏一处"）。

---

## 8. 手撕/伪代码：最小 ReAct 循环

```python
import json

def react_agent(llm, tools, question, max_steps=10, max_tokens_budget=20000):
    """tools: {name: {"fn": callable, "schema": {...}}}"""
    history, used_tokens, seen = [], 0, set()
    for step in range(max_steps):
        resp = llm.chat(messages=build_messages(question, history),
                        tools=[t["schema"] for t in tools.values()])
        used_tokens += resp.usage.total_tokens
        if resp.tool_calls is None:                       # 模型给出最终答案
            return resp.content
        for call in resp.tool_calls:                      # 支持并行多调用
            key = (call.name, json.dumps(call.arguments, sort_keys=True))
            if key in seen:                               # 重复动作检测
                obs = "该动作已执行过，结果见上文，请换一种思路或直接作答。"
            elif call.name not in tools:
                obs = f"未知工具 {call.name}，可用：{list(tools)}"
            else:
                seen.add(key)
                try:
                    obs = truncate(tools[call.name]["fn"](**call.arguments))
                except Exception as e:                    # 错误信息要可行动
                    obs = f"工具执行失败：{e}；请检查参数或改用其他工具。"
            history.append({"call": call, "observation": obs})
        if used_tokens > max_tokens_budget:                # 预算护栏
            return llm.chat(messages=force_summarize(question, history)).content
    return llm.chat(messages=force_summarize(question, history)).content
```

要点：**终止条件、重复检测、错误可行动、预算护栏、超限强制总结**——这五点是面试官真正想听的。

---

## 9. 面试高频问题速查

1. **Agent 与单次 LLM 调用的区别？** → 带反馈的多步决策循环 + 工具 + 记忆，可获取实时信息并改变外部状态。
2. **Agent 的核心组件？** → 规划、工具、记忆、执行控制（循环与护栏）。
3. **ReAct 的循环是什么？** → Thought → Action → Observation 交错，推理指导行动、观察修正推理。
4. **ReAct 怎么终止、怎么防死循环？** → 显式 finish + 步数/预算上限 + 重复动作检测 + 无进展检测 + 超时看护。
5. **Function Calling 是模型自己执行函数吗？** → 不是，模型只输出结构化调用意图，执行由框架完成并回填结果。
6. **怎么保证参数 JSON 合法？** → 训练 + 约束解码（schema 编译成 FSM 掩码非法 token）。
7. **工具太多怎么办？** → 分组 + 二级路由/检索工具、精简描述、合并同类工具。
8. **ReAct 与 Plan-and-Execute 怎么选？** → 步骤不确定、需边做边调用 ReAct；结构清晰、可并行、要省 token 用 Plan-Execute；两者可混合（先粗规划再局部 ReAct）。
9. **Agent 记忆分几类？** → 短期/工作、长期语义、情景（轨迹）、程序性（技能）。
10. **长任务上下文爆了怎么办？** → 滚动摘要、状态外置到文件/todo、观察裁剪、大结果落盘只留引用。
11. **什么是任务漂移？怎么控？** → 长上下文中偏离原目标；用结构化 todo 锚定、定期复述目标、分段验收。
12. **Agent 怎么评估？** → 轨迹级评测：任务成功率、步数、成本、工具成功率、人工介入率 + 失败回归集。
13. **什么时候不该用 Agent？** → 流程确定、可枚举的任务用工作流/固定 pipeline 更可控、更便宜、更可复现。

---

## 参考

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, arXiv:2303.11366
- Schick et al., *Toolformer*, arXiv:2302.04761
- Wang et al., *A Survey on Large Language Model based Autonomous Agents*, arXiv:2308.11432
- Anthropic, *Building Effective Agents*, 2024（"能用工作流就别上 Agent"的工程观点）
- 卡码笔记《2026 年 Agent 大厂面试题汇总》
