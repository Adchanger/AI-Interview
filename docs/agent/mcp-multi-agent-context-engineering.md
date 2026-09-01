# MCP、多 Agent 与上下文工程（Agent 八股 02）

> **更新时间**：2026-08-31

> **标签**：MCP、A2A、多Agent、上下文工程、可观测性、面试八股

> **一句话**：MCP 把"模型 ↔ 工具/数据"的连接标准化（Host-Client-Server + Tools/Resources/Prompts 三大原语，走 JSON-RPC，本地 stdio 或远程 Streamable HTTP），A2A 解决"Agent ↔ Agent"的协作；而生产级 Agent 的胜负手是**上下文工程**（给模型看什么）与**可观测性 + 成本治理**。

> **关联阅读**：[[/docs/agent/agent-fundamentals.md]]、[[/docs/rag/retrieval-optimization-and-graphrag.md]]、[[/docs/llm/long-context-and-flashattention.md]]

---

## 1. MCP（Model Context Protocol）

### 1.1 解决什么问题

在 MCP 之前，每个 Agent 框架都自己定义工具接入方式 → **N 个模型 × M 个工具 = N×M 套适配**。MCP（Anthropic 2024 年 11 月开源）像"AI 应用的 USB-C"：定义统一协议，**工具方实现一次 Server，任何支持 MCP 的 Host 都能用**，把 N×M 降为 N+M。

### 1.2 架构与原语

```
Host（宿主应用，如 IDE / Claude Desktop / 自研 Agent）
 └── Client（每个 Server 一个连接，负责协议与权限）
      └── Server（暴露能力：文件系统 / 数据库 / Git / 浏览器 / 内部 API）
```

**三大原语（必背）**：

| 原语 | 谁来控制 | 用途 |
|------|----------|------|
| **Tools** | **模型控制**（model-controlled） | 可执行动作，模型自主决定调用（查询、写入、计算） |
| **Resources** | **应用控制**（app-controlled） | 只读上下文数据（文件、表结构、文档），由宿主决定塞给模型 |
| **Prompts** | **用户控制**（user-controlled） | 预置提示模板 / 工作流（如 slash command），用户主动触发 |

反向能力：**Sampling**（Server 请求 Host 帮它调用 LLM）、**Roots**（声明可访问的文件根）、**Elicitation**（Server 向用户追问缺失信息）。

### 1.3 传输与协议

- 消息格式：**JSON-RPC 2.0**；生命周期含 `initialize` 握手（协商协议版本与能力）→ 正常通信 → 关闭；
- 传输：**stdio**（本地进程，最常用、天然隔离）与 **Streamable HTTP**（2025 年 3 月引入，替代早期的 HTTP+SSE，支持单端点、可选流式、更好的可扩展性与断线重连）；
- 版本以日期命名（如 `2025-06-18`、`2026-01-15`），要注意 Host/Server 的版本兼容。

### 1.4 安全（面试加分点）

MCP 让模型能触达真实系统，风险显著：
- **提示注入 → 工具滥用**：文档/网页里的恶意指令诱导 Agent 调用危险工具（"混淆代理"问题）；
- **权限过宽**：Server 拿到过大的 token/文件权限；
- **供应链风险**：随便安装第三方 MCP Server 等于给它进程权限；
- **数据外泄**：Resources 把敏感数据塞进上下文再被外发。

**对策**：最小权限（scoped token、只读优先）、危险操作人工确认、工具白名单与来源审计、沙箱执行、输入/输出双向过滤、OAuth 授权（远程 Server）、完整审计日志。

> 面试高频：**MCP 和 Function Calling 是什么关系？** → 不冲突。Function Calling 是**模型侧能力**（输出结构化调用意图），MCP 是**系统侧协议**（工具如何被发现、描述、调用、授权）。MCP Server 暴露的 Tools 最终仍通过 Function Calling 被模型选择。

### 1.5 MCP vs A2A

| | **MCP** | **A2A**（Agent2Agent，Google 2025 提出） |
|--|---------|------------------------------------------|
| 连接对象 | Agent ↔ 工具/数据 | **Agent ↔ Agent** |
| 关注点 | 能力发现、调用、授权 | 身份/能力声明（Agent Card）、任务委派、长任务状态与协商 |
| 关系 | 互补：**A2A 负责"找谁做"，MCP 负责"用什么做"** | |

---

## 2. 多 Agent 系统

### 2.1 常见编排模式

| 模式 | 说明 | 适用 |
|------|------|------|
| **主-子（Orchestrator-Worker）** | 主 Agent 拆任务派给子 Agent，汇总结果 | 最常用，如"研究 Agent 派多个检索子 Agent" |
| 流水线（Sequential） | 固定顺序传递（撰写 → 审校 → 排版） | 流程确定 |
| 并行 + 聚合（Map-Reduce） | 同一任务多子 Agent 并行，投票/合并 | 提高召回与鲁棒性 |
| 辩论 / 评审 | 生成者 + 批评者互相挑战 | 质量优先、成本可接受 |
| 层级组织 | 多层管理者-执行者 | 超大任务，但延迟与成本剧增 |

### 2.2 多 Agent 的真实代价（面试很爱追问）

1. **上下文不共享**：子 Agent 各自的上下文难以完整传递 → 信息丢失、重复工作、结论冲突；
2. **成本与延迟成倍**：每个子 Agent 都要完整推理；
3. **错误传播**：上游子 Agent 出错，下游无法察觉；
4. **难调试**：轨迹交织，定位问题成本高；
5. **一致性**：多个 Agent 对同一事实给出不同答案时缺少裁决机制。

> **务实结论**：**优先单 Agent + 好工具 + 好上下文**；只有当任务天然可并行（多来源调研）、或需要角色隔离（生成/审查分离、权限隔离）时才上多 Agent。这是 2025–2026 年业界（含 Anthropic 的工程博客）反复强调的经验。

---

## 3. 上下文工程（Context Engineering）

**定义**：在每一步为模型精确准备"恰好够用"的上下文——比 prompt 工程更宽：包括**放什么、放多少、什么顺序、什么时候清理**。

### 3.1 四个操作原语

| 操作 | 手段 |
|------|------|
| **Write（写入）** | 系统提示、工具描述、任务目标、结构化 todo、外置状态文件 |
| **Select（选择）** | 检索相关记忆/文档/历史片段（RAG 化的记忆）、只加载当前需要的工具子集 |
| **Compress（压缩）** | 滚动摘要（auto-compact）、观察裁剪、大结果落盘只留引用/句柄 |
| **Isolate（隔离）** | 子 Agent 独立上下文、沙箱内运行代码只回传结论、分阶段清理 |

### 3.2 常见失效模式

| 失效 | 表现 | 对策 |
|------|------|------|
| **上下文中毒**（poisoning） | 早期错误结论被反复引用放大 | 阶段性校验与清理、关键结论要求带证据 |
| **上下文干扰**（distraction） | 无关内容太多，模型忽略关键指令 | 精简、rerank、把目标复述在末尾 |
| **上下文混淆**（confusion） | 工具/资源过多导致选错 | 工具分组与二级路由 |
| **上下文冲突**（clash） | 新旧信息矛盾 | 版本管理、显式标注时间与优先级 |
| **注意力稀释 / 任务漂移** | 长任务偏离原目标 | 结构化 todo 锚定、定期回读目标、分段验收 |
| Lost in the Middle | 中间证据被忽略 | 关键内容放首尾，见 [[/docs/llm/long-context-and-flashattention.md]] |

### 3.3 KV Cache 友好的上下文设计（工程加分）

- **前缀稳定**：把系统提示、工具定义放最前且**不要每轮变动**（别在开头塞时间戳），才能命中 prefix cache / RadixAttention，大幅降低 TTFT 与成本；
- **只追加不修改**：历史消息避免中途改写（一改就使后面全部 cache 失效）；
- 摘要压缩要**在固定切点**做，减少缓存抖动。

---

## 4. 可观测性与成本治理

**必须记录的 Trace 字段**：请求 id、每步的 thought/tool/args/observation、耗时、token 数与费用、模型版本、prompt 版本、检索命中、最终结果与用户反馈。

**核心指标**：
- 质量：任务成功率、人工介入率、幻觉/引用错误率、回归集通过率；
- 效率：平均步数、平均 token、单任务成本、p50/p95 延迟、TTFT；
- 稳定：工具失败率与重试率、超时率、循环中断率。

**成本优化手段（按性价比）**：
1. **模型路由/级联**：简单任务小模型，复杂/失败再升级到大模型（cascade），配置阈值与置信度判据；
2. **前缀缓存 + 语义缓存**；
3. 减少无效步数（更好的工具与提示 > 更强的模型）；
4. 观察裁剪、控制 top-k 与 rerank 候选数；
5. 批处理与并行工具调用降低延迟（延迟也是成本）；
6. 思考预算控制（推理模型按难度分档）。

---

## 5. 手撕/伪代码：上下文压缩与预算管理

```python
def build_context(system, tools, goal, history, docs, budget=16000, tokenizer=None):
    """稳定前缀 + 目标锚定 + 预算内择优填充，超预算则压缩历史"""
    n = lambda s: len(tokenizer.encode(s))
    parts = [("system", system), ("tools", tools)]      # 稳定前缀，利于 KV cache 命中
    used = sum(n(p[1]) for p in parts)

    fixed_tail = f"【当前目标】{goal}\n请只依据以上材料作答并给出引用。"
    reserve = n(fixed_tail) + 1500                      # 预留生成空间

    # 历史：保留最近 K 轮原文，更早的用摘要替代
    recent, older = history[-4:], history[:-4]
    if older:
        parts.append(("history_summary", summarize(older)))
    for h in recent:
        parts.append(("history", h))

    # 文档按 rerank 分数填入，超预算即停；最相关的放最后（贴近提问）
    for d in sorted(docs, key=lambda x: -x["score"]):
        chunk = f"[{d['id']}] {truncate(d['text'], 800)}"
        if used + n(chunk) + reserve > budget:
            break
        parts.append(("doc", chunk)); used += n(chunk)

    parts.append(("goal", fixed_tail))                  # 目标复述在末尾，抗漂移
    return parts
```

---

## 6. 面试高频问题速查

1. **MCP 解决什么问题？** → 统一"模型↔工具/数据"的接入协议，把 N×M 适配降为 N+M。
2. **MCP 的三层架构与三大原语？** → Host-Client-Server；Tools（模型控制）、Resources（应用控制）、Prompts（用户控制）。
3. **MCP 的传输方式？** → 本地 stdio 与远程 Streamable HTTP（替代早期 HTTP+SSE），消息用 JSON-RPC 2.0。
4. **MCP 与 Function Calling 的关系？** → 前者是系统侧协议，后者是模型侧能力，配合使用而非替代。
5. **MCP 有哪些安全风险？** → 提示注入诱导工具滥用、权限过宽、第三方 Server 供应链风险、数据外泄；用最小权限 + 人工确认 + 沙箱 + 审计。
6. **A2A 与 MCP 的区别？** → A2A 管 Agent 间协作（身份/能力/任务委派），MCP 管 Agent 与工具的连接；互补。
7. **多 Agent 一定比单 Agent 好吗？** → 不是。上下文不共享、成本延迟成倍、错误传播、难调试；优先单 Agent + 好工具。
8. **什么时候该上多 Agent？** → 任务天然可并行、需要角色/权限隔离、需要独立评审时。
9. **什么是上下文工程？** → 写入/选择/压缩/隔离四类操作，决定每步给模型看什么，是生产级 Agent 的核心。
10. **上下文有哪些失效模式？** → 中毒、干扰、混淆、冲突、任务漂移、Lost in the Middle。
11. **怎么让 Agent 的上下文对 KV Cache 友好？** → 前缀稳定、只追加不修改、固定切点压缩，命中 prefix cache 降 TTFT 与成本。
12. **Agent 成本怎么降？** → 模型路由/级联、前缀与语义缓存、减少无效步数、裁剪观察、并行调用、思考预算控制。
13. **Agent 要监控什么？** → 成功率、步数、token/成本、工具失败率、p95 延迟、人工介入率，并有失败回归集。

---

## 参考

- Anthropic, *Model Context Protocol* 官方规范与文档：<https://modelcontextprotocol.io>
- Google, *Agent2Agent (A2A) Protocol*, 2025
- Anthropic, *Building Effective Agents* / *Effective Context Engineering for AI Agents*
- Anthropic Engineering, *How we built our multi-agent research system*（多 Agent 的代价与适用边界）
- 卡码笔记《2026 年 Agent 大厂面试题汇总》《Agent Harness 可观测性》
