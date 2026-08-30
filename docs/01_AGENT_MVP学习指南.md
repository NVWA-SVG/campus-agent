# 第 1 课：从可运行项目理解 Agent

## 本课目标

本课不从语法表开始，而是从已经能运行的 Campus Agent 反向学习。完成后应能解释：

1. Agent 与普通函数调用的区别。
2. 规划器、工具、记忆和主循环分别负责什么。
3. `Plan → Retrieve → Grade → Rewrite → Verify → Commit` 如何流动。
4. 为什么工具错误要在 Agent 边界被捕获，以及为什么完整回合最后才提交。

## 第一步：观察 Agent 行为

在项目目录运行：

```powershell
python -m campus_agent --trace
```

依次输入：

```text
周三有什么课程？
在哪里上？
告诉我周一的课程和校园卡补办流程
/history
/quit
```

观察三件事：

- 第一个问题触发 `query_courses`。
- “在哪里上”没有写星期，但 Agent 从会话记忆中取得“周三”。
- 第三个问题在一个计划中调用两个工具。

## 第二步：沿数据流阅读代码

### 1. 入口

从 `campus_agent/cli.py` 的 `main()` 开始。它读取输入，然后调用：

```python
response = agent.ask(query, session_id=session_id)
```

CLI 只负责输入输出，不应该包含课程查询规则。

### 2. 主状态图

`campus_agent/agent.py` 现在是兼容入口。阅读 `campus_agent/graph/workflow.py` 和 `campus_agent/graph/nodes.py`：

```text
读取历史
→ 让 Planner 生成 Plan
→ 通过 ToolRegistry 执行 ToolCall
→ 如果检索不足，则保守改写并最多重试一次
→ Planner 整理 ToolResult
→ Verifier 检查引用和关键事实
→ 最后一次性保存 user + assistant
```

这是整个项目最核心的代码。任何中间节点失败都不会写入半个回合。

### 3. 规划

阅读 `campus_agent/planner.py`。当前使用规则识别意图和参数，因此无须联网。以后真实 LLM 也必须返回相同的 `Plan` 和 `ToolCall`，这样主循环不需要重写。

### 4. 工具

阅读 `campus_agent/tooling.py` 和 `campus_agent/tools/`：

- `Tool` 定义所有工具共同遵守的接口。
- `ToolRegistry` 负责查找和执行工具。
- 具体工具只负责一个明确的业务能力。

工具不能直接读取 CLI，也不应该自行保存会话记忆。

### 5. 记忆与 Checkpoint

CLI 阅读 `campus_agent/memory.py`；网页继续阅读 `campus_agent/checkpoint.py` 与 `campus_agent/web.py`。不同 `session_id` 映射为不同 LangGraph thread：逻辑历史最多 20 条消息，checkpoint 表中每个 thread 只保留一个最新快照，并限制最近 256 个会话；空闲时还会维护 WAL。除 `history` 外的回合工作区都是 `UntrackedValue`；异常或断流会回滚。Web 服务重启后历史仍能恢复，`/api/history` 的 DELETE 会删除对应 thread。

## 第三步：用测试理解需求

运行：

```powershell
python -m unittest discover -s tests -v
```

重点阅读以下测试：

- `test_agent_can_call_multiple_tools`
- `test_follow_up_uses_conversation_memory`
- `test_invalid_arguments_are_contained`
- `test_history_is_separated_by_session`

测试名称就是系统应当满足的行为。修改代码后必须重新运行全部测试。

## 本课动手任务

不要立即增加框架。先完成下面的项目修改：

1. 在课程工具中增加一门“数据结构”课程，安排在周二 14:30、教四-301。
2. 在 `tests/test_agent.py` 增加测试，确认“周二有什么课”能查到它。
3. 在知识工具中增加“实验室预约流程”。
4. 让规划器能识别“实验室”和“预约”。
5. 增加一个 Agent 测试，确认用户询问实验室预约时会调用知识工具。
6. 运行全部测试，确保原有测试和新增测试都通过。

## 验收问题

1. `RuleBasedPlanner` 为什么应该依赖工具描述，而不是直接调用工具对象？
2. `ToolRegistry.execute()` 为什么捕获异常后返回 `ToolResult`，而不是让程序退出？
3. `session_id` 解决了什么问题？
4. 如果真实 LLM 生成了不存在的工具名，当前系统会怎样处理？
5. 当前规则规划器与真实 LLM 规划器之间，哪些模块可以复用？

完成动手任务并回答问题后，阅读 `02_P0_P4架构与面试讲解.md`，再分别跟踪 Hybrid RAG、Agentic RAG、Checkpoint 和 SSE。
