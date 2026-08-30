"""Campus Agent 的稳定公开入口。

从 P0 开始，原有 ``CampusAgent`` 接口内部由 LangGraph 状态图驱动。这样
CLI、Web 和已有测试无需更改，同时所有生产请求都会经过同一张图。
"""

from __future__ import annotations

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver

from campus_agent.graph.workflow import CampusGraphAgent
from campus_agent.memory import ConversationMemory
from campus_agent.planner import Planner, RuleBasedPlanner
from campus_agent.tooling import Tool, ToolRegistry
from campus_agent.tools import CampusKnowledgeTool, CourseQueryTool


class CampusAgent(CampusGraphAgent):
    """兼容旧名称的 LangGraph Agent。"""


def build_default_agent(
    planner: Planner | None = None,
    memory: ConversationMemory | None = None,
    knowledge_tool: CampusKnowledgeTool | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    extra_tools: Sequence[Tool] = (),
) -> CampusAgent:
    registry = ToolRegistry()
    registry.register(CourseQueryTool())
    registry.register(knowledge_tool or CampusKnowledgeTool())
    for tool in extra_tools:
        registry.register(tool)
    return CampusAgent(
        planner=planner or RuleBasedPlanner(),
        tools=registry,
        memory=memory,
        checkpointer=checkpointer,
    )
