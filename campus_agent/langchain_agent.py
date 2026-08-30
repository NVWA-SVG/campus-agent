"""基于 LangChain v1 `create_agent` 的 Campus Agent 集成。

业务逻辑仍复用项目原有工具，LangChain 负责编排模型—工具循环。
调用方可以传入任意支持工具调用的 LangChain ChatModel。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel

from campus_agent.tools import CampusKnowledgeTool, CourseQueryTool


_course_tool = CourseQueryTool()
_knowledge_tool = CampusKnowledgeTool()


@tool("query_courses")
def query_courses_tool(weekday: str, keyword: str = "") -> str:
    """按星期查询课程，可使用课程名称关键字进一步筛选。

    Args:
        weekday: 要查询的星期，例如“周三”或“Wednesday”。
        keyword: 可选的课程名关键字；不需要筛选时传空字符串。
    """
    return _course_tool.run(weekday=weekday, keyword=keyword)


@tool("search_campus_knowledge")
def search_campus_knowledge_tool(query: str) -> str:
    """查询图书馆、成绩单、校园卡和奖学金等校园办事流程。

    Args:
        query: 用户提出的完整校园事务问题。
    """
    return _knowledge_tool.run(query=query)


LANGCHAIN_TOOLS = (query_courses_tool, search_campus_knowledge_tool)


SYSTEM_PROMPT = """你是校园事务智能助手。
只回答工具能力覆盖的校园课程和办事问题；需要事实时必须调用工具，不得猜测。
工具报错或没有结果时，应明确说明，不能编造课程、地点或办事流程。
回答使用简洁中文，并保留知识工具返回的来源。
"""


def build_langchain_agent(model: BaseChatModel) -> Any:
    """使用调用方提供的 ChatModel 创建 LangChain Agent。"""
    return create_agent(
        model=model,
        tools=list(LANGCHAIN_TOOLS),
        system_prompt=SYSTEM_PROMPT,
        name="campus_assistant",
    )


def final_text(result: dict[str, Any]) -> str:
    """从 `create_agent().invoke()` 的状态中取得最终文本。"""
    messages = result.get("messages", [])
    if not messages:
        raise ValueError("LangChain Agent 没有返回消息")
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return str(content)

