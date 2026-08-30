"""LangGraph 节点之间传递的可序列化状态。"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langgraph.channels import UntrackedValue


class CampusGraphState(TypedDict):
    # 只有 history 是持久状态。其余字段都是一次回合的工作区：即使节点异常
    # 或 SSE 被客户端中断，也不会把问题、工具输出或知识正文写入 checkpoint。
    query: Annotated[str, UntrackedValue]
    session_id: Annotated[str, UntrackedValue]
    planner_mode: Annotated[str, UntrackedValue]
    history: list[dict[str, str]]
    events: Annotated[list[dict[str, str]], UntrackedValue]
    plan_summary: NotRequired[Annotated[str, UntrackedValue]]
    direct_answer: NotRequired[Annotated[str | None, UntrackedValue]]
    tool_calls: NotRequired[Annotated[list[dict[str, object]], UntrackedValue]]
    tool_results: NotRequired[Annotated[list[dict[str, object]], UntrackedValue]]
    citations: NotRequired[Annotated[list[dict[str, object]], UntrackedValue]]
    retrieval_filters: NotRequired[Annotated[dict[str, str], UntrackedValue]]
    retrieved_documents: NotRequired[
        Annotated[list[dict[str, object]], UntrackedValue]
    ]
    active_retrieval_query: NotRequired[Annotated[str, UntrackedValue]]
    rewritten_query: NotRequired[Annotated[str, UntrackedValue]]
    rewrite_changed: NotRequired[Annotated[bool, UntrackedValue]]
    retrieval_attempts: NotRequired[Annotated[int, UntrackedValue]]
    retrieval_grade: NotRequired[Annotated[str, UntrackedValue]]
    retrieval_reason: NotRequired[Annotated[str, UntrackedValue]]
    verification_status: NotRequired[Annotated[str, UntrackedValue]]
    verification_reason: NotRequired[Annotated[str, UntrackedValue]]
    grounded: NotRequired[Annotated[bool, UntrackedValue]]
    answer: NotRequired[Annotated[str, UntrackedValue]]
    errors: NotRequired[Annotated[list[str], UntrackedValue]]
