"""Agent 各模块之间传递的结构化数据。

这里的类只描述“数据长什么样”，不执行规划、工具或模型调用。把这些稳定的
领域对象集中在一个模块里，可以避免 Planner、LangGraph、Web 和存储层各自
发明一套不兼容的数据格式。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast


# 会话历史只保存真实的用户/助手对话。系统提示词和工具结果分别由 Prompt 与
# ToolResult 管理，不能为了模仿某个模型 API 而混入持久化历史。
ConversationRole = Literal["user", "assistant"]
# 保留旧名称，避免外部调用方因领域名称收紧而立刻中断。
Role = ConversationRole


class MessagePayload(TypedDict):
    """适合写入 LangGraph State、SQLite 和 JSON 的消息字典结构。

    ``TypedDict`` 仍是普通 ``dict``，只为 IDE/类型检查器声明固定键和值类型；
    它不会像 ``Message`` 一样在运行时主动校验数据。
    """

    role: ConversationRole
    content: str


# dataclass 自动生成 __init__/__repr__/__eq__；frozen 防止字段被重新赋值，
# slots 固定字段集合并避免为每个短消息创建普通 __dict__。
@dataclass(frozen=True, slots=True)
class Message:
    """只表示需要跨回合保存的用户/助手对话消息。"""

    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        """补上类型注解不会提供的运行时边界校验。"""

        if self.role not in {"user", "assistant"}:
            raise ValueError(f"不支持的消息角色：{self.role}")
        if not isinstance(self.content, str):
            raise TypeError("消息内容必须是字符串")

    def as_dict(self) -> MessagePayload:
        """转换为可安全写入 Graph State、SQLite 和 JSON 的基础类型。"""

        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Message:
        """在存储/Graph 字典重新进入领域层时校验并恢复消息对象。"""

        role = value.get("role")
        content = value.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"不支持的消息角色：{role}")
        if not isinstance(content, str):
            raise TypeError("消息内容必须是字符串")
        return cls(
            role=cast(ConversationRole, role),
            content=content,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Planner 生成的一次待执行工具调用。"""

    call_id: str
    name: str
    # default_factory 确保每个 ToolCall 得到独立字典，避免共享可变默认值。
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    """一条可展示、可追溯的 RAG 文档引用。"""

    citation_id: str
    document_id: str
    chunk_id: str
    source: str
    title: str
    score: float
    retrieval_method: str
    snippet: str = ""

    def as_dict(self) -> dict[str, object]:
        """转换为 Graph State 和 Web API 可以直接序列化的字典。"""

        return {
            "citation_id": self.citation_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "snippet": self.snippet,
        }


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """工具自身成功执行后产生的业务输出。

    ``text`` 给模型/用户阅读，``citations`` 与 ``data`` 给后续 Grade、Verify 和
    Web 来源卡片继续使用。
    """

    text: str
    citations: tuple[Citation, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """ToolRegistry 为成功或失败的工具调用生成的统一结果。"""

    call_id: str
    tool_name: str
    ok: bool
    output: str = ""
    error: str | None = None
    citations: tuple[Citation, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Plan:
    """Planner 的结构化决策：调用工具，或给出直接回答。"""

    summary: str
    tool_calls: tuple[ToolCall, ...] = ()
    direct_answer: str | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """允许 CLI/Web 对外展示的白名单执行轨迹。"""

    event_type: Literal[
        "plan",
        "retrieve",
        "tool_result",
        "grade",
        "rewrite",
        "verify",
        "final",
    ]
    detail: str


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """一次 Agent 回合最终交给调用方的稳定返回结构。"""

    answer: str
    events: tuple[AgentEvent, ...]
    citations: tuple[Citation, ...] = ()
