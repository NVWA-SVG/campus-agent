"""Agent 各模块之间传递的结构化数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    document_id: str
    chunk_id: str
    source: str
    title: str
    score: float
    retrieval_method: str
    snippet: str = ""

    def as_dict(self) -> dict[str, object]:
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
    text: str
    citations: tuple[Citation, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    output: str = ""
    error: str | None = None
    citations: tuple[Citation, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Plan:
    summary: str
    tool_calls: tuple[ToolCall, ...] = ()
    direct_answer: str | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
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
    answer: str
    events: tuple[AgentEvent, ...]
    citations: tuple[Citation, ...] = ()
