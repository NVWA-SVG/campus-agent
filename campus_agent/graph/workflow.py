"""编译并运行 Campus Agent 的 LangGraph 状态图。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from campus_agent.domain import AgentEvent, AgentResponse, Citation, Message
from campus_agent.graph.nodes import (
    build_commit_turn_node,
    build_compose_node,
    build_execute_tools_node,
    build_grade_documents_node,
    build_plan_node,
    build_retry_knowledge_node,
    build_rewrite_query_node,
    build_verify_answer_node,
    direct_answer_node,
    route_after_grade,
    route_after_plan,
    route_after_rewrite,
    route_after_tools,
)
from campus_agent.graph.state import CampusGraphState
from campus_agent.memory import ConversationMemory
from campus_agent.planner import Planner, RuleBasedPlanner
from campus_agent.rag.quality import (
    OfflineAnswerVerifier,
    OfflineDocumentGrader,
    OfflineQueryRewriter,
)
from campus_agent.tooling import Tool, ToolRegistry
from campus_agent.tools import CampusKnowledgeTool, CourseQueryTool


class CampusGraphAgent:
    """与 CampusAgent 接口兼容的 LangGraph 运行时。"""

    def __init__(
        self,
        planner: Planner,
        tools: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_tool_calls: int = 4,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须至少为 1")
        self._planner = planner
        self._tools = tools
        self._checkpointer = checkpointer
        self._memory = None if checkpointer is not None else (memory or ConversationMemory())
        self._max_tool_calls = max_tool_calls
        self._graph = self._compile_graph()

    def _compile_graph(self):
        workflow = StateGraph(CampusGraphState)
        workflow.add_node("plan", build_plan_node(self._planner, self._tools))
        workflow.add_node("direct_answer", direct_answer_node)
        workflow.add_node(
            "execute_tools",
            build_execute_tools_node(
                self._tools,
                max_tool_calls=self._max_tool_calls,
            ),
        )
        workflow.add_node("compose", build_compose_node(self._planner))
        workflow.add_node(
            "grade_documents",
            build_grade_documents_node(OfflineDocumentGrader()),
        )
        workflow.add_node(
            "rewrite_query",
            build_rewrite_query_node(OfflineQueryRewriter()),
        )
        workflow.add_node(
            "retry_knowledge",
            build_retry_knowledge_node(self._tools),
        )
        workflow.add_node(
            "verify_answer",
            build_verify_answer_node(
                OfflineAnswerVerifier(),
                RuleBasedPlanner(),
            ),
        )
        workflow.add_node("commit_turn", build_commit_turn_node(self._memory))
        workflow.add_edge(START, "plan")
        workflow.add_conditional_edges(
            "plan",
            route_after_plan,
            {
                "direct_answer": "direct_answer",
                "execute_tools": "execute_tools",
            },
        )
        workflow.add_edge("direct_answer", "commit_turn")
        workflow.add_conditional_edges(
            "execute_tools",
            route_after_tools,
            {
                "grade_documents": "grade_documents",
                "compose": "compose",
            },
        )
        workflow.add_conditional_edges(
            "grade_documents",
            route_after_grade,
            {
                "compose": "compose",
                "rewrite_query": "rewrite_query",
            },
        )
        workflow.add_conditional_edges(
            "rewrite_query",
            route_after_rewrite,
            {
                "retry_knowledge": "retry_knowledge",
                "compose": "compose",
            },
        )
        workflow.add_edge("retry_knowledge", "grade_documents")
        workflow.add_edge("compose", "verify_answer")
        workflow.add_edge("verify_answer", "commit_turn")
        workflow.add_edge("commit_turn", END)
        return workflow.compile(checkpointer=self._checkpointer)

    def ask(
        self,
        query: str,
        session_id: str = "default",
        retrieval_filters: dict[str, str] | None = None,
    ) -> AgentResponse:
        input_state = self._input_state(query, session_id, retrieval_filters)
        invoke_options = {"durability": "exit"} if self._checkpointer else {}
        rollback_supported, before_checkpoint = self._checkpoint_marker(session_id)
        try:
            result = self._graph.invoke(
                input_state,
                config=self._thread_config(session_id),
                **invoke_options,
            )
            response = self._response_from_state(result)
            self._prune_checkpoint(session_id)
            return response
        except BaseException:
            if rollback_supported:
                self._rollback_checkpoint(session_id, before_checkpoint)
            raise

    def stream(
        self,
        query: str,
        session_id: str = "default",
        retrieval_filters: dict[str, str] | None = None,
    ) -> Iterator[dict[str, object]]:
        """逐节点输出白名单 Trace，最后输出完整 AgentResponse。"""

        input_state = self._input_state(query, session_id, retrieval_filters)
        final_state = dict(input_state)
        emitted_events = 0
        stream_options = {"durability": "exit"} if self._checkpointer else {}
        rollback_supported, before_checkpoint = self._checkpoint_marker(session_id)
        committed = False
        try:
            for update in self._graph.stream(
                input_state,
                config=self._thread_config(session_id),
                stream_mode="updates",
                **stream_options,
            ):
                if not isinstance(update, dict):
                    continue
                for payload in update.values():
                    if not isinstance(payload, dict):
                        continue
                    final_state.update(payload)
                    raw_events = payload.get("events")
                    if not isinstance(raw_events, list):
                        continue
                    for event in raw_events[emitted_events:]:
                        if not isinstance(event, dict):
                            continue
                        agent_event = AgentEvent(
                            event_type=event["type"],
                            detail=str(event.get("detail", "")),
                        )
                        yield {"kind": "trace", "event": agent_event}
                        emitted_events += 1
            response = self._response_from_state(final_state)
            self._prune_checkpoint(session_id)
            committed = True
            yield {"kind": "result", "response": response}
        finally:
            if not committed and rollback_supported:
                self._rollback_checkpoint(session_id, before_checkpoint)

    def _response_from_state(self, result: dict[str, object]) -> AgentResponse:
        answer = str(result["answer"])
        return AgentResponse(
            answer=answer,
            events=tuple(
                AgentEvent(
                    event_type=event["type"],
                    detail=event["detail"],
                )
                for event in result.get("events", [])
            ),
            citations=tuple(
                Citation(
                    citation_id=str(citation["citation_id"]),
                    document_id=str(citation["document_id"]),
                    chunk_id=str(citation["chunk_id"]),
                    source=str(citation["source"]),
                    title=str(citation["title"]),
                    score=float(citation["score"]),
                    retrieval_method=str(citation["retrieval_method"]),
                    snippet=str(citation.get("snippet", "")),
                )
                for citation in result.get("citations", [])
            ),
        )

    def history(self, session_id: str = "default") -> tuple[Message, ...]:
        if self._checkpointer is None:
            assert self._memory is not None
            return self._memory.get(session_id)
        snapshot = self._graph.get_state(self._thread_config(session_id))
        # SQLite checkpoint 保存的是 MessagePayload；公开接口始终返回统一的
        # Message 领域对象，使调用方无需关心底层使用内存还是 SQLite。
        return tuple(
            Message.from_dict(message)
            for message in snapshot.values.get("history", [])
        )

    def clear_history(self, session_id: str = "default") -> None:
        if self._checkpointer is None:
            assert self._memory is not None
            self._memory.clear(session_id)
            return
        self._checkpointer.delete_thread(self._thread_id(session_id))

    @property
    def checkpoint_enabled(self) -> bool:
        return self._checkpointer is not None

    def planner_metrics(self) -> dict[str, object]:
        metrics = getattr(self._planner, "metrics", None)
        if callable(metrics):
            return metrics()
        return {"planner": "rule", "network_calls": 0}

    def _planner_name(self) -> str:
        metrics = self.planner_metrics()
        return str(metrics.get("planner", "rule"))

    def _input_state(
        self,
        query: str,
        session_id: str,
        retrieval_filters: dict[str, str] | None,
    ) -> dict[str, object]:
        state: dict[str, object] = {
            "query": query,
            "session_id": session_id,
            "planner_mode": self._planner_name(),
            "events": [],
            "errors": [],
            "plan_summary": "",
            "direct_answer": None,
            "tool_calls": [],
            "tool_results": [],
            "answer": "",
            "citations": [],
            "retrieval_filters": dict(retrieval_filters or {}),
            "retrieved_documents": [],
            "active_retrieval_query": "",
            "rewritten_query": "",
            "rewrite_changed": False,
            "retrieval_attempts": 0,
            "retrieval_grade": "",
            "retrieval_reason": "",
            "verification_status": "",
            "verification_reason": "",
            "grounded": False,
        }
        if self._checkpointer is None:
            assert self._memory is not None
            # 没有 checkpointer 时，先把内存中的领域对象转成和 SQLite 路径
            # 完全相同的 Graph State 结构，避免两种运行模式产生不同语义。
            state["history"] = [
                message.as_dict()
                for message in self._memory.get(session_id)
            ]
        return state

    @staticmethod
    def _thread_id(session_id: str) -> str:
        return f"campus:{session_id}"

    def _thread_config(self, session_id: str) -> dict[str, object]:
        return {"configurable": {"thread_id": self._thread_id(session_id)}}

    def _checkpoint_marker(self, session_id: str) -> tuple[bool, str | None]:
        latest_id = getattr(self._checkpointer, "latest_id", None)
        if not callable(latest_id):
            return False, None
        return True, latest_id(self._thread_id(session_id))

    def _rollback_checkpoint(
        self,
        session_id: str,
        keep_id: str | None,
    ) -> None:
        rollback_after = getattr(self._checkpointer, "rollback_after", None)
        if callable(rollback_after):
            rollback_after(self._thread_id(session_id), keep_id)

    def _prune_checkpoint(self, session_id: str) -> None:
        prune_thread = getattr(self._checkpointer, "prune_thread", None)
        if callable(prune_thread):
            prune_thread(self._thread_id(session_id), keep=1)


def build_graph_agent(
    planner: Planner | None = None,
    memory: ConversationMemory | None = None,
    knowledge_tool: CampusKnowledgeTool | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    extra_tools: Sequence[Tool] = (),
) -> CampusGraphAgent:
    registry = ToolRegistry()
    registry.register(CourseQueryTool())
    registry.register(knowledge_tool or CampusKnowledgeTool())
    for tool in extra_tools:
        registry.register(tool)
    return CampusGraphAgent(
        planner=planner or RuleBasedPlanner(),
        tools=registry,
        memory=memory,
        checkpointer=checkpointer,
    )
