"""复用现有 Planner 与 ToolRegistry 的 LangGraph 节点。"""

from __future__ import annotations

from collections.abc import Callable

from campus_agent.domain import Citation, Message, ToolCall, ToolResult
from campus_agent.graph.state import CampusGraphState
from campus_agent.memory import ConversationMemory
from campus_agent.planner import Planner
from campus_agent.rag.quality import AnswerVerifier, DocumentGrader, QueryRewriter
from campus_agent.tooling import ToolRegistry


KNOWLEDGE_TOOL_NAME = "search_campus_knowledge"


def _history_messages(state: CampusGraphState) -> tuple[Message, ...]:
    return tuple(
        Message(role=message["role"], content=message["content"])
        for message in state.get("history", [])
    )


def _append_event(
    state: CampusGraphState,
    event_type: str,
    detail: str,
) -> list[dict[str, str]]:
    return [*state.get("events", []), {"type": event_type, "detail": detail}]


def _citation_from_dict(value: dict[str, object]) -> Citation:
    return Citation(
        citation_id=str(value["citation_id"]),
        document_id=str(value["document_id"]),
        chunk_id=str(value["chunk_id"]),
        source=str(value["source"]),
        title=str(value["title"]),
        score=float(value["score"]),
        retrieval_method=str(value["retrieval_method"]),
        snippet=str(value.get("snippet", "")),
    )


def _normalize_citations(
    citations: list[dict[str, object]],
) -> list[dict[str, object]]:
    """按 chunk 去重并生成当前回答内唯一、连续的 Citation ID。"""

    normalized: list[dict[str, object]] = []
    seen_chunks: set[str] = set()
    for citation in citations:
        chunk_id = str(citation.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        normalized.append(
            {
                **citation,
                "citation_id": f"C{len(normalized) + 1}",
            }
        )
    return normalized


def _serialize_tool_result(result: ToolResult) -> dict[str, object]:
    return {
        "call_id": result.call_id,
        "tool_name": result.tool_name,
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
        "citations": [citation.as_dict() for citation in result.citations],
        "data": dict(result.data),
    }


def _tool_result_from_dict(result: dict[str, object]) -> ToolResult:
    return ToolResult(
        call_id=str(result["call_id"]),
        tool_name=str(result["tool_name"]),
        ok=bool(result["ok"]),
        output=str(result.get("output", "")),
        error=(str(result["error"]) if result.get("error") is not None else None),
        citations=tuple(
            _citation_from_dict(citation)
            for citation in result.get("citations", [])
            if isinstance(citation, dict)
        ),
        data=dict(result.get("data", {})),
    )


def _call_from_raw(
    state: CampusGraphState,
    raw_call: dict[str, object],
    *,
    query_override: str | None = None,
) -> ToolCall:
    call_name = str(raw_call["name"])
    arguments = dict(raw_call.get("arguments", {}))
    if query_override is not None and call_name == KNOWLEDGE_TOOL_NAME:
        arguments["query"] = query_override
    if call_name == KNOWLEDGE_TOOL_NAME and state.get("retrieval_filters"):
        arguments["filters"] = dict(state["retrieval_filters"])
    return ToolCall(
        call_id=str(raw_call["call_id"]),
        name=call_name,
        arguments=arguments,
    )


def _tool_event(call: ToolCall, result: ToolResult) -> dict[str, str]:
    status = "成功" if result.ok else "失败"
    return {
        "type": "tool_result",
        "detail": f"{call.name} {status}；参数={call.arguments}",
    }


def build_plan_node(planner: Planner, tools: ToolRegistry) -> Callable:
    def plan_node(state: CampusGraphState) -> dict[str, object]:
        plan = planner.plan(
            state["query"],
            _history_messages(state),
            tools.descriptions(),
        )
        if sum(call.name == KNOWLEDGE_TOOL_NAME for call in plan.tool_calls) > 1:
            # 当前 Agentic-RAG 状态机按一次知识任务进行 Grade/Rewrite/Retry。
            # 在图边界再次校验，避免自定义 Planner 绕过 DeepSeek 的本地校验。
            raise ValueError("单轮最多允许一次知识检索，请合并为一个完整查询")
        return {
            "plan_summary": plan.summary,
            "direct_answer": plan.direct_answer,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                }
                for call in plan.tool_calls
            ],
            "events": _append_event(state, "plan", plan.summary),
        }

    return plan_node


def route_after_plan(state: CampusGraphState) -> str:
    if state.get("direct_answer") is not None:
        return "direct_answer"
    return "execute_tools"


def direct_answer_node(state: CampusGraphState) -> dict[str, object]:
    direct_answer = state.get("direct_answer")
    return {"answer": "" if direct_answer is None else str(direct_answer)}


def build_execute_tools_node(
    tools: ToolRegistry,
    *,
    max_tool_calls: int,
) -> Callable:
    def execute_tools_node(state: CampusGraphState) -> dict[str, object]:
        events = list(state.get("events", []))
        serialized_results: list[dict[str, object]] = []
        citations: list[dict[str, object]] = []
        retrieved_documents: list[dict[str, object]] = []
        active_retrieval_query = ""
        retrieval_attempts = 0

        for raw_call in state.get("tool_calls", [])[:max_tool_calls]:
            call = _call_from_raw(state, raw_call)
            if call.name == KNOWLEDGE_TOOL_NAME:
                active_retrieval_query = str(call.arguments.get("query", state["query"]))
                retrieval_attempts = 1
                events.append(
                    {
                        "type": "retrieve",
                        "detail": f"第1次检索：{active_retrieval_query}",
                    }
                )
            result = tools.execute(call)
            serialized_results.append(_serialize_tool_result(result))
            citations.extend(citation.as_dict() for citation in result.citations)
            if call.name == KNOWLEDGE_TOOL_NAME:
                raw_hits = result.data.get("hits", [])
                if isinstance(raw_hits, list):
                    retrieved_documents.extend(
                        hit for hit in raw_hits if isinstance(hit, dict)
                    )
            events.append(_tool_event(call, result))

        return {
            "tool_results": serialized_results,
            "citations": _normalize_citations(citations),
            "retrieved_documents": retrieved_documents,
            "active_retrieval_query": active_retrieval_query,
            "retrieval_attempts": retrieval_attempts,
            "events": events,
        }

    return execute_tools_node


def route_after_tools(state: CampusGraphState) -> str:
    return "grade_documents" if state.get("retrieval_attempts", 0) else "compose"


def build_grade_documents_node(grader: DocumentGrader) -> Callable:
    def grade_documents_node(state: CampusGraphState) -> dict[str, object]:
        result = grader.grade(
            state.get("active_retrieval_query") or state["query"],
            list(state.get("retrieved_documents", [])),
        )
        return {
            "retrieval_grade": result.status,
            "retrieval_reason": result.reason,
            "events": _append_event(
                state,
                "grade",
                f"{result.status}：{result.reason}",
            ),
        }

    return grade_documents_node


def route_after_grade(state: CampusGraphState) -> str:
    if state.get("retrieval_grade") == "relevant":
        return "compose"
    if int(state.get("retrieval_attempts", 0)) < 2:
        return "rewrite_query"
    return "compose"


def build_rewrite_query_node(rewriter: QueryRewriter) -> Callable:
    def rewrite_query_node(state: CampusGraphState) -> dict[str, object]:
        active_query = state.get("active_retrieval_query") or state["query"]
        result = rewriter.rewrite(active_query)
        return {
            "rewritten_query": result.query,
            "rewrite_changed": result.changed,
            "events": _append_event(
                state,
                "rewrite",
                (
                    f"改写为：{result.query}（{result.reason}）"
                    if result.changed
                    else f"停止改写：{result.reason}"
                ),
            ),
        }

    return rewrite_query_node


def route_after_rewrite(state: CampusGraphState) -> str:
    return "retry_knowledge" if state.get("rewrite_changed") else "compose"


def build_retry_knowledge_node(tools: ToolRegistry) -> Callable:
    def retry_knowledge_node(state: CampusGraphState) -> dict[str, object]:
        raw_knowledge_call = next(
            (
                raw_call
                for raw_call in state.get("tool_calls", [])
                if raw_call.get("name") == KNOWLEDGE_TOOL_NAME
            ),
            None,
        )
        if raw_knowledge_call is None:
            return {
                "retrieval_attempts": 2,
                "retrieved_documents": [],
            }

        rewritten_query = str(state.get("rewritten_query") or state["query"])
        call = _call_from_raw(
            state,
            raw_knowledge_call,
            query_override=rewritten_query,
        )
        result = tools.execute(call)
        serialized = _serialize_tool_result(result)
        replaced = False
        tool_results: list[dict[str, object]] = []
        for existing in state.get("tool_results", []):
            if existing.get("tool_name") == KNOWLEDGE_TOOL_NAME:
                if not replaced:
                    tool_results.append(serialized)
                    replaced = True
                continue
            tool_results.append(existing)
        if not replaced:
            tool_results.append(serialized)

        non_knowledge_citations = [
            citation
            for existing in state.get("tool_results", [])
            if existing.get("tool_name") != KNOWLEDGE_TOOL_NAME
            for citation in existing.get("citations", [])
            if isinstance(citation, dict)
        ]
        citations = [
            *non_knowledge_citations,
            *(citation.as_dict() for citation in result.citations),
        ]
        raw_hits = result.data.get("hits", [])
        retrieved_documents = (
            [hit for hit in raw_hits if isinstance(hit, dict)]
            if isinstance(raw_hits, list)
            else []
        )
        attempt = int(state.get("retrieval_attempts", 1)) + 1
        events = [
            *state.get("events", []),
            {"type": "retrieve", "detail": f"第{attempt}次检索：{rewritten_query}"},
            _tool_event(call, result),
        ]
        return {
            "tool_results": tool_results,
            "citations": _normalize_citations(citations),
            "retrieved_documents": retrieved_documents,
            "active_retrieval_query": rewritten_query,
            "retrieval_attempts": attempt,
            "events": events,
        }

    return retry_knowledge_node


def build_compose_node(planner: Planner) -> Callable:
    def compose_node(state: CampusGraphState) -> dict[str, object]:
        serialized_results = list(state.get("tool_results", []))
        citations = list(state.get("citations", []))
        retrieved_documents = list(state.get("retrieved_documents", []))
        if state.get("retrieval_grade") == "insufficient":
            safe_output = (
                "知识库中暂未找到与问题足够相关的信息，请联系对应业务部门确认。"
                "我目前可以查询课程和已经收录的校园办事资料。"
            )
            serialized_results = [
                (
                    {
                        **result,
                        "output": safe_output,
                        "citations": [],
                        "data": {
                            "query": state.get("active_retrieval_query")
                            or state["query"],
                            "hits": [],
                        },
                    }
                    if result.get("tool_name") == KNOWLEDGE_TOOL_NAME
                    else result
                )
                for result in serialized_results
            ]
            # 相关性门控拒绝的候选不能继续作为引用或 Verifier 证据。
            citations = []
            retrieved_documents = []
        results = tuple(
            _tool_result_from_dict(result)
            for result in serialized_results
        )
        return {
            "answer": planner.compose(state["query"], results),
            "tool_results": serialized_results,
            "citations": citations,
            "retrieved_documents": retrieved_documents,
        }

    return compose_node


def build_verify_answer_node(
    verifier: AnswerVerifier,
    fallback_planner: Planner,
) -> Callable:
    def verify_answer_node(state: CampusGraphState) -> dict[str, object]:
        serialized_results = list(state.get("tool_results", []))
        result = verifier.verify(
            str(state.get("answer", "")),
            serialized_results,
            list(state.get("citations", [])),
        )
        answer = str(state.get("answer", ""))
        detail = f"{result.status}：{result.reason}"
        if result.status in {"unsupported", "insufficient_evidence"}:
            answer = fallback_planner.compose(
                state["query"],
                tuple(_tool_result_from_dict(item) for item in serialized_results),
            )
            detail += "；已回退到可验证的工具原文"
        return {
            "answer": answer,
            "verification_status": result.status,
            "verification_reason": result.reason,
            "grounded": result.status == "grounded",
            "events": _append_event(state, "verify", detail),
        }

    return verify_answer_node


def build_commit_turn_node(
    memory: ConversationMemory | None,
    *,
    max_history_messages: int = 20,
) -> Callable:
    """只在整个回合成功后原子写入两条消息。"""

    def commit_turn_node(state: CampusGraphState) -> dict[str, object]:
        answer = str(state.get("answer", ""))
        user_message = Message(role="user", content=state["query"])
        assistant_message = Message(role="assistant", content=answer)
        if memory is not None:
            memory.add_turn(
                state["session_id"],
                user_message,
                assistant_message,
            )
        history = [
            *state.get("history", []),
            {"role": user_message.role, "content": user_message.content},
            {"role": assistant_message.role, "content": assistant_message.content},
        ][-max_history_messages:]
        return {
            "answer": answer,
            "history": history,
            # 最终 checkpoint 不保留命中文档正文和工具原始输出。
            "tool_calls": [],
            "tool_results": [],
            "retrieved_documents": [],
            "events": _append_event(
                state,
                "final",
                "回答已生成并写入会话记忆",
            ),
        }

    return commit_turn_node
