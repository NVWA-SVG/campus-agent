"""基于本地 Markdown 知识库和 BM25 检索的 RAG 工具。"""

from __future__ import annotations

from typing import Any

from campus_agent.domain import ToolOutput
from campus_agent.rag import LocalRAG
from campus_agent.rag.models import RAGAnswer, RetrievalFilter
from campus_agent.tooling import Tool


class CampusKnowledgeTool(Tool):
    name = "search_campus_knowledge"
    description = "使用RAG查询校园办事流程，返回答案及Markdown知识来源。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "需要查询的校园事务问题"}
        },
        "required": ["query"],
    }

    def __init__(self, rag: LocalRAG | None = None) -> None:
        self._rag = rag or LocalRAG()

    def run(self, **arguments: Any) -> str:
        return self._answer(arguments).answer

    def invoke(self, **arguments: Any) -> ToolOutput:
        answer = self._answer(arguments)
        return ToolOutput(
            text=answer.answer,
            citations=answer.citations,
            data={
                "query": answer.query,
                "hits": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "title": hit.chunk.title,
                        "content": hit.chunk.content,
                        "source": hit.chunk.metadata.source_name,
                        "score": hit.score,
                        "lexical_score": hit.lexical_score,
                        "vector_score": hit.vector_score,
                        "retrieval_method": hit.retrieval_method,
                        "metadata": hit.chunk.metadata.as_dict(),
                    }
                    for hit in answer.hits
                ],
            },
        )

    @property
    def rag(self) -> LocalRAG:
        return self._rag

    def _answer(self, arguments: dict[str, Any]) -> RAGAnswer:
        query_value = arguments.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            raise ValueError("query 必须是非空字符串")
        filters = RetrievalFilter.from_mapping(arguments.get("filters"))
        return self._rag.answer(query_value.strip(), filters=filters)
