"""Agentic RAG 的离线质量控制组件。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from campus_agent.rag.retriever import GENERIC_BIGRAMS, tokenize


@dataclass(frozen=True, slots=True)
class GradeResult:
    status: Literal["relevant", "insufficient"]
    reason: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    query: str
    changed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: Literal["grounded", "insufficient_evidence", "unsupported"]
    reason: str


class DocumentGrader(Protocol):
    def grade(self, query: str, hits: list[dict[str, object]]) -> GradeResult: ...


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> RewriteResult: ...


class AnswerVerifier(Protocol):
    def verify(
        self,
        answer: str,
        tool_results: list[dict[str, object]],
        citations: list[dict[str, object]],
    ) -> VerificationResult: ...


class OfflineDocumentGrader:
    """根据检索通道分数和片段完整性判断资料是否可用。"""

    def grade(self, query: str, hits: list[dict[str, object]]) -> GradeResult:
        if not hits:
            return GradeResult("insufficient", "没有检索到候选片段")
        query_terms = {
            term
            for term in tokenize(query)
            if (term.isascii() and len(term) >= 2)
            or (len(term) == 2 and term not in GENERIC_BIGRAMS)
        }
        for hit in hits:
            title = str(hit.get("title", "")).strip()
            content = str(hit.get("content", "")).strip()
            lexical_score = float(hit.get("lexical_score", 0.0) or 0.0)
            vector_score = float(hit.get("vector_score", 0.0) or 0.0)
            document_terms = set(tokenize(f"{title} {content}"))
            overlap = query_terms & document_terms
            if content and lexical_score > 0 and len(overlap) >= 2:
                return GradeResult("relevant", "词法命中覆盖至少两个问题信息词")
            if content and vector_score >= 0.55:
                return GradeResult("relevant", "语义相似度达到强相关阈值")
            if content and vector_score >= 0.30 and len(overlap) >= 2:
                return GradeResult("relevant", "语义候选同时具备信息词覆盖")
        return GradeResult("insufficient", "候选片段与问题缺少足够相关证据")


class OfflineQueryRewriter:
    """只做保守别名替换，不丢弃时间、课程名和业务编号。"""

    _replacements = (
        ("一卡通", "校园卡"),
        ("饭卡", "校园卡"),
        ("不见了", "遗失"),
        ("弄丢了", "遗失"),
        ("丢掉了", "遗失"),
        ("重新办理", "补办"),
        ("重新办", "补办"),
        ("重办", "补办"),
        ("换一张", "补办"),
        ("坏了", "异常"),
    )
    _prefixes = ("请问", "麻烦问一下", "我想知道", "能不能告诉我")

    def rewrite(self, query: str) -> RewriteResult:
        rewritten = query.strip()
        for prefix in self._prefixes:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix) :].lstrip("，,：: ")
        applied: list[str] = []
        for source, target in self._replacements:
            if source in rewritten:
                rewritten = rewritten.replace(source, target)
                applied.append(f"{source}→{target}")
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        changed = bool(rewritten) and rewritten != query.strip()
        return RewriteResult(
            query=rewritten or query.strip(),
            changed=changed,
            reason=("；".join(applied) if applied else "没有可安全替换的别名"),
        )


class OfflineAnswerVerifier:
    """校验引用归属，以及回答中新出现的编号、时间和金额。"""

    _fact_pattern = re.compile(
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b|\b\d{1,2}:\d{2}\b|"
        r"\b\d+(?:\.\d+)?\s*(?:元|万元|%|天|日|小时|分钟)"
    )

    def verify(
        self,
        answer: str,
        tool_results: list[dict[str, object]],
        citations: list[dict[str, object]],
    ) -> VerificationResult:
        successful = [result for result in tool_results if bool(result.get("ok"))]
        if not successful:
            return VerificationResult("insufficient_evidence", "没有成功工具结果")

        knowledge_results = [
            result
            for result in successful
            if result.get("tool_name") == "search_campus_knowledge"
        ]
        known_chunk_ids: set[str] = set()
        evidence_parts: list[str] = []
        for result in successful:
            evidence_parts.append(str(result.get("output", "")))
            data = result.get("data", {})
            if not isinstance(data, dict):
                continue
            hits = data.get("hits", [])
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                known_chunk_ids.add(str(hit.get("chunk_id", "")))
                evidence_parts.append(str(hit.get("content", "")))

        knowledge_without_hits = [
            result
            for result in knowledge_results
            if not isinstance(result.get("data"), dict)
            or not result["data"].get("hits")
        ]
        if knowledge_without_hits:
            return VerificationResult(
                "insufficient_evidence",
                "至少一个知识检索没有返回可引用片段",
            )
        if known_chunk_ids:
            cited_ids = {str(citation.get("chunk_id", "")) for citation in citations}
            if not cited_ids or not cited_ids.issubset(known_chunk_ids):
                return VerificationResult("unsupported", "引用不属于本轮检索结果")

        evidence = "\n".join(evidence_parts)
        unsupported_facts = sorted(
            fact
            for fact in set(self._fact_pattern.findall(answer))
            if fact not in evidence
        )
        if unsupported_facts:
            return VerificationResult(
                "unsupported",
                f"回答包含证据中不存在的事实：{', '.join(unsupported_facts)}",
            )
        return VerificationResult("grounded", "回答中的可验证事实均有工具证据")
