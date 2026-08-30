"""评测集的语法、唯一性与 Ground Truth 一致性检查。"""

from __future__ import annotations

from dataclasses import dataclass

from campus_agent.evaluation.models import EvaluationCase
from campus_agent.rag.models import DocumentChunk, RetrievalFilter


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    case_id: str
    message: str
    source_file: str = ""
    source_line: int = 0

    def __str__(self) -> str:
        location = (
            f"{self.source_file}:{self.source_line}: " if self.source_file else ""
        )
        return f"{location}{self.case_id}: {self.message}"


class DatasetValidationError(ValueError):
    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        preview = "\n".join(str(issue) for issue in issues[:20])
        if len(issues) > 20:
            preview += f"\n……另有 {len(issues) - 20} 个问题"
        super().__init__(f"评测数据校验失败（{len(issues)} 个问题）：\n{preview}")


def validate_dataset(
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
    *,
    raise_on_error: bool = True,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    document_ids = {chunk.metadata.document_id for chunk in chunks}
    seen_ids: set[str] = set()
    seen_queries: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def add(case: EvaluationCase, message: str) -> None:
        issues.append(
            ValidationIssue(
                case_id=case.case_id,
                message=message,
                source_file=case.source_file,
                source_line=case.source_line,
            )
        )

    for case in cases:
        if case.case_id in seen_ids:
            add(case, "id 重复")
        seen_ids.add(case.case_id)
        query_key = (case.query.casefold(), tuple(sorted(case.request.filters.items())))
        if query_key in seen_queries:
            add(case, "query 与 filters 的组合重复")
        seen_queries.add(query_key)

        expected = case.expected
        if expected.answerable:
            if not expected.relevant_chunk_ids:
                add(case, "可回答用例必须指定 relevant_chunk_ids")
            if not expected.relevant_document_ids:
                add(case, "可回答用例必须指定 relevant_document_ids")
        elif expected.relevant_chunk_ids or expected.relevant_document_ids:
            add(case, "不可回答用例不能声明相关文档或片段")
        if not expected.answerable:
            modes = {"retrieval_no_hit", "missing_fact"} & set(case.tags)
            if len(modes) != 1:
                add(
                    case,
                    "不可回答用例必须且只能标记 retrieval_no_hit 或 missing_fact",
                )

        missing_chunks = set(expected.relevant_chunk_ids) - chunks_by_id.keys()
        if missing_chunks:
            add(case, f"相关 chunk 不存在：{sorted(missing_chunks)}")
        missing_documents = set(expected.relevant_document_ids) - document_ids
        if missing_documents:
            add(case, f"相关 document 不存在：{sorted(missing_documents)}")

        relevant_chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in expected.relevant_chunk_ids
            if chunk_id in chunks_by_id
        ]
        relevant_document_ids = {
            chunk.metadata.document_id for chunk in relevant_chunks
        }
        if relevant_document_ids - set(expected.relevant_document_ids):
            add(case, "relevant_chunk_ids 与 relevant_document_ids 不一致")

        try:
            active_filter = RetrievalFilter.from_mapping(case.request.filters)
        except ValueError as error:
            add(case, str(error))
            active_filter = RetrievalFilter()
        for chunk in relevant_chunks:
            if not active_filter.matches(chunk.metadata):
                add(case, f"Ground Truth {chunk.chunk_id} 被 request.filters 排除")
            if expected.domain and chunk.metadata.domain != expected.domain:
                add(case, f"Ground Truth {chunk.chunk_id} 的 domain 不匹配")
            if expected.category and chunk.metadata.category != expected.category:
                add(case, f"Ground Truth {chunk.chunk_id} 的 category 不匹配")

        evidence = "\n".join(
            f"{chunk.title}\n{chunk.content}" for chunk in relevant_chunks
        ).casefold()
        for fact in expected.required_facts:
            if fact.casefold() not in evidence:
                add(case, f"required_fact 不在 Ground Truth 中：{fact}")
        overlap = {
            fact.casefold() for fact in expected.required_facts
        } & {fact.casefold() for fact in expected.forbidden_facts}
        if overlap:
            add(case, f"required_facts 与 forbidden_facts 冲突：{sorted(overlap)}")

        if not case.tags:
            add(case, "至少需要一个 tag")
        if case.task == "agentic" and case.agentic is not None:
            if (
                case.agentic.rewrite_expected is False
                and case.agentic.rewritten_query_contains
            ):
                add(case, "不期望改写时不能声明 rewritten_query_contains")
            if case.agentic.max_retrieval_attempts > 2:
                add(case, "当前 Agent 图最多允许两次检索")

    frozen_issues = tuple(issues)
    if frozen_issues and raise_on_error:
        raise DatasetValidationError(frozen_issues)
    return frozen_issues
