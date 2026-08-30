"""在同一数据集上运行 BM25、向量和混合检索。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from campus_agent.evaluation.metrics import CaseResult, summarize_results
from campus_agent.evaluation.models import EvaluationCase
from campus_agent.evaluation.validator import validate_dataset
from campus_agent.rag.models import RetrievalFilter, RetrievalHit
from campus_agent.rag.quality import (
    OfflineAnswerVerifier,
    OfflineDocumentGrader,
    OfflineQueryRewriter,
)
from campus_agent.rag.retriever import (
    DEFAULT_BM25_MINIMUM_INFORMATIVE_MATCHES,
    DEFAULT_HYBRID_LEXICAL_WEIGHT,
    DEFAULT_HYBRID_RRF_K,
    DEFAULT_HYBRID_VECTOR_WEIGHT,
)
from campus_agent.rag.service import LocalRAG


SUPPORTED_STRATEGIES = ("bm25", "vector", "hybrid")
SAFE_ABSTENTION_MARKERS = (
    "暂未找到",
    "没有提供",
    "无法确认",
    "无法从",
    "证据不足",
    "为避免编造",
)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    generated_at: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str | None
    embedding_fingerprint: str
    embedding_device: str
    embedding_runtime_versions: dict[str, str]
    embedding_minimum_similarity: float
    embedding_query_prompt: str
    vector_status: str
    configuration: dict[str, object]
    configuration_fingerprint: str
    corpus_fingerprint: str
    dataset_fingerprint: str
    case_count: int
    metric_k: int
    strategies: tuple[str, ...]
    summaries: dict[str, dict[str, object]]
    results: tuple[CaseResult, ...]

    def as_dict(self, *, include_results: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "1.2",
            "generated_at": self.generated_at,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_fingerprint": self.embedding_fingerprint,
            "embedding_device": self.embedding_device,
            "embedding_runtime_versions": self.embedding_runtime_versions,
            "embedding_minimum_similarity": self.embedding_minimum_similarity,
            "embedding_query_prompt": self.embedding_query_prompt,
            "vector_status": self.vector_status,
            "configuration": self.configuration,
            "configuration_fingerprint": self.configuration_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "case_count": self.case_count,
            "metric_k": self.metric_k,
            "strategies": list(self.strategies),
            "summaries": self.summaries,
        }
        if include_results:
            value["results"] = [result.as_dict() for result in self.results]
        return value


class EvaluationRunner:
    def __init__(self, rag: LocalRAG | None = None) -> None:
        self.rag = rag or LocalRAG()
        self._grader = OfflineDocumentGrader()
        self._rewriter = OfflineQueryRewriter()
        self._verifier = OfflineAnswerVerifier()

    def run(
        self,
        cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
        *,
        strategies: tuple[str, ...] | list[str] = SUPPORTED_STRATEGIES,
        split: str | None = None,
        metric_k: int = 3,
        validate: bool = True,
    ) -> EvaluationRun:
        selected_strategies = tuple(dict.fromkeys(strategies))
        unknown = set(selected_strategies) - set(SUPPORTED_STRATEGIES)
        if not selected_strategies or unknown:
            raise ValueError(
                f"strategy 只支持 {SUPPORTED_STRATEGIES}，收到：{sorted(unknown)}"
            )
        if metric_k < 1:
            raise ValueError("metric_k 必须至少为1")
        selected_cases = tuple(
            case for case in cases if split is None or case.split == split
        )
        if not selected_cases:
            raise ValueError("筛选后没有可运行的评测用例")
        if validate:
            validate_dataset(selected_cases, self.rag.chunks)

        results = tuple(
            self._run_case(case, strategy=strategy, metric_k=metric_k)
            for strategy in selected_strategies
            for case in selected_cases
        )
        summaries = {
            strategy: summarize_results(
                [result for result in results if result.strategy == strategy],
                k=metric_k,
            )
            for strategy in selected_strategies
        }
        configuration: dict[str, object] = {
            "evaluation": {
                "metric_k": metric_k,
                "strategies": list(selected_strategies),
                "selected_split": split or "all",
                "case_top_k_values": sorted(
                    {case.request.top_k for case in selected_cases}
                ),
                "dataset_validation": validate,
            },
            "retrieval": {
                "bm25": {
                    "minimum_informative_matches": (
                        DEFAULT_BM25_MINIMUM_INFORMATIVE_MATCHES
                    ),
                },
                "vector": {
                    "provider": self.rag.embedding_provider_name,
                    "model": self.rag.embedding_model_name,
                    "revision": self.rag.embedding_revision,
                    "fingerprint": self.rag.embedding_fingerprint,
                    "device": self.rag.embedding_device,
                    "runtime_versions": self.rag.embedding_runtime_versions,
                    "minimum_similarity": self.rag.embedding_minimum_similarity,
                    "query_prompt": self.rag.embedding_query_prompt,
                },
                "hybrid": {
                    "lexical_weight": DEFAULT_HYBRID_LEXICAL_WEIGHT,
                    "vector_weight": DEFAULT_HYBRID_VECTOR_WEIGHT,
                    "rrf_k": DEFAULT_HYBRID_RRF_K,
                },
            },
        }
        return EvaluationRun(
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedding_provider=self.rag.embedding_provider_name,
            embedding_model=self.rag.embedding_model_name,
            embedding_revision=self.rag.embedding_revision,
            embedding_fingerprint=self.rag.embedding_fingerprint,
            embedding_device=self.rag.embedding_device,
            embedding_runtime_versions=self.rag.embedding_runtime_versions,
            embedding_minimum_similarity=self.rag.embedding_minimum_similarity,
            embedding_query_prompt=self.rag.embedding_query_prompt,
            vector_status=self.rag.vector_status,
            configuration=configuration,
            configuration_fingerprint=self._fingerprint(configuration),
            corpus_fingerprint=self._corpus_fingerprint(),
            dataset_fingerprint=self._dataset_fingerprint(selected_cases),
            case_count=len(selected_cases),
            metric_k=metric_k,
            strategies=selected_strategies,
            summaries=summaries,
            results=results,
        )

    def _run_case(
        self,
        case: EvaluationCase,
        *,
        strategy: str,
        metric_k: int,
    ) -> CaseResult:
        started = perf_counter()
        active_filter = RetrievalFilter.from_mapping(case.request.filters)
        fetch_k = max(metric_k, case.request.top_k)
        trace_events: tuple[str, ...] = ()
        rewritten_query = ""
        rewrite_actual: bool | None = None
        initial_grade_actual: str | None = None
        grade_actual: str | None = None
        verification_actual: str | None = None
        retrieval_attempts = 1
        trace_conformant: bool | None = None

        hits = self.rag.retrieve(
            case.query,
            top_k=fetch_k,
            strategy=strategy,
            filters=active_filter,
        )
        if case.task == "agentic":
            (
                hits,
                rewritten_query,
                rewrite_actual,
                initial_grade_actual,
                grade_actual,
                verification_actual,
                retrieval_attempts,
                trace_events,
            ) = self._run_agentic(
                case,
                initial_hits=hits,
                strategy=strategy,
                top_k=fetch_k,
                filters=active_filter,
            )

        retrieved_hits = hits[: case.request.top_k]
        chunk_ids = tuple(hit.chunk.chunk_id for hit in retrieved_hits)
        document_ids = tuple(hit.chunk.metadata.document_id for hit in retrieved_hits)
        sources = tuple(hit.chunk.source for hit in retrieved_hits)
        evidence = "\n".join(
            f"{hit.chunk.title}\n{hit.chunk.content}" for hit in retrieved_hits
        ).casefold()
        generated_answer = ""
        if not case.expected.answerable:
            generated_answer = self.rag.answer(
                case.query,
                top_k=case.request.top_k,
                strategy=strategy,
                filters=active_filter,
            ).answer
        required_matches = sum(
            fact.casefold() in evidence for fact in case.expected.required_facts
        )
        forbidden_evidence = (
            generated_answer.casefold()
            if not case.expected.answerable
            else evidence
        )
        forbidden_violations = tuple(
            fact
            for fact in case.expected.forbidden_facts
            if fact.casefold() in forbidden_evidence
        )
        leakage = sum(
            not active_filter.matches(hit.chunk.metadata) for hit in retrieved_hits
        )

        failures: list[str] = []
        expected_chunks = set(case.expected.relevant_chunk_ids)
        safe_abstained: bool | None = None
        if case.retrieval_expectation == "hit":
            if not expected_chunks & set(chunk_ids[:metric_k]):
                failures.append(f"相关片段未进入 Top-{metric_k}")
            if required_matches < len(case.expected.required_facts):
                failures.append("必要事实未被完整召回")
        elif case.retrieval_expectation == "no_hit":
            safe_abstained = self._is_safe_abstention(
                generated_answer,
                case.expected.forbidden_facts,
            )
            if chunk_ids:
                failures.append("要求检索拒答的问题产生了命中")
        else:
            # missing_fact 允许召回主题资料；这里评估产品实际回答，而不是
            # 直接调用回答器内部的缺失事实规则，否则指标会与实现循环论证。
            safe_abstained = self._is_safe_abstention(
                generated_answer,
                case.expected.forbidden_facts,
            )
            if not safe_abstained:
                failures.append("相关主题已召回，但没有对缺失事实安全拒答")
        if leakage:
            failures.append("检索结果越过 metadata 过滤边界")
        if forbidden_violations:
            failures.append("召回了禁止事实：" + "、".join(forbidden_violations))

        agentic = case.agentic
        rewrite_policy: str | None = None
        resolved_rewrite_expected: bool | None = None
        if agentic is not None:
            assert rewrite_actual is not None
            assert initial_grade_actual is not None
            rewrite_policy = (
                agentic.rewrite_expected
                if isinstance(agentic.rewrite_expected, str)
                else "always"
                if agentic.rewrite_expected is True
                else "never"
                if agentic.rewrite_expected is False
                else None
            )
            resolved_rewrite_expected = agentic.resolve_rewrite_expected(
                initial_grade_actual
            )
            if (
                resolved_rewrite_expected is not None
                and rewrite_actual != resolved_rewrite_expected
            ):
                failures.append("Query Rewrite 触发结果不符合预期")
            missing_rewrite_terms = (
                [
                    term
                    for term in agentic.rewritten_query_contains
                    if term.casefold() not in rewritten_query.casefold()
                ]
                if resolved_rewrite_expected is True
                else []
            )
            if missing_rewrite_terms:
                failures.append("改写结果缺少：" + "、".join(missing_rewrite_terms))
            if agentic.grade_expected and grade_actual != agentic.grade_expected:
                failures.append("文档评分结果不符合预期")
            if (
                agentic.verification_expected
                and verification_actual != agentic.verification_expected
            ):
                failures.append("回答验证结果不符合预期")
            if retrieval_attempts > agentic.max_retrieval_attempts:
                failures.append("检索重试次数超过上限")
            missing_events = set(agentic.required_events) - set(trace_events)
            trace_conformant = not missing_events
            if missing_events:
                failures.append("Trace 缺少事件：" + "、".join(sorted(missing_events)))

        return CaseResult(
            case_id=case.case_id,
            task=case.task,
            split=case.split,
            strategy=strategy,
            query=case.query,
            tags=case.tags,
            difficulty=case.difficulty,
            expected_domain=case.expected.domain,
            expected_answerable=case.expected.answerable,
            retrieval_expectation=case.retrieval_expectation,
            expected_chunk_ids=case.expected.relevant_chunk_ids,
            expected_document_ids=case.expected.relevant_document_ids,
            retrieved_chunk_ids=chunk_ids,
            retrieved_document_ids=document_ids,
            retrieved_sources=sources,
            duration_ms=(perf_counter() - started) * 1000,
            filter_leakage_count=leakage,
            required_fact_total=len(case.expected.required_facts),
            required_fact_matches=required_matches,
            forbidden_fact_violations=forbidden_violations,
            rewrite_policy=rewrite_policy,
            rewrite_expected=resolved_rewrite_expected,
            rewrite_actual=rewrite_actual,
            rewritten_query=rewritten_query,
            initial_grade_actual=initial_grade_actual,
            grade_expected=agentic.grade_expected if agentic else None,
            grade_actual=grade_actual,
            verification_expected=(agentic.verification_expected if agentic else None),
            verification_actual=verification_actual,
            retrieval_attempts=retrieval_attempts,
            trace_events=trace_events,
            trace_conformant=trace_conformant,
            safe_abstained=safe_abstained,
            generated_answer=generated_answer,
            injection_contained=None,
            failures=tuple(failures),
        )

    def _run_agentic(
        self,
        case: EvaluationCase,
        *,
        initial_hits: tuple[RetrievalHit, ...],
        strategy: str,
        top_k: int,
        filters: RetrievalFilter,
    ) -> tuple[
        tuple[RetrievalHit, ...],
        str,
        bool,
        str,
        str,
        str,
        int,
        tuple[str, ...],
    ]:
        events = ["retrieve"]
        hits = initial_hits
        grade = self._grader.grade(case.query, self._serialize_hits(hits))
        initial_grade_status = grade.status
        events.append("grade")
        rewritten_query = case.query
        rewrite_actual = False
        attempts = 1

        if grade.status == "insufficient":
            rewrite = self._rewriter.rewrite(case.query)
            events.append("rewrite")
            rewritten_query = rewrite.query
            rewrite_actual = rewrite.changed
            if rewrite.changed:
                attempts += 1
                hits = self.rag.retrieve(
                    rewrite.query,
                    top_k=top_k,
                    strategy=strategy,
                    filters=filters,
                )
                events.append("retrieve")
                grade = self._grader.grade(
                    rewrite.query,
                    self._serialize_hits(hits),
                )
                events.append("grade")

        answer, tool_results, citations = self._verification_inputs(
            query=rewritten_query,
            hits=hits,
        )
        verification = self._verifier.verify(answer, tool_results, citations)
        events.append("verify")
        return (
            hits,
            rewritten_query,
            rewrite_actual,
            initial_grade_status,
            grade.status,
            verification.status,
            attempts,
            tuple(events),
        )

    @staticmethod
    def _is_safe_abstention(
        answer: str,
        forbidden_facts: tuple[str, ...],
    ) -> bool:
        normalized = answer.casefold()
        has_abstention_marker = any(
            marker.casefold() in normalized for marker in SAFE_ABSTENTION_MARKERS
        )
        contains_forbidden_fact = any(
            fact.casefold() in normalized for fact in forbidden_facts
        )
        return has_abstention_marker and not contains_forbidden_fact

    @staticmethod
    def _fingerprint(value: object) -> str:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _dataset_fingerprint(
        cls,
        cases: tuple[EvaluationCase, ...],
    ) -> str:
        payload = [
            case.as_dict()
            for case in sorted(cases, key=lambda item: item.case_id)
        ]
        return cls._fingerprint(payload)

    def _corpus_fingerprint(self) -> str:
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "source": chunk.source,
                "metadata": chunk.metadata.as_dict(),
            }
            for chunk in sorted(self.rag.chunks, key=lambda item: item.chunk_id)
        ]
        return self._fingerprint(payload)

    @staticmethod
    def _serialize_hits(hits: tuple[RetrievalHit, ...]) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title": hit.chunk.title,
                "content": hit.chunk.content,
                "source": hit.chunk.source,
                "score": hit.score,
                "lexical_score": hit.lexical_score,
                "vector_score": hit.vector_score,
            }
            for hit in hits
        ]

    @classmethod
    def _verification_inputs(
        cls,
        *,
        query: str,
        hits: tuple[RetrievalHit, ...],
    ) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
        serialized_hits = cls._serialize_hits(hits)
        if hits:
            answer = f"{hits[0].chunk.title}：{hits[0].chunk.content}"
            citations = [{"chunk_id": hits[0].chunk.chunk_id}]
        else:
            answer = "知识库中暂未找到相关信息。"
            citations = []
        return (
            answer,
            [
                {
                    "ok": True,
                    "tool_name": "search_campus_knowledge",
                    "output": answer,
                    "data": {"query": query, "hits": serialized_hits},
                }
            ],
            citations,
        )
