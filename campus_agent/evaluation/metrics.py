"""检索与 Agentic RAG 指标；Hit 与 Recall 使用不同分母。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    task: str
    split: str
    strategy: str
    query: str
    tags: tuple[str, ...]
    difficulty: str
    expected_domain: str | None
    expected_answerable: bool
    retrieval_expectation: str
    expected_chunk_ids: tuple[str, ...]
    expected_document_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    duration_ms: float
    filter_leakage_count: int = 0
    required_fact_total: int = 0
    required_fact_matches: int = 0
    forbidden_fact_violations: tuple[str, ...] = ()
    rewrite_policy: str | None = None
    rewrite_expected: bool | None = None
    rewrite_actual: bool | None = None
    rewritten_query: str = ""
    initial_grade_actual: str | None = None
    grade_expected: str | None = None
    grade_actual: str | None = None
    verification_expected: str | None = None
    verification_actual: str | None = None
    retrieval_attempts: int = 1
    trace_events: tuple[str, ...] = ()
    trace_conformant: bool | None = None
    safe_abstained: bool | None = None
    generated_answer: str = ""
    injection_contained: bool | None = None
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "task": self.task,
            "split": self.split,
            "strategy": self.strategy,
            "query": self.query,
            "tags": list(self.tags),
            "difficulty": self.difficulty,
            "expected_domain": self.expected_domain,
            "expected_answerable": self.expected_answerable,
            "retrieval_expectation": self.retrieval_expectation,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_document_ids": list(self.expected_document_ids),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_document_ids": list(self.retrieved_document_ids),
            "retrieved_sources": list(self.retrieved_sources),
            "duration_ms": round(self.duration_ms, 3),
            "filter_leakage_count": self.filter_leakage_count,
            "required_fact_total": self.required_fact_total,
            "required_fact_matches": self.required_fact_matches,
            "forbidden_fact_violations": list(self.forbidden_fact_violations),
            "rewrite_policy": self.rewrite_policy,
            "rewrite_expected": self.rewrite_expected,
            "rewrite_actual": self.rewrite_actual,
            "rewritten_query": self.rewritten_query,
            "initial_grade_actual": self.initial_grade_actual,
            "grade_expected": self.grade_expected,
            "grade_actual": self.grade_actual,
            "verification_expected": self.verification_expected,
            "verification_actual": self.verification_actual,
            "retrieval_attempts": self.retrieval_attempts,
            "trace_events": list(self.trace_events),
            "trace_conformant": self.trace_conformant,
            "safe_abstained": self.safe_abstained,
            "generated_answer": self.generated_answer,
            "injection_contained": self.injection_contained,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _ratio(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    """没有适用样本时返回N/A，而不是会被误解为失败的0分。"""

    return float(numerator / denominator) if denominator else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_metrics(
    results: list[CaseResult] | tuple[CaseResult, ...],
    *,
    k: int = 3,
) -> dict[str, object]:
    selected = tuple(results)
    positives = tuple(
        result for result in selected if result.retrieval_expectation == "hit"
    )
    no_hit_cases = tuple(
        result for result in selected if result.retrieval_expectation == "no_hit"
    )
    hard_negatives = tuple(
        result
        for result in selected
        if result.retrieval_expectation == "topic_hit_allowed"
    )

    hit_at_1 = 0
    hit_at_k = 0
    document_hit_at_k = 0
    reciprocal_rank_total = 0.0
    relevant_retrieved = 0
    relevant_total = 0
    for result in positives:
        expected_chunks = set(result.expected_chunk_ids)
        expected_documents = set(result.expected_document_ids)
        retrieved_at_k = result.retrieved_chunk_ids[:k]
        if result.retrieved_chunk_ids[:1] and result.retrieved_chunk_ids[0] in expected_chunks:
            hit_at_1 += 1
        matching_ranks = [
            index
            for index, chunk_id in enumerate(retrieved_at_k, start=1)
            if chunk_id in expected_chunks
        ]
        if matching_ranks:
            hit_at_k += 1
            reciprocal_rank_total += 1 / matching_ranks[0]
        if expected_documents & set(result.retrieved_document_ids[:k]):
            document_hit_at_k += 1
        relevant_retrieved += len(expected_chunks & set(retrieved_at_k))
        relevant_total += len(expected_chunks)

    false_positive_count = sum(
        bool(result.retrieved_chunk_ids) for result in no_hit_cases
    )
    no_hit_count = len(no_hit_cases) - false_positive_count
    fact_total = sum(result.required_fact_total for result in selected)
    fact_matches = sum(result.required_fact_matches for result in selected)
    forbidden_violations = sum(
        len(result.forbidden_fact_violations) for result in selected
    )
    retrieved_count = sum(len(result.retrieved_chunk_ids) for result in selected)
    filter_leakage = sum(result.filter_leakage_count for result in selected)

    rewrite_cases = [result for result in selected if result.rewrite_expected is not None]
    grade_cases = [result for result in selected if result.grade_expected is not None]
    verification_cases = [
        result for result in selected if result.verification_expected is not None
    ]
    trace_cases = [result for result in selected if result.trace_conformant is not None]
    injection_case_count = sum("prompt_injection" in result.tags for result in selected)

    return {
        "cases": len(selected),
        "positive_cases": len(positives),
        "negative_cases": len(no_hit_cases) + len(hard_negatives),
        "retrieval_no_hit_cases": len(no_hit_cases),
        "hard_negative_cases": len(hard_negatives),
        "passed_cases": sum(result.passed for result in selected),
        "failed_cases": sum(not result.passed for result in selected),
        "hit_at_1": _ratio(hit_at_1, len(positives)),
        f"hit_at_{k}": _ratio(hit_at_k, len(positives)),
        f"document_hit_at_{k}": _ratio(document_hit_at_k, len(positives)),
        f"recall_at_{k}": _ratio(relevant_retrieved, relevant_total),
        f"mrr_at_{k}": _ratio(reciprocal_rank_total, len(positives)),
        "no_hit_accuracy": _ratio(no_hit_count, len(no_hit_cases)),
        "false_positive_rate": _ratio(false_positive_count, len(no_hit_cases)),
        "hard_negative_safe_abstention_rate": _ratio(
            sum(result.safe_abstained is True for result in hard_negatives),
            len(hard_negatives),
        ),
        "required_fact_coverage": _ratio(fact_matches, fact_total),
        "forbidden_fact_violations": forbidden_violations,
        "filter_leakage_rate": _ratio(filter_leakage, retrieved_count),
        "rewrite_accuracy": _ratio(
            sum(result.rewrite_actual == result.rewrite_expected for result in rewrite_cases),
            len(rewrite_cases),
        ),
        "grade_accuracy": _ratio(
            sum(result.grade_actual == result.grade_expected for result in grade_cases),
            len(grade_cases),
        ),
        "verification_accuracy": _ratio(
            sum(
                result.verification_actual == result.verification_expected
                for result in verification_cases
            ),
            len(verification_cases),
        ),
        "trace_conformance": _ratio(
            sum(result.trace_conformant is True for result in trace_cases),
            len(trace_cases),
        ),
        # 当前离线 runner 没有执行真实 Agent 工具路由，不能据合成 trace 声称
        # prompt injection 已被拦截。保留显式 N/A，避免把“未执行工具”误报为安全能力。
        "injection_case_count": injection_case_count,
        "injection_containment_status": "not_evaluated",
        "injection_containment_rate": None,
        "latency_ms_p50": round(median([result.duration_ms for result in selected]), 3)
        if selected
        else 0.0,
        "latency_ms_p95": round(
            _percentile([result.duration_ms for result in selected], 0.95),
            3,
        ),
    }


def summarize_results(
    results: list[CaseResult] | tuple[CaseResult, ...],
    *,
    k: int = 3,
) -> dict[str, object]:
    selected = tuple(results)
    tags = sorted({tag for result in selected for tag in result.tags})
    domains = sorted(
        {result.expected_domain for result in selected if result.expected_domain}
    )
    difficulties = sorted({result.difficulty for result in selected})
    splits = sorted({result.split for result in selected})
    tasks = sorted({result.task for result in selected})
    return {
        "overall": aggregate_metrics(selected, k=k),
        "by_split": {
            split: aggregate_metrics(
                [result for result in selected if result.split == split],
                k=k,
            )
            for split in splits
        },
        "by_task": {
            task: aggregate_metrics(
                [result for result in selected if result.task == task],
                k=k,
            )
            for task in tasks
        },
        "by_tag": {
            tag: aggregate_metrics(
                [result for result in selected if tag in result.tags],
                k=k,
            )
            for tag in tags
        },
        "by_domain": {
            domain: aggregate_metrics(
                [result for result in selected if result.expected_domain == domain],
                k=k,
            )
            for domain in domains
        },
        "by_difficulty": {
            difficulty: aggregate_metrics(
                [result for result in selected if result.difficulty == difficulty],
                k=k,
            )
            for difficulty in difficulties
        },
    }
