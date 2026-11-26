"""RAGAS scoring over the project's reviewed cases and real retrieval traces."""

from __future__ import annotations

import asyncio
import math
import os
import warnings
from collections import defaultdict
from statistics import mean
from typing import Any

from campus_agent.evaluation.metrics import CaseResult
from campus_agent.evaluation.models import EvaluationCase
from campus_agent.rag.embeddings import EmbeddingProvider
from campus_agent.rag.models import RetrievalFilter
from campus_agent.rag.service import LocalRAG


def _ragas_retrieval_metrics() -> tuple[Any, Any, Any]:
    # RAGAS 0.4.3 still exposes these ID metrics through its legacy namespace.
    # Disable optional telemetry before importing it: evaluation stays offline.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        from ragas.dataset_schema import SingleTurnSample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except ImportError as error:
        raise RuntimeError(
            "缺少 RAGAS 依赖；请安装 campus-agent[ragas-eval]"
        ) from error
    return SingleTurnSample, IDBasedContextPrecision(), IDBasedContextRecall()


def _average(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


async def score_retrieval(
    cases: tuple[EvaluationCase, ...],
    results: tuple[CaseResult, ...],
) -> dict[str, object]:
    """Score only answerable cases; no-hit/missing-fact are separate checks.

    RAGAS returns NaN for empty retrieved IDs (precision) or reference IDs
    (recall). We represent those scores as null and retain their counts.
    """

    sample_type, precision_metric, recall_metric = _ragas_retrieval_metrics()
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("评测用例 ID 重复")
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        case = by_id[result.case_id]
        if case.expected.answerable:
            sample = sample_type(
                user_input=case.query,
                retrieved_context_ids=list(result.retrieved_chunk_ids),
                reference_context_ids=list(case.expected.relevant_chunk_ids),
            )
            raw_precision, raw_recall = await asyncio.gather(
                precision_metric.single_turn_ascore(sample),
                recall_metric.single_turn_ascore(sample),
            )
            precision = None if math.isnan(raw_precision) else float(raw_precision)
            recall = None if math.isnan(raw_recall) else float(raw_recall)
        else:
            precision = recall = None
        row = {
            "id": case.case_id,
            "split": case.split,
            "task": case.task,
            "strategy": result.strategy,
            "retrieval_expectation": case.retrieval_expectation,
            "retrieved_context_ids": list(result.retrieved_chunk_ids),
            "reference_context_ids": list(case.expected.relevant_chunk_ids),
            "context_precision": precision,
            "context_recall": recall,
            "safe_abstained": result.safe_abstained,
            "legacy_passed": result.passed,
            "legacy_failures": list(result.failures),
        }
        rows.append(row)
        grouped[result.strategy].append(row)

    summaries: dict[str, dict[str, object]] = {}
    for strategy, items in grouped.items():
        answerable = [item for item in items if item["retrieval_expectation"] == "hit"]
        no_hit = [item for item in items if item["retrieval_expectation"] == "no_hit"]
        missing_fact = [
            item for item in items if item["retrieval_expectation"] == "topic_hit_allowed"
        ]
        precisions = [float(item["context_precision"]) for item in answerable if item["context_precision"] is not None]
        recalls = [float(item["context_recall"]) for item in answerable if item["context_recall"] is not None]
        no_hit_fp = sum(bool(item["retrieved_context_ids"]) for item in no_hit)
        summaries[strategy] = {
            "total_cases": len(items),
            "answerable_cases": len(answerable),
            "context_precision": _average(precisions),
            "context_precision_scored": len(precisions),
            "context_precision_na": len(answerable) - len(precisions),
            "context_recall": _average(recalls),
            "context_recall_scored": len(recalls),
            "no_hit_cases": len(no_hit),
            "no_hit_false_positives": no_hit_fp,
            "no_hit_false_positive_rate": round(no_hit_fp / len(no_hit), 6) if no_hit else None,
            "missing_fact_cases": len(missing_fact),
            "missing_fact_safe_abstentions": sum(item["safe_abstained"] is True for item in missing_fact),
            "legacy_failures": sum(item["legacy_passed"] is False for item in items),
        }
    return {"summaries": summaries, "rows": rows}


def _judge_embeddings(provider: EmbeddingProvider) -> Any:
    from ragas.embeddings.base import BaseRagasEmbedding

    class CampusEmbedding(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return list(provider.embed_query(text))

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

    return CampusEmbedding()


async def score_rule_answers(
    cases: tuple[EvaluationCase, ...],
    rag: LocalRAG,
    *,
    judge_model: str,
    max_cases: int | None = None,
) -> dict[str, object]:
    """Opt-in LLM judge for actual LocalRAG rule answers, not synthetic traces."""

    if rag.embedding_provider_name != "sentence-transformers":
        raise ValueError(
            "Answer Relevancy 需要语义向量；先配置 CAMPUS_EMBEDDING_PROVIDER=sentence-transformers"
        )
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("生成评测需要 DEEPSEEK_API_KEY；离线检索评测不需要")
    if not judge_model.strip():
        raise ValueError("judge_model 不能为空")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")
    llm = llm_factory(judge_model, provider="openai", client=client)
    faithfulness_metric = Faithfulness(llm=llm)
    relevance_metric = AnswerRelevancy(
        llm=llm, embeddings=_judge_embeddings(rag._embedding_provider)
    )
    selected = [case for case in cases if case.expected.answerable]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases 必须大于 0")
        selected = selected[:max_cases]
    rows: list[dict[str, object]] = []
    try:
        for case in selected:
            answer = rag.answer(
                case.query,
                top_k=case.request.top_k,
                filters=RetrievalFilter.from_mapping(case.request.filters),
                strategy="hybrid",
            )
            contexts = [f"{hit.chunk.title}\n{hit.chunk.content}" for hit in answer.hits]
            row: dict[str, object] = {
                "id": case.case_id,
                "response_source": "LocalRAG.answer rule baseline",
                "response": answer.answer,
                "retrieved_context_ids": [hit.chunk.chunk_id for hit in answer.hits],
                "retrieved_contexts": contexts,
                "faithfulness": None,
                "answer_relevancy": None,
            }
            if contexts:
                result = await faithfulness_metric.ascore(
                    user_input=case.query,
                    response=answer.answer,
                    retrieved_contexts=contexts,
                )
                row["faithfulness"] = float(result.value)
            result = await relevance_metric.ascore(
                user_input=case.query, response=answer.answer
            )
            row["answer_relevancy"] = float(result.value)
            rows.append(row)
    finally:
        await client.close()
    faithfulness = [float(row["faithfulness"]) for row in rows if row["faithfulness"] is not None]
    relevance = [float(row["answer_relevancy"]) for row in rows if row["answer_relevancy"] is not None]
    return {
        "source": "LocalRAG.answer rule baseline; not CampusGraphAgent",
        "judge_model": judge_model,
        "embedding_fingerprint": rag.embedding_fingerprint,
        "selected_cases": len(selected),
        "completed_cases": len(rows),
        "faithfulness": _average(faithfulness),
        "faithfulness_scored": len(faithfulness),
        "answer_relevancy": _average(relevance),
        "answer_relevancy_scored": len(relevance),
        "rows": rows,
    }
