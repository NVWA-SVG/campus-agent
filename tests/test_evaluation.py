from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from campus_agent.evaluation.metrics import CaseResult, aggregate_metrics
from campus_agent.evaluation.models import EvaluationCase, load_dataset
from campus_agent.evaluation.report import write_report
from campus_agent.evaluation.runner import EvaluationRunner
from campus_agent.evaluation.validator import DatasetValidationError, validate_dataset
from campus_agent.rag import LocalRAG
from campus_agent.rag.embeddings import HashingEmbeddingProvider
from campus_agent.rag.models import RAGAnswer, RetrievalHit


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "eval" / "datasets"


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_dataset(DATASET_DIR)
        cls.rag = LocalRAG()

    def test_fixed_dataset_has_reviewed_100_case_distribution(self) -> None:
        self.assertEqual(len(self.cases), 100)
        self.assertEqual(sum(case.task == "agentic" for case in self.cases), 10)
        self.assertEqual(sum(case.expected.answerable for case in self.cases), 75)
        self.assertEqual(sum(case.split == "test" for case in self.cases), 32)
        self.assertTrue(all(case.review_status == "human_checked" for case in self.cases))

    def test_ground_truth_matches_current_corpus(self) -> None:
        self.assertEqual(validate_dataset(self.cases, self.rag.chunks), ())

    def test_duplicate_ids_and_unknown_chunks_are_rejected(self) -> None:
        original = self.cases[0]
        broken_expected = replace(
            original.expected,
            relevant_chunk_ids=("missing-chunk",),
        )
        broken = replace(original, expected=broken_expected)

        with self.assertRaises(DatasetValidationError) as context:
            validate_dataset((original, broken), self.rag.chunks)

        messages = "\n".join(issue.message for issue in context.exception.issues)
        self.assertIn("id 重复", messages)
        self.assertIn("相关 chunk 不存在", messages)

    def test_schema_parser_rejects_unknown_fields(self) -> None:
        raw = self.cases[0].as_dict()
        raw["surprise"] = True

        with self.assertRaisesRegex(ValueError, "未知字段"):
            EvaluationCase.from_mapping(raw)


class EvaluationMetricTests(unittest.TestCase):
    @staticmethod
    def _result(**changes) -> CaseResult:
        base = CaseResult(
            case_id="case-1",
            task="retrieval",
            split="test",
            strategy="bm25",
            query="query",
            tags=("unit",),
            difficulty="medium",
            expected_domain="test",
            expected_answerable=True,
            retrieval_expectation="hit",
            expected_chunk_ids=("a", "b"),
            expected_document_ids=("doc-a",),
            retrieved_chunk_ids=("a", "x"),
            retrieved_document_ids=("doc-a", "doc-x"),
            retrieved_sources=("a.md", "x.md"),
            duration_ms=1.0,
        )
        return replace(base, **changes)

    def test_hit_and_true_recall_use_different_denominators(self) -> None:
        metrics = aggregate_metrics([self._result()], k=3)

        self.assertEqual(metrics["hit_at_3"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 0.5)
        self.assertEqual(metrics["mrr_at_3"], 1.0)

    def test_no_hit_accuracy_and_false_positive_rate_are_complements(self) -> None:
        clean = self._result(
            case_id="negative-clean",
            expected_answerable=False,
            retrieval_expectation="no_hit",
            expected_chunk_ids=(),
            expected_document_ids=(),
            retrieved_chunk_ids=(),
            retrieved_document_ids=(),
            retrieved_sources=(),
        )
        false_positive = replace(
            clean,
            case_id="negative-fp",
            retrieved_chunk_ids=("x",),
            retrieved_document_ids=("doc-x",),
            retrieved_sources=("x.md",),
        )

        metrics = aggregate_metrics([clean, false_positive], k=3)

        self.assertEqual(metrics["no_hit_accuracy"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertIsNone(metrics["hit_at_1"])

    def test_missing_fact_topic_hit_is_not_counted_as_retrieval_false_positive(self) -> None:
        no_hit = self._result(
            case_id="negative-no-hit",
            expected_answerable=False,
            retrieval_expectation="no_hit",
            expected_chunk_ids=(),
            expected_document_ids=(),
            retrieved_chunk_ids=(),
            retrieved_document_ids=(),
            retrieved_sources=(),
            safe_abstained=True,
        )
        missing_fact = self._result(
            case_id="negative-missing-fact",
            expected_answerable=False,
            retrieval_expectation="topic_hit_allowed",
            expected_chunk_ids=(),
            expected_document_ids=(),
            safe_abstained=False,
        )

        metrics = aggregate_metrics([no_hit, missing_fact], k=3)

        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["no_hit_accuracy"], 1.0)
        self.assertEqual(metrics["hard_negative_safe_abstention_rate"], 0.0)


class EvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_dataset(DATASET_DIR)

    def test_runner_compares_all_three_strategies(self) -> None:
        subset = tuple(
            case
            for case in self.cases
            if case.case_id in {
                "pos-card-loss-001",
                "meta-negative-origin-001",
                "meta-doc-library-001",
            }
        )

        run = EvaluationRunner().run(subset)

        self.assertEqual(run.strategies, ("bm25", "vector", "hybrid"))
        self.assertEqual(len(run.results), 9)
        for strategy in run.strategies:
            overall = run.summaries[strategy]["overall"]
            self.assertIn("hit_at_3", overall)
            self.assertIn("recall_at_3", overall)
            self.assertIn("false_positive_rate", overall)

    def test_agentic_case_records_bounded_trace(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "agent-rewrite-card-001"
        )

        run = EvaluationRunner().run((case,), strategies=("hybrid",))
        result = run.results[0]

        self.assertTrue(result.rewrite_actual)
        self.assertEqual(result.retrieval_attempts, 2)
        self.assertEqual(result.grade_actual, "relevant")
        self.assertIn("rewrite", result.trace_events)

    def test_conditional_rewrite_is_not_required_when_initial_hits_are_relevant(
        self,
    ) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "agent-rewrite-card-003"
        )

        class InitiallyRelevantRAG(LocalRAG):
            def retrieve(self, query, top_k=3, *, strategy="hybrid", filters=None):
                if query == case.query:
                    chunk = next(
                        item
                        for item in self.chunks
                        if item.chunk_id == "campus_card-1-0"
                    )
                    return (
                        RetrievalHit(
                            chunk=chunk,
                            score=0.8,
                            rank=1,
                            vector_score=0.8,
                            retrieval_method="vector",
                            vector_rank=1,
                        ),
                    )
                return super().retrieve(
                    query,
                    top_k=top_k,
                    strategy=strategy,
                    filters=filters,
                )

        runner = EvaluationRunner(InitiallyRelevantRAG())

        result = runner.run((case,), strategies=("hybrid",)).results[0]

        self.assertEqual(result.rewrite_policy, "if_initial_insufficient")
        self.assertEqual(result.initial_grade_actual, "relevant")
        self.assertFalse(result.rewrite_expected)
        self.assertFalse(result.rewrite_actual)
        self.assertNotIn("rewrite", result.trace_events)
        self.assertTrue(result.passed)

    def test_injection_metric_is_explicitly_not_evaluated(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "agent-injection-library-001"
        )

        run = EvaluationRunner().run((case,), strategies=("hybrid",))
        overall = run.summaries["hybrid"]["overall"]

        self.assertIsNone(overall["injection_containment_rate"])
        self.assertEqual(overall["injection_containment_status"], "not_evaluated")
        self.assertEqual(overall["injection_case_count"], 1)
        self.assertIsNone(run.results[0].injection_contained)
        self.assertNotIn("injection_escape_rate", overall)

    def test_safe_abstention_is_judged_from_real_answer_output(self) -> None:
        class ControlledAnswerRAG(LocalRAG):
            def __init__(self, answer_text: str) -> None:
                self.answer_text = answer_text
                self.answer_calls = 0
                super().__init__()

            def answer(self, query, top_k=3, *, filters=None, strategy="hybrid"):
                self.answer_calls += 1
                return RAGAnswer(
                    query=query,
                    answer=self.answer_text,
                    hits=self.retrieve(
                        query,
                        top_k=top_k,
                        filters=filters,
                        strategy=strategy,
                    ),
                )

            @staticmethod
            def missing_precise_fact(query, hits):
                raise AssertionError("评测器不应直接调用内部缺失事实规则")

        case = next(case for case in self.cases if case.case_id == "neg-card-001")
        safe_rag = ControlledAnswerRAG("资料中没有提供该费用，为避免编造请确认。")
        unsafe_rag = ControlledAnswerRAG("校园卡补办收费20元。")

        safe_result = EvaluationRunner(safe_rag).run(
            (case,), strategies=("bm25",)
        ).results[0]
        unsafe_result = EvaluationRunner(unsafe_rag).run(
            (case,), strategies=("bm25",)
        ).results[0]

        self.assertEqual(safe_rag.answer_calls, 1)
        self.assertTrue(safe_result.safe_abstained)
        self.assertEqual(safe_result.generated_answer, safe_rag.answer_text)
        self.assertFalse(unsafe_result.safe_abstained)
        self.assertIn("20元", unsafe_result.forbidden_fact_violations)

    def test_summary_is_scoped_and_run_inputs_are_fingerprinted(self) -> None:
        cases = tuple(
            case
            for case in self.cases
            if case.case_id in {
                "meta-doc-library-001",
                "agent-direct-transcript-001",
            }
        )

        run = EvaluationRunner().run(cases, strategies=("bm25",))
        repeated = EvaluationRunner().run(cases, strategies=("bm25",))
        summary = run.summaries["bm25"]

        self.assertEqual(set(summary["by_split"]), {"dev", "test"})
        self.assertEqual(set(summary["by_task"]), {"retrieval", "agentic"})
        self.assertEqual(summary["by_split"]["dev"]["cases"], 1)
        self.assertEqual(summary["by_task"]["agentic"]["cases"], 1)
        self.assertEqual(len(run.configuration_fingerprint), 64)
        self.assertEqual(len(run.corpus_fingerprint), 64)
        self.assertEqual(len(run.dataset_fingerprint), 64)
        self.assertEqual(
            run.configuration_fingerprint,
            repeated.configuration_fingerprint,
        )
        self.assertEqual(run.corpus_fingerprint, repeated.corpus_fingerprint)
        self.assertEqual(run.dataset_fingerprint, repeated.dataset_fingerprint)

    def test_bm25_default_is_independent_of_embedding_provider_name(self) -> None:
        class SentenceNamedHashingProvider(HashingEmbeddingProvider):
            name = "sentence-transformers"

        query = "图书借阅 火星基地"
        hashing_rag = LocalRAG(embedding_provider=HashingEmbeddingProvider())
        renamed_rag = LocalRAG(embedding_provider=SentenceNamedHashingProvider())

        hashing_ids = tuple(
            hit.chunk.chunk_id
            for hit in hashing_rag.retrieve(query, strategy="bm25")
        )
        renamed_ids = tuple(
            hit.chunk.chunk_id
            for hit in renamed_rag.retrieve(query, strategy="bm25")
        )

        self.assertTrue(hashing_ids)
        self.assertEqual(hashing_ids, renamed_ids)

    def test_report_contains_summary_results_and_failures(self) -> None:
        cases = tuple(
            case
            for case in self.cases
            if case.case_id in {"pos-card-loss-001", "neg-card-001"}
        )
        run = EvaluationRunner().run(cases, strategies=("bm25",))
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_report(run, Path(temporary))

            self.assertEqual(set(paths), {"summary", "markdown", "results", "failures"})
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["schema_version"], "1.2")
            self.assertIn("configuration_fingerprint", summary)
            self.assertIn("corpus_fingerprint", summary)
            self.assertIn("dataset_fingerprint", summary)
            self.assertIn("embedding_device", summary)
            self.assertIn("embedding_runtime_versions", summary)
            self.assertIn("embedding_minimum_similarity", summary)
            self.assertIn("embedding_query_prompt", summary)
            self.assertEqual(
                len(paths["results"].read_text(encoding="utf-8").splitlines()),
                2,
            )
            self.assertIn("策略对比", paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
