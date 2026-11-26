from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campus_agent.evaluation.metrics import CaseResult
from campus_agent.evaluation.models import load_dataset
from campus_agent.evaluation.ragas_eval import score_retrieval
from scripts.evaluate_ragas import _gate


class RagasAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_answerable_and_excludes_negative_cases(self) -> None:
        cases = load_dataset(Path(__file__).resolve().parent.parent / "eval" / "datasets")
        positive = next(case for case in cases if case.expected.answerable)
        no_hit = next(case for case in cases if case.retrieval_expectation == "no_hit")
        expected_id = positive.expected.relevant_chunk_ids[0]
        results = (
            self._result(positive, (expected_id, "irrelevant")),
            self._result(no_hit, ()),
        )

        class Sample:
            def __init__(self, **values):
                self.values = values

        class Precision:
            async def single_turn_ascore(self, sample):
                found = set(sample.values["retrieved_context_ids"])
                relevant = set(sample.values["reference_context_ids"])
                return len(found & relevant) / len(found) if found else math.nan

        class Recall:
            async def single_turn_ascore(self, sample):
                found = set(sample.values["retrieved_context_ids"])
                relevant = set(sample.values["reference_context_ids"])
                return len(found & relevant) / len(relevant)

        with patch(
            "campus_agent.evaluation.ragas_eval._ragas_retrieval_metrics",
            return_value=(Sample, Precision(), Recall()),
        ):
            report = await score_retrieval((positive, no_hit), results)

        summary = report["summaries"]["hybrid"]
        self.assertEqual(summary["context_precision"], 0.5)
        self.assertEqual(summary["context_recall"], round(1 / len(positive.expected.relevant_chunk_ids), 6))
        self.assertEqual(summary["no_hit_false_positive_rate"], 0.0)
        self.assertEqual(report["rows"][1]["context_precision"], None)
        self.assertEqual(report["rows"][1]["context_recall"], None)

    async def test_empty_retrieval_is_na_precision_and_zero_recall(self) -> None:
        cases = load_dataset(Path(__file__).resolve().parent.parent / "eval" / "datasets")
        positive = next(case for case in cases if case.expected.answerable)
        result = self._result(positive, ())

        class Sample:
            def __init__(self, **values):
                self.values = values

        class Precision:
            async def single_turn_ascore(self, sample):
                return math.nan

        class Recall:
            async def single_turn_ascore(self, sample):
                return 0.0

        with patch(
            "campus_agent.evaluation.ragas_eval._ragas_retrieval_metrics",
            return_value=(Sample, Precision(), Recall()),
        ):
            report = await score_retrieval((positive,), (result,))
        summary = report["summaries"]["hybrid"]
        self.assertIsNone(summary["context_precision"])
        self.assertEqual(summary["context_precision_na"], 1)
        self.assertEqual(summary["context_recall"], 0.0)

    @staticmethod
    def _result(case, retrieved_ids) -> CaseResult:
        return CaseResult(
            case_id=case.case_id,
            task=case.task,
            split=case.split,
            strategy="hybrid",
            query=case.query,
            tags=case.tags,
            difficulty=case.difficulty,
            expected_domain=case.expected.domain,
            expected_answerable=case.expected.answerable,
            retrieval_expectation=case.retrieval_expectation,
            expected_chunk_ids=case.expected.relevant_chunk_ids,
            expected_document_ids=case.expected.relevant_document_ids,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_document_ids=(),
            retrieved_sources=(),
            duration_ms=0.0,
        )


class RagasGateTests(unittest.TestCase):
    def test_unconfigured_is_not_reported_as_passed(self) -> None:
        summary = {"hybrid": {"context_precision": 0.5, "context_recall": 1.0, "no_hit_false_positive_rate": 0.0}}
        args = SimpleNamespace(min_context_precision=None, min_context_recall=None, max_no_hit_false_positive_rate=None, min_faithfulness=None, min_answer_relevancy=None)
        self.assertEqual(_gate(summary, args)["status"], "not_configured")

    def test_threshold_failure_is_reported(self) -> None:
        summary = {"hybrid": {"context_precision": 0.5, "context_recall": 1.0, "no_hit_false_positive_rate": 0.0}}
        args = SimpleNamespace(min_context_precision=0.6, min_context_recall=0.9, max_no_hit_false_positive_rate=0.0, min_faithfulness=None, min_answer_relevancy=None)
        self.assertEqual(_gate(summary, args)["status"], "failed")
