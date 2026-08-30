from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.calibrate_embedding import (
    DEFAULT_THRESHOLDS,
    main,
    recommend_threshold,
)


def _row(
    threshold: float,
    *,
    hit_at_1: float,
    hit_at_3: float,
    no_hit: float,
    fpr: float,
) -> dict[str, object]:
    metrics = {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "no_hit_accuracy": no_hit,
        "false_positive_rate": fpr,
    }
    return {
        "threshold": threshold,
        "strategies": {"vector": dict(metrics), "hybrid": dict(metrics)},
    }


class CalibrationSelectionTests(unittest.TestCase):
    def test_default_grid_contains_the_production_candidate(self) -> None:
        self.assertIn(0.48, DEFAULT_THRESHOLDS)

    def test_safety_gate_precedes_recall_and_ties_choose_higher_threshold(self) -> None:
        rows = [
            _row(0.44, hit_at_1=1.0, hit_at_3=1.0, no_hit=0.8, fpr=0.2),
            _row(0.48, hit_at_1=0.95, hit_at_3=0.99, no_hit=0.9, fpr=0.1),
            _row(0.50, hit_at_1=0.95, hit_at_3=0.99, no_hit=0.9, fpr=0.1),
        ]

        result = recommend_threshold(rows, minimum_no_hit_accuracy=0.9)

        self.assertTrue(result["production_gate_passed"])
        self.assertEqual(result["recommended_threshold"], 0.50)
        self.assertEqual(result["mode"], "meets_safety_gate")

    def test_failed_gate_is_explicit_and_prefers_best_no_hit_observation(self) -> None:
        rows = [
            _row(0.44, hit_at_1=1.0, hit_at_3=1.0, no_hit=0.6, fpr=0.4),
            _row(0.55, hit_at_1=0.8, hit_at_3=0.9, no_hit=0.8, fpr=0.2),
        ]

        result = recommend_threshold(rows, minimum_no_hit_accuracy=0.9)

        self.assertFalse(result["production_gate_passed"])
        self.assertEqual(result["recommended_threshold"], 0.55)
        self.assertEqual(result["mode"], "best_observed_gate_not_met")

    def test_cli_refuses_to_calibrate_hashing_provider(self) -> None:
        provider = SimpleNamespace(name="local-hashing")
        with patch(
            "scripts.calibrate_embedding.build_embedding_provider_from_environment",
            return_value=provider,
        ):
            with self.assertRaisesRegex(SystemExit, "只允许真实 Sentence Transformers"):
                main(["--thresholds", "0.48"])

    def test_cli_saves_complete_dev_artifact(self) -> None:
        provider = SimpleNamespace(
            name="sentence-transformers",
            minimum_similarity=0.48,
            runtime_metadata={
                "device": "cpu",
                "versions": {"sentence-transformers": "5.5.1"},
            },
        )
        rag = SimpleNamespace(
            vector_status="ready",
            vector_degraded_reason=None,
            embedding_provider_name="sentence-transformers",
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            embedding_revision="locked-revision",
            embedding_fingerprint="fingerprint1234",
            embedding_device="cpu",
            embedding_runtime_versions={
                "sentence-transformers": "5.5.1",
                "torch": "2.12.0",
            },
            embedding_query_prompt="检索：",
        )
        cases = (
            SimpleNamespace(
                case_id="dev-1",
                split="dev",
                as_dict=lambda: {"id": "dev-1", "split": "dev"},
            ),
            SimpleNamespace(
                case_id="test-1",
                split="test",
                as_dict=lambda: {"id": "test-1", "split": "test"},
            ),
        )

        class FakeRunner:
            def __init__(self, value) -> None:
                self.rag = value

            def run(self, cases, *, strategies, split):
                threshold = provider.minimum_similarity
                no_hit = 0.9 if threshold >= 0.48 else 0.8
                hit_at_3 = 0.98 if threshold == 0.48 else 0.95
                overall = {
                    "hit_at_1": 0.94,
                    "hit_at_3": hit_at_3,
                    "recall_at_3": hit_at_3,
                    "mrr_at_3": 0.96,
                    "no_hit_accuracy": no_hit,
                    "false_positive_rate": 1.0 - no_hit,
                    "hard_negative_safe_abstention_rate": 1.0,
                    "failed_cases": 1,
                }
                return SimpleNamespace(
                    summaries={name: {"overall": overall} for name in strategies}
                )

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "calibration.json"
            with (
                patch(
                    "scripts.calibrate_embedding.build_embedding_provider_from_environment",
                    return_value=provider,
                ),
                patch("scripts.calibrate_embedding.LocalRAG", return_value=rag),
                patch("scripts.calibrate_embedding.EvaluationRunner", FakeRunner),
                patch("scripts.calibrate_embedding.load_dataset", return_value=cases),
                redirect_stdout(io.StringIO()),
            ):
                result = main(
                    [
                        "--thresholds",
                        "0.46",
                        "0.48",
                        "--output",
                        str(output_path),
                    ]
                )
            artifact = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(artifact["split"], "dev")
        self.assertEqual(artifact["dataset"]["case_count"], 1)
        self.assertEqual(len(artifact["dataset"]["sha256"]), 64)
        self.assertEqual(artifact["embedding"]["device"], "cpu")
        self.assertEqual(
            artifact["embedding"]["runtime_versions"]["torch"],
            "2.12.0",
        )
        self.assertEqual(
            artifact["recommendation"]["recommended_threshold"],
            0.48,
        )
        self.assertTrue(artifact["recommendation"]["production_gate_passed"])


if __name__ == "__main__":
    unittest.main()
