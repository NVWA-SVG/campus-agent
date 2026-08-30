"""在 dev split 上校准真实语义检索阈值并保存可审计产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from campus_agent.evaluation.models import EvaluationCase, load_dataset
from campus_agent.evaluation.runner import EvaluationRunner
from campus_agent.rag.embeddings import build_embedding_provider_from_environment
from campus_agent.rag.service import LocalRAG


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 dev 集校准真实 Embedding 阈值")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--minimum-no-hit-accuracy",
        type=float,
        default=0.90,
        help="进入生产推荐候选的最低 dev No-hit Accuracy，默认0.90",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "eval" / "datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="校准产物路径；默认写入 eval/calibration/<model>-<fingerprint>-dev.json",
    )
    return parser


def _dataset_digest(cases: tuple[EvaluationCase, ...]) -> str:
    canonical = json.dumps(
        [case.as_dict() for case in sorted(cases, key=lambda item: item.case_id)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _number(
    row: dict[str, object],
    strategy: str,
    metric: str,
    *,
    default: float = -1.0,
) -> float:
    strategies = row["strategies"]
    assert isinstance(strategies, dict)
    summary = strategies[strategy]
    assert isinstance(summary, dict)
    value = summary.get(metric)
    return float(value) if value is not None else default


def recommend_threshold(
    rows: list[dict[str, object]],
    *,
    minimum_no_hit_accuracy: float,
) -> dict[str, object]:
    """先满足安全门槛，再按召回、误报与保守阈值确定唯一推荐。"""

    eligible = [
        row
        for row in rows
        if _number(row, "hybrid", "no_hit_accuracy") >= minimum_no_hit_accuracy
    ]
    gate_passed = bool(eligible)
    candidates = eligible or rows
    if gate_passed:
        # 通过安全门槛后依次最大化 Hit@3、Hit@1、No-hit，最小化FPR；
        # 全部相同时选择更高阈值，减少未知问题的偶然命中。
        ranking = lambda row: (
            _number(row, "hybrid", "hit_at_3"),
            _number(row, "hybrid", "hit_at_1"),
            _number(row, "hybrid", "no_hit_accuracy"),
            -_number(row, "hybrid", "false_positive_rate", default=1.0),
            float(row["threshold"]),
        )
        mode = "meets_safety_gate"
    else:
        # 无候选通过门槛时优先选择 No-hit 最好的观测点，但明确标记为
        # 非生产就绪，不能把它描述成已通过验收。
        ranking = lambda row: (
            _number(row, "hybrid", "no_hit_accuracy"),
            -_number(row, "hybrid", "false_positive_rate", default=1.0),
            _number(row, "hybrid", "hit_at_3"),
            _number(row, "hybrid", "hit_at_1"),
            float(row["threshold"]),
        )
        mode = "best_observed_gate_not_met"
    winner = max(candidates, key=ranking)
    return {
        "recommended_threshold": float(winner["threshold"]),
        "production_gate_passed": gate_passed,
        "mode": mode,
        "minimum_no_hit_accuracy": minimum_no_hit_accuracy,
        "rule": (
            "先筛选 Hybrid No-hit Accuracy 达到门槛的阈值；再依次最大化 "
            "Hybrid Hit@3、Hit@1、No-hit Accuracy，最小化 FPR，完全并列时"
            "选择更高阈值。若没有阈值通过门槛，只报告最佳观测值并标记非生产就绪。"
        ),
    }


def _default_output_path(rag: LocalRAG) -> Path:
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", rag.embedding_model_name).strip("-")
    return (
        ROOT
        / "eval"
        / "calibration"
        / f"{model_slug}-{rag.embedding_fingerprint}-dev.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = tuple(sorted(set(args.thresholds)))
    if not thresholds or any(not -1.0 <= value <= 1.0 for value in thresholds):
        raise SystemExit("threshold 必须位于 -1 到 1 之间且至少提供一个")
    if not 0.0 <= args.minimum_no_hit_accuracy <= 1.0:
        raise SystemExit("minimum-no-hit-accuracy 必须位于 0 到 1 之间")

    provider = build_embedding_provider_from_environment()
    if provider.name != "sentence-transformers":
        raise SystemExit(
            "阈值校准只允许真实 Sentence Transformers；请先设置 "
            "CAMPUS_EMBEDDING_PROVIDER=sentence-transformers"
        )
    rag = LocalRAG(embedding_provider=provider)
    if rag.vector_status != "ready":
        raise SystemExit(
            f"语义向量索引未就绪：{rag.vector_degraded_reason or 'unknown'}"
        )

    cases = load_dataset(args.dataset_dir)
    dev_cases = tuple(case for case in cases if case.split == "dev")
    if not dev_cases:
        raise SystemExit("评测集中没有 dev 用例")
    runner = EvaluationRunner(rag)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        provider.minimum_similarity = threshold
        run = runner.run(
            cases,
            strategies=("vector", "hybrid"),
            split="dev",
        )
        rows.append(
            {
                "threshold": threshold,
                "strategies": {
                    strategy: {
                        metric: summary["overall"][metric]
                        for metric in (
                            "hit_at_1",
                            "hit_at_3",
                            "recall_at_3",
                            "mrr_at_3",
                            "no_hit_accuracy",
                            "false_positive_rate",
                            "hard_negative_safe_abstention_rate",
                            "failed_cases",
                        )
                    }
                    for strategy, summary in run.summaries.items()
                },
            }
        )

    recommendation = recommend_threshold(
        rows,
        minimum_no_hit_accuracy=args.minimum_no_hit_accuracy,
    )
    artifact: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "dev",
        "dataset": {
            "case_count": len(dev_cases),
            "sha256": _dataset_digest(dev_cases),
        },
        "embedding": {
            "provider": rag.embedding_provider_name,
            "model": rag.embedding_model_name,
            "revision": rag.embedding_revision,
            "fingerprint": rag.embedding_fingerprint,
            "device": rag.embedding_device,
            "runtime_versions": rag.embedding_runtime_versions,
            "runtime_metadata": provider.runtime_metadata,
            "query_prompt": rag.embedding_query_prompt,
            "normalize_embeddings": True,
        },
        "vector_status": rag.vector_status,
        "thresholds": list(thresholds),
        "recommendation": recommendation,
        "rows": rows,
    }
    output_path = args.output or _default_output_path(rag)
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                **artifact,
                "artifact_path": str(output_path.absolute()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
