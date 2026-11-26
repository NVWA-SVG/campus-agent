"""Run RAGAS ID-based retrieval metrics and optional LLM-judged answer metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from campus_agent.evaluation.models import load_dataset
from campus_agent.evaluation.ragas_eval import score_retrieval, score_rule_answers
from campus_agent.evaluation.runner import EvaluationRunner, SUPPORTED_STRATEGIES


ROOT = Path(__file__).resolve().parent.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGAS 评测：离线检索 + 可选模型评委")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "eval" / "datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "reports")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--strategies", nargs="+", choices=SUPPORTED_STRATEGIES, default=["hybrid"]
    )
    parser.add_argument("--min-context-precision", type=float)
    parser.add_argument("--min-context-recall", type=float)
    parser.add_argument("--max-no-hit-false-positive-rate", type=float)
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--min-answer-relevancy", type=float)
    parser.add_argument("--judge", action="store_true", help="调用 DeepSeek 评估规则回答，产生 API 费用")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--max-judge-cases", type=int)
    return parser


def _gate(
    summaries: dict[str, dict[str, object]],
    args: argparse.Namespace,
    generation: dict[str, object] | None = None,
) -> dict[str, object]:
    gates: list[dict[str, object]] = []
    specs = (
        ("context_precision", args.min_context_precision, ">="),
        ("context_recall", args.min_context_recall, ">="),
        ("no_hit_false_positive_rate", args.max_no_hit_false_positive_rate, "<="),
    )
    for strategy, summary in summaries.items():
        for name, threshold, operator in specs:
            if threshold is None:
                continue
            actual = summary[name]
            passed = actual is not None and (
                float(actual) >= threshold if operator == ">=" else float(actual) <= threshold
            )
            gates.append(
                {
                    "strategy": strategy,
                    "metric": name,
                    "actual": actual,
                    "operator": operator,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
        if any(threshold is not None for _, threshold, _ in specs):
            for name, passed in (
                ("legacy_failures", summary.get("legacy_failures", 0) == 0),
                ("missing_fact_safe_abstentions", summary.get("missing_fact_safe_abstentions", 0) == summary.get("missing_fact_cases", 0)),
            ):
                gates.append({"strategy": strategy, "metric": name, "passed": passed})
    for name, threshold in (
        ("faithfulness", args.min_faithfulness),
        ("answer_relevancy", args.min_answer_relevancy),
    ):
        if threshold is None:
            continue
        actual = generation.get(name) if generation else None
        gates.append({
            "strategy": "rule_answer_hybrid",
            "metric": name,
            "actual": actual,
            "operator": ">=",
            "threshold": threshold,
            "passed": actual is not None and float(actual) >= threshold,
        })
    return {
        "status": "not_configured" if not gates else "passed" if all(g["passed"] for g in gates) else "failed",
        "checks": gates,
    }


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    for name in (
        "min_context_precision",
        "min_context_recall",
        "max_no_hit_false_positive_rate",
        "min_faithfulness",
        "min_answer_relevancy",
    ):
        threshold = getattr(args, name)
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"{name} 必须在 0 到 1 之间")
    if args.max_judge_cases is not None and args.max_judge_cases < 1:
        raise ValueError("max-judge-cases 必须大于 0")
    if args.judge and not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("--judge 需要 DEEPSEEK_API_KEY；去掉 --judge 可离线运行")
    if not args.judge and (args.min_faithfulness is not None or args.min_answer_relevancy is not None):
        raise ValueError("生成指标阈值需要同时设置 --judge")

    cases = load_dataset(args.dataset_dir)
    selected = tuple(case for case in cases if case.split == args.split)
    runner = EvaluationRunner()
    run = runner.run(selected, strategies=tuple(args.strategies))
    retrieval = await score_retrieval(selected, run.results)
    generation: dict[str, object] = {
        "status": "not_run",
        "reason": "需要显式 --judge、DEEPSEEK_API_KEY 和本地语义 Embedding",
    }
    if args.judge:
        scored = await score_rule_answers(
            selected,
            runner.rag,
            judge_model=args.judge_model,
            max_cases=args.max_judge_cases,
        )
        generation = {"status": "completed", **scored}

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ragas_version": version("ragas"),
        "split": args.split,
        "dataset_fingerprint": run.dataset_fingerprint,
        "corpus_fingerprint": run.corpus_fingerprint,
        "embedding_fingerprint": run.embedding_fingerprint,
        "embedding_provider": run.embedding_provider,
        "vector_status": run.vector_status,
        "case_count": len(selected),
        "strategies": list(run.strategies),
        "retrieval": retrieval,
        "generation": generation,
        "gate": _gate(retrieval["summaries"], args, generation),
    }


def main() -> None:
    args = _parser().parse_args()
    try:
        report = asyncio.run(evaluate(args))
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"ragas-{args.split}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "case_count": report["case_count"],
        "summaries": report["retrieval"]["summaries"],
        "generation_status": report["generation"]["status"],
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))
    if report["gate"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
