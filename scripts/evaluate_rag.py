from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_agent.evaluation.models import load_dataset
from campus_agent.evaluation.report import write_baseline, write_report
from campus_agent.evaluation.runner import EvaluationRunner, SUPPORTED_STRATEGIES


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "eval" / "datasets"
REPORT_DIR = ROOT / "eval" / "reports"


def evaluate(
    *,
    dataset_dir: Path = DATASET_DIR,
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES,
    split: str | None = None,
    metric_k: int = 3,
) -> dict[str, object]:
    """运行评测并返回完整摘要；保留函数入口方便测试和Notebook复用。"""

    cases = load_dataset(dataset_dir)
    run = EvaluationRunner().run(
        cases,
        strategies=strategies,
        split=split,
        metric_k=metric_k,
    )
    return run.as_dict(include_results=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Campus Agent 离线RAG评测")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=list(SUPPORTED_STRATEGIES),
    )
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--metric-k", type=int, default=3)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="可选：同时把本次摘要保存为指定基线文件",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    cases = load_dataset(args.dataset_dir)
    run = EvaluationRunner().run(
        cases,
        strategies=tuple(args.strategies),
        split=args.split,
        metric_k=args.metric_k,
    )
    paths = write_report(run, args.output_dir)
    if args.baseline is not None:
        write_baseline(run, args.baseline)
    payload = {
        "schema_version": "1.2",
        "generated_at": run.generated_at,
        "embedding_provider": run.embedding_provider,
        "embedding_model": run.embedding_model,
        "embedding_revision": run.embedding_revision,
        "embedding_fingerprint": run.embedding_fingerprint,
        "embedding_device": run.embedding_device,
        "embedding_runtime_versions": run.embedding_runtime_versions,
        "embedding_minimum_similarity": run.embedding_minimum_similarity,
        "embedding_query_prompt": run.embedding_query_prompt,
        "vector_status": run.vector_status,
        "configuration": run.configuration,
        "configuration_fingerprint": run.configuration_fingerprint,
        "corpus_fingerprint": run.corpus_fingerprint,
        "dataset_fingerprint": run.dataset_fingerprint,
        "case_count": run.case_count,
        "metric_k": run.metric_k,
        "summaries": {
            strategy: run.summaries[strategy]["overall"]
            for strategy in run.strategies
        },
    }
    payload["report_files"] = {name: str(path) for name, path in paths.items()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
