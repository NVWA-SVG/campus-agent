from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from campus_agent.evaluation.models import load_dataset
from campus_agent.evaluation.validator import validate_dataset
from campus_agent.rag import LocalRAG


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "eval" / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description="校验评测JSONL与知识库Ground Truth")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    try:
        cases = load_dataset(args.dataset_dir)
        rag = LocalRAG()
        validate_dataset(cases, rag.chunks)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    payload = {
        "valid": True,
        "cases": len(cases),
        "tasks": dict(sorted(Counter(case.task for case in cases).items())),
        "splits": dict(sorted(Counter(case.split for case in cases).items())),
        "difficulties": dict(
            sorted(Counter(case.difficulty for case in cases).items())
        ),
        "answerable": dict(
            sorted(Counter(str(case.expected.answerable) for case in cases).items())
        ),
        "retrieval_expectations": dict(
            sorted(Counter(case.retrieval_expectation for case in cases).items())
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
