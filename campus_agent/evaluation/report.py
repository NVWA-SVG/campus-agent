"""把一次评测写成机器可读 JSON/JSONL 与面试可读 Markdown。"""

from __future__ import annotations

import json
from pathlib import Path

from campus_agent.evaluation.runner import EvaluationRun


def write_report(run: EvaluationRun, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    results_path = output_dir / "case_results.jsonl"
    failures_path = output_dir / "failures.jsonl"

    summary_path.write_text(
        json.dumps(run.as_dict(include_results=False), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results_path.write_text(
        "".join(
            json.dumps(result.as_dict(), ensure_ascii=False) + "\n"
            for result in run.results
        ),
        encoding="utf-8",
    )
    failures_path.write_text(
        "".join(
            json.dumps(result.as_dict(), ensure_ascii=False) + "\n"
            for result in run.results
            if not result.passed
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(run), encoding="utf-8")
    return {
        "summary": summary_path,
        "markdown": markdown_path,
        "results": results_path,
        "failures": failures_path,
    }


def write_baseline(run: EvaluationRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.as_dict(include_results=False), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _markdown(run: EvaluationRun) -> str:
    def percentage(value: object) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    lines = [
        "# Campus Agent RAG 评测报告",
        "",
        f"- 用例数：{run.case_count}",
        f"- Embedding Provider：`{run.embedding_provider}`",
        f"- Embedding Model：`{run.embedding_model}`",
        f"- Model Revision：`{run.embedding_revision or 'N/A'}`",
        f"- Model Fingerprint：`{run.embedding_fingerprint}`",
        f"- Device：`{run.embedding_device}`",
        f"- Minimum Similarity：`{run.embedding_minimum_similarity:g}`",
        f"- Query Prompt：`{run.embedding_query_prompt or 'N/A'}`",
        f"- Dataset Fingerprint：`{run.dataset_fingerprint}`",
        f"- Corpus Fingerprint：`{run.corpus_fingerprint}`",
        f"- Configuration Fingerprint：`{run.configuration_fingerprint}`",
        "- Runtime Versions：`{}`".format(
            ", ".join(
                f"{name}={version}"
                for name, version in run.embedding_runtime_versions.items()
            )
            or "N/A"
        ),
        f"- Vector状态：`{run.vector_status}`",
        f"- 生成时间：{run.generated_at}",
        "- Prompt Injection：`not_evaluated`（当前离线评测未执行真实工具路由）",
        "",
        "## 策略对比",
        "",
        f"| 策略 | Hit@1 | Hit@{run.metric_k} | Recall@{run.metric_k} | MRR@{run.metric_k} | No-hit | FPR | Hard-negative安全拒答 | 失败数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in run.strategies:
        overall = run.summaries[strategy]["overall"]
        lines.append(
            "| {strategy} | {hit1} | {hit3} | {recall3} | "
            "{mrr3} | {no_hit} | {fpr} | {safe} | {failed} |".format(
                strategy=strategy,
                hit1=percentage(overall["hit_at_1"]),
                hit3=percentage(overall[f"hit_at_{run.metric_k}"]),
                recall3=percentage(overall[f"recall_at_{run.metric_k}"]),
                mrr3=percentage(overall[f"mrr_at_{run.metric_k}"]),
                no_hit=percentage(overall["no_hit_accuracy"]),
                fpr=percentage(overall["false_positive_rate"]),
                safe=percentage(overall["hard_negative_safe_abstention_rate"]),
                failed=overall["failed_cases"],
            )
        )
    lines.extend(
        [
            "",
            "## 分口径结果",
            "",
            "全量结果同时包含 dev/test 和 retrieval/agentic；发布判断应优先查看对应口径。",
            "",
            f"| 策略 | 口径 | 用例数 | Hit@{run.metric_k} | No-hit | Hard-negative安全拒答 | 失败数 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in run.strategies:
        summary = run.summaries[strategy]
        for group_name, label_prefix in (("by_split", "split"), ("by_task", "task")):
            grouped = summary[group_name]
            for scope, metrics in grouped.items():
                lines.append(
                    "| {strategy} | {label}:{scope} | {cases} | {hit} | "
                    "{no_hit} | {safe} | {failed} |".format(
                        strategy=strategy,
                        label=label_prefix,
                        scope=scope,
                        cases=metrics["cases"],
                        hit=percentage(metrics[f"hit_at_{run.metric_k}"]),
                        no_hit=percentage(metrics["no_hit_accuracy"]),
                        safe=percentage(
                            metrics["hard_negative_safe_abstention_rate"]
                        ),
                        failed=metrics["failed_cases"],
                    )
                )
    lines.extend(
        [
            "",
            "## 失败用例",
            "",
            "失败详情位于 `failures.jsonl`，每条记录保留预期Chunk、实际排序和失败原因。",
            "",
        ]
    )
    return "\n".join(lines)
