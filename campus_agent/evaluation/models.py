"""评测数据契约与 JSONL 加载器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


TaskType = Literal["retrieval", "agentic"]
SplitType = Literal["dev", "test"]
Difficulty = Literal["easy", "medium", "hard"]


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field_name} 必须是非空字符串数组")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} 不能包含重复值")
    return normalized


def _strict_mapping(
    value: object,
    *,
    field_name: str,
    allowed: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是对象")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field_name} 包含未知字段：{sorted(unknown)}")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    top_k: int = 3
    filters: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluationRequest":
        if value is None:
            return cls()
        raw = _strict_mapping(
            value,
            field_name="request",
            allowed={"top_k", "filters"},
        )
        top_k = raw.get("top_k", 3)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("request.top_k 必须是 1 到 20 的整数")
        filters = raw.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError("request.filters 必须是对象")
        normalized_filters: dict[str, str] = {}
        for key, item in filters.items():
            if not isinstance(key, str) or not isinstance(item, str) or not item.strip():
                raise ValueError("request.filters 的键和值必须是非空字符串")
            normalized_filters[key] = item.strip()
        return cls(top_k=top_k, filters=normalized_filters)

    def as_dict(self) -> dict[str, object]:
        return {"top_k": self.top_k, "filters": dict(self.filters)}


@dataclass(frozen=True, slots=True)
class EvaluationExpected:
    answerable: bool
    relevant_document_ids: tuple[str, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()
    domain: str | None = None
    category: str | None = None
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> "EvaluationExpected":
        raw = _strict_mapping(
            value,
            field_name="expected",
            allowed={
                "answerable",
                "relevant_document_ids",
                "relevant_chunk_ids",
                "domain",
                "category",
                "required_facts",
                "forbidden_facts",
            },
        )
        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("expected.answerable 必须是布尔值")
        domain = raw.get("domain")
        category = raw.get("category")
        for name, item in (("domain", domain), ("category", category)):
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"expected.{name} 必须是非空字符串或 null")
        return cls(
            answerable=answerable,
            relevant_document_ids=_string_tuple(
                raw.get("relevant_document_ids"),
                "expected.relevant_document_ids",
            ),
            relevant_chunk_ids=_string_tuple(
                raw.get("relevant_chunk_ids"),
                "expected.relevant_chunk_ids",
            ),
            domain=domain.strip() if isinstance(domain, str) else None,
            category=category.strip() if isinstance(category, str) else None,
            required_facts=_string_tuple(
                raw.get("required_facts"),
                "expected.required_facts",
            ),
            forbidden_facts=_string_tuple(
                raw.get("forbidden_facts"),
                "expected.forbidden_facts",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "answerable": self.answerable,
            "relevant_document_ids": list(self.relevant_document_ids),
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "domain": self.domain,
            "category": self.category,
            "required_facts": list(self.required_facts),
            "forbidden_facts": list(self.forbidden_facts),
        }


@dataclass(frozen=True, slots=True)
class AgenticExpectation:
    rewrite_expected: bool | Literal["if_initial_insufficient"] | None = None
    rewritten_query_contains: tuple[str, ...] = ()
    grade_expected: Literal["relevant", "insufficient"] | None = None
    verification_expected: Literal[
        "grounded", "insufficient_evidence", "unsupported"
    ] | None = None
    max_retrieval_attempts: int = 2
    required_events: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> "AgenticExpectation | None":
        if value is None:
            return None
        raw = _strict_mapping(
            value,
            field_name="agentic",
            allowed={
                "rewrite_expected",
                "rewritten_query_contains",
                "grade_expected",
                "verification_expected",
                "max_retrieval_attempts",
                "required_events",
                "forbidden_tools",
            },
        )
        rewrite_expected = raw.get("rewrite_expected")
        if rewrite_expected not in {None, True, False, "if_initial_insufficient"}:
            raise ValueError(
                "agentic.rewrite_expected 必须是布尔值、"
                "if_initial_insufficient 或 null"
            )
        grade_expected = raw.get("grade_expected")
        if grade_expected not in {None, "relevant", "insufficient"}:
            raise ValueError("agentic.grade_expected 值无效")
        verification_expected = raw.get("verification_expected")
        if verification_expected not in {
            None,
            "grounded",
            "insufficient_evidence",
            "unsupported",
        }:
            raise ValueError("agentic.verification_expected 值无效")
        attempts = raw.get("max_retrieval_attempts", 2)
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise ValueError("agentic.max_retrieval_attempts 必须是正整数")
        return cls(
            rewrite_expected=rewrite_expected,
            rewritten_query_contains=_string_tuple(
                raw.get("rewritten_query_contains"),
                "agentic.rewritten_query_contains",
            ),
            grade_expected=grade_expected,
            verification_expected=verification_expected,
            max_retrieval_attempts=attempts,
            required_events=_string_tuple(
                raw.get("required_events"),
                "agentic.required_events",
            ),
            forbidden_tools=_string_tuple(
                raw.get("forbidden_tools"),
                "agentic.forbidden_tools",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rewrite_expected": self.rewrite_expected,
            "rewritten_query_contains": list(self.rewritten_query_contains),
            "grade_expected": self.grade_expected,
            "verification_expected": self.verification_expected,
            "max_retrieval_attempts": self.max_retrieval_attempts,
            "required_events": list(self.required_events),
            "forbidden_tools": list(self.forbidden_tools),
        }

    def resolve_rewrite_expected(self, initial_grade: str) -> bool | None:
        if self.rewrite_expected == "if_initial_insufficient":
            return initial_grade == "insufficient"
        if isinstance(self.rewrite_expected, bool):
            return self.rewrite_expected
        return None


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    schema_version: str
    task: TaskType
    split: SplitType
    query: str
    request: EvaluationRequest
    expected: EvaluationExpected
    tags: tuple[str, ...]
    difficulty: Difficulty
    review_status: Literal["human_checked"]
    agentic: AgenticExpectation | None = None
    source_file: str = ""
    source_line: int = 0

    @property
    def retrieval_expectation(self) -> Literal["hit", "no_hit", "topic_hit_allowed"]:
        """区分检索拒答与“可召回主题但缺少答案事实”。"""

        if self.expected.answerable:
            return "hit"
        if "retrieval_no_hit" in self.tags:
            return "no_hit"
        return "topic_hit_allowed"

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        source_file: str = "",
        source_line: int = 0,
    ) -> "EvaluationCase":
        raw = _strict_mapping(
            value,
            field_name="case",
            allowed={
                "id",
                "schema_version",
                "task",
                "split",
                "query",
                "request",
                "expected",
                "tags",
                "difficulty",
                "review_status",
                "agentic",
                "retrieval_expectation",
            },
        )
        case_id = raw.get("id")
        query = raw.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("id 必须是非空字符串")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        schema_version = raw.get("schema_version")
        if schema_version != "1.0":
            raise ValueError("schema_version 只支持 1.0")
        task = raw.get("task")
        if task not in {"retrieval", "agentic"}:
            raise ValueError("task 只支持 retrieval 或 agentic")
        split = raw.get("split")
        if split not in {"dev", "test"}:
            raise ValueError("split 只支持 dev 或 test")
        difficulty = raw.get("difficulty")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty 只支持 easy、medium 或 hard")
        if raw.get("review_status") != "human_checked":
            raise ValueError("review_status 必须是 human_checked")
        agentic = AgenticExpectation.from_mapping(raw.get("agentic"))
        if task == "agentic" and agentic is None:
            raise ValueError("agentic 任务必须提供 agentic 期望")
        if task == "retrieval" and agentic is not None:
            raise ValueError("retrieval 任务不能提供 agentic 期望")
        case = cls(
            case_id=case_id.strip(),
            schema_version="1.0",
            task=task,
            split=split,
            query=query.strip(),
            request=EvaluationRequest.from_mapping(raw.get("request")),
            expected=EvaluationExpected.from_mapping(raw.get("expected")),
            tags=_string_tuple(raw.get("tags"), "tags"),
            difficulty=difficulty,
            review_status="human_checked",
            agentic=agentic,
            source_file=source_file,
            source_line=source_line,
        )
        declared_expectation = raw.get("retrieval_expectation")
        if (
            declared_expectation is not None
            and declared_expectation != case.retrieval_expectation
        ):
            raise ValueError(
                "retrieval_expectation 与 answerable/tags 推导结果不一致"
            )
        return case

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.case_id,
            "schema_version": self.schema_version,
            "task": self.task,
            "split": self.split,
            "query": self.query,
            "request": self.request.as_dict(),
            "expected": self.expected.as_dict(),
            "tags": list(self.tags),
            "difficulty": self.difficulty,
            "review_status": self.review_status,
            "retrieval_expectation": self.retrieval_expectation,
        }
        if self.agentic is not None:
            result["agentic"] = self.agentic.as_dict()
        return result


def discover_dataset_files(dataset_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(dataset_dir.glob("*.jsonl")))


def load_dataset(paths: Path | list[Path] | tuple[Path, ...]) -> tuple[EvaluationCase, ...]:
    if isinstance(paths, Path):
        selected = discover_dataset_files(paths) if paths.is_dir() else (paths,)
    else:
        selected = tuple(paths)
    if not selected:
        raise ValueError("没有找到评测 JSONL 文件")

    cases: list[EvaluationCase] = []
    for path in selected:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    value = json.loads(line)
                    cases.append(
                        EvaluationCase.from_mapping(
                            value,
                            source_file=str(path),
                            source_line=line_number,
                        )
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not cases:
        raise ValueError("评测数据集不能为空")
    return tuple(cases)
