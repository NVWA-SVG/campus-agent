"""进入 Planner 前的确定性输入理解。

这一层只处理高置信度、可解释的校园领域归一化。它不尝试成为通用中文
纠错器，避免擅自修改课程名、业务编号或用户上传资料中的专有名词。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from campus_agent.tools.course import WEEKDAY_ALIASES


CHINA_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass(frozen=True, slots=True)
class InputUnderstanding:
    original_query: str
    normalized_query: str
    course_intent: bool
    course_only: bool
    course_weekday: str | None
    changes: tuple[str, ...] = ()


class CampusInputUnderstander:
    """保守纠错并把课程问题中的相对日期转换为确定星期。"""

    _course_patterns = (
        "什么课",
        "啥课",
        "课程",
        "课表",
        "上课",
        "在哪里上",
        "哪上",
        "教室",
        "几点上",
    )
    _non_course_subjects = (
        "校园卡",
        "饭卡",
        "图书馆",
        "借书",
        "还书",
        "成绩单",
        "奖学金",
        "实验室",
        "预约",
        "补办",
        "挂失",
        "退选",
        "流程",
        "材料",
        "规定",
        "政策",
    )
    _multi_intent_words = ("另外", "同时", "以及", "并且", "还想", "还要", "顺便")
    _relative_dates = (("后天", 2), ("明天", 1), ("今天", 0), ("今日", 0))

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(CHINA_TIMEZONE))

    def understand(self, query: str) -> InputUnderstanding:
        original = query.strip()
        normalized, typo_changes = self._correct_course_typos(original)
        course_intent = self._is_course_intent(normalized)
        changes = list(typo_changes)

        if course_intent:
            normalized, date_changes = self._resolve_relative_course_date(normalized)
            changes.extend(date_changes)

        weekday = self.extract_weekday(normalized) if course_intent else None
        course_only = course_intent and not any(
            word in normalized
            for word in (*self._non_course_subjects, *self._multi_intent_words)
        )
        return InputUnderstanding(
            original_query=original,
            normalized_query=normalized,
            course_intent=course_intent,
            course_only=course_only,
            course_weekday=weekday,
            changes=tuple(changes),
        )

    @staticmethod
    def _correct_course_typos(query: str) -> tuple[str, tuple[str, ...]]:
        corrected = query
        applied: list[str] = []
        rules = (
            (r"(?<=什么)[可客刻](?=$|[？?。！!，,])", "课", "什么可/客/刻→什么课"),
            (r"(?<=啥)[可客刻](?=$|[？?。！!，,])", "课", "啥可/客/刻→啥课"),
            (r"客表", "课表", "客表→课表"),
            (r"课[成呈城]", "课程", "课成/呈/城→课程"),
        )
        for pattern, replacement, label in rules:
            corrected, count = re.subn(pattern, replacement, corrected)
            if count:
                applied.append(label)
        return corrected, tuple(applied)

    def _resolve_relative_course_date(
        self,
        query: str,
    ) -> tuple[str, tuple[str, ...]]:
        matches = [(word, offset) for word, offset in self._relative_dates if word in query]
        if len(matches) != 1:
            return query, ()

        word, offset = matches[0]
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=CHINA_TIMEZONE)
        local_now = now.astimezone(CHINA_TIMEZONE)
        target = local_now.date() + timedelta(days=offset)
        weekday = WEEKDAYS[target.weekday()]
        normalized = query.replace(word, weekday)
        detail = f"{word}={target.isoformat()}（{weekday}）"
        return normalized, (detail,)

    @staticmethod
    def extract_weekday(text: str) -> str | None:
        lowered = text.lower()
        for alias in sorted(WEEKDAY_ALIASES, key=len, reverse=True):
            if alias in lowered:
                return WEEKDAY_ALIASES[alias]
        return None

    @classmethod
    def _is_course_intent(cls, query: str) -> bool:
        if any(pattern in query for pattern in cls._course_patterns):
            return True
        weekday = cls.extract_weekday(query)
        return weekday is not None and "课" in query and "退选" not in query
