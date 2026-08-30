"""课程查询工具。"""

from __future__ import annotations

from typing import Any, TypedDict

from campus_agent.tooling import Tool


class Course(TypedDict):
    name: str
    weekday: str
    start_time: str
    location: str


DEFAULT_COURSES: tuple[Course, ...] = (
    {
        "name": "Python程序设计",
        "weekday": "周一",
        "start_time": "08:00",
        "location": "教四-201",
    },
    {
        "name": "机器学习基础",
        "weekday": "周三",
        "start_time": "10:10",
        "location": "教五-305",
    },
    {
        "name": "人工智能导论",
        "weekday": "周三",
        "start_time": "14:30",
        "location": "教四-102",
    },
    {
        "name": "大学英语",
        "weekday": "周五",
        "start_time": "08:00",
        "location": "教三-408",
    },
)


WEEKDAY_ALIASES = {
    "周一": "周一",
    "星期一": "周一",
    "monday": "周一",
    "周二": "周二",
    "星期二": "周二",
    "tuesday": "周二",
    "周三": "周三",
    "星期三": "周三",
    "wednesday": "周三",
    "周四": "周四",
    "星期四": "周四",
    "thursday": "周四",
    "周五": "周五",
    "星期五": "周五",
    "friday": "周五",
    "周六": "周六",
    "星期六": "周六",
    "saturday": "周六",
    "周日": "周日",
    "星期日": "周日",
    "星期天": "周日",
    "sunday": "周日",
}


def normalize_weekday(raw_weekday: str) -> str:
    cleaned = raw_weekday.strip().lower()
    try:
        return WEEKDAY_ALIASES[cleaned]
    except KeyError as exc:
        raise ValueError(f"不支持的星期：{raw_weekday}") from exc


class CourseQueryTool(Tool):
    name = "query_courses"
    description = "按星期查询课程，也可以使用课程名关键字进一步筛选。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "weekday": {"type": "string", "description": "要查询的星期"},
            "keyword": {"type": "string", "description": "可选的课程名关键字"},
        },
        "required": ["weekday"],
    }

    def __init__(self, courses: tuple[Course, ...] = DEFAULT_COURSES) -> None:
        self._courses = courses

    def run(self, **arguments: Any) -> str:
        weekday_value = arguments.get("weekday")
        if not isinstance(weekday_value, str):
            raise ValueError("weekday 必须是字符串")

        keyword_value = arguments.get("keyword", "")
        if not isinstance(keyword_value, str):
            raise ValueError("keyword 必须是字符串")

        weekday = normalize_weekday(weekday_value)
        keyword = keyword_value.strip().lower()
        matched = [
            course
            for course in self._courses
            if course["weekday"] == weekday
            and (not keyword or keyword in course["name"].lower())
        ]

        if not matched:
            suffix = f"，关键字“{keyword_value.strip()}”" if keyword else ""
            return f"{weekday}{suffix}没有找到课程。"

        lines = [f"{weekday}共找到{len(matched)}门课程："]
        lines.extend(
            f"- {course['start_time']} {course['name']}，地点：{course['location']}"
            for course in matched
        )
        return "\n".join(lines)

