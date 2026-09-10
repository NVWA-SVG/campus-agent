from __future__ import annotations

from datetime import datetime

from campus_agent.agent import build_default_agent
from campus_agent.domain import Plan, ToolCall, ToolResult
from campus_agent.input_understanding import CHINA_TIMEZONE, CampusInputUnderstander


def fixed_monday() -> datetime:
    return datetime(2026, 9, 7, 9, 0, tzinfo=CHINA_TIMEZONE)


def test_today_course_is_resolved_without_clarification() -> None:
    agent = build_default_agent(
        input_understander=CampusInputUnderstander(clock=fixed_monday)
    )

    response = agent.ask("今天有什么课？")

    assert "Python程序设计" in response.answer
    assert response.events[0].event_type == "understand"
    assert "2026-09-07（周一）" in response.events[0].detail
    assert agent.history()[0].content == "今天有什么课？"


def test_high_confidence_course_typo_is_corrected() -> None:
    agent = build_default_agent(
        input_understander=CampusInputUnderstander(clock=fixed_monday)
    )

    response = agent.ask("周三有什么可？")

    assert "机器学习基础" in response.answer
    assert "人工智能导论" in response.answer
    assert "什么可/客/刻→什么课" in response.events[0].detail


class WrongCourseRoutePlanner:
    def plan(self, query, history, tool_descriptions):
        return Plan(
            summary="错误地选择了知识库",
            tool_calls=(
                ToolCall("wrong-route", "search_campus_knowledge", {"query": query}),
            ),
        )

    def compose(self, query: str, results: tuple[ToolResult, ...]) -> str:
        return "\n".join(result.output for result in results)


def test_clear_course_intent_overrides_wrong_model_tool_route() -> None:
    agent = build_default_agent(
        planner=WrongCourseRoutePlanner(),
        input_understander=CampusInputUnderstander(clock=fixed_monday),
    )

    response = agent.ask("周三有什么可？")

    assert "机器学习基础" in response.answer
    assert any(
        event.event_type == "plan" and "课程路由校正" in event.detail
        for event in response.events
    )
    assert all("search_campus_knowledge" not in event.detail for event in response.events)
