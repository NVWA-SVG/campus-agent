from __future__ import annotations

import threading
import unittest
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from campus_agent.agent import build_default_agent
from campus_agent.domain import Message, Plan, ToolCall, ToolResult
from campus_agent.planner import RuleBasedPlanner
from campus_agent.tool_calling_planner import NativeToolCallingPlanner
from campus_agent.tooling import Tool


TOOL_DESCRIPTIONS = (
    {
        "name": "query_courses",
        "description": "查询课程",
        "parameters": {
            "type": "object",
            "properties": {
                "weekday": {"type": "string"},
                "keyword": {"type": "string"},
            },
            "required": ["weekday"],
        },
    },
)


class RecordingChatModel:
    """只实现规划器需要的 LangChain ChatModel 表面，且完全不访问网络。"""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        self._responses = iter(responses)
        self.bound_tools: list[dict[str, Any]] = []
        self.invocations: list[list[BaseMessage]] = []

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> "RecordingChatModel":
        self.bound_tools = list(tools)
        return self

    def invoke(
        self,
        input: Sequence[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.invocations.append(list(input))
        return next(self._responses)


class NativeToolCallingPlannerTests(unittest.TestCase):
    def test_plan_uses_bind_tools_and_standard_message_history(self) -> None:
        model = RecordingChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_courses",
                            "args": {"weekday": "周三", "keyword": ""},
                            "id": "provider-call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        planner = NativeToolCallingPlanner(model, model_name="fake")

        plan = planner.plan(
            "那周三呢？",
            (
                Message("user", "我想查询课程"),
                Message("assistant", "你想查询星期几？"),
            ),
            TOOL_DESCRIPTIONS,
        )

        self.assertEqual(plan.tool_calls[0].call_id, "provider-call-1")
        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertEqual(
            model.bound_tools[0]["function"]["name"],
            "query_courses",
        )
        self.assertIsInstance(model.invocations[0][-1], HumanMessage)
        self.assertEqual(model.invocations[0][-1].content, "那周三呢？")

    def test_agent_returns_tool_results_to_model_as_tool_messages(self) -> None:
        model = RecordingChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_courses",
                            "args": {"weekday": "周三", "keyword": "人工智能"},
                            "id": "provider-call-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="周三14:30的人工智能导论在教四-102上课。"
                ),
            ]
        )
        planner = NativeToolCallingPlanner(model, model_name="fake")

        response = build_default_agent(planner=planner).ask(
            "周三人工智能导论在哪里上？"
        )

        self.assertIn("教四-102", response.answer)
        compose_messages = model.invocations[1]
        tool_message = next(
            message for message in compose_messages if isinstance(message, ToolMessage)
        )
        self.assertEqual(tool_message.tool_call_id, "provider-call-2")
        self.assertIn("教四-102", tool_message.content)
        self.assertEqual(tool_message.status, "success")
        self.assertEqual(planner.metrics()["tool_protocol"], "native_tool_calling")
        self.assertEqual(planner.metrics()["api"]["calls"], 2)

    def test_unknown_model_tool_falls_back_to_rule_plan(self) -> None:
        model = RecordingChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delete_everything",
                            "args": {},
                            "id": "unsafe-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        planner = NativeToolCallingPlanner(model, fallback=RuleBasedPlanner())

        plan = planner.plan("周三有什么课？", (), TOOL_DESCRIPTIONS)

        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertEqual(planner.metrics()["fallback_count"], 1)

    def test_missing_api_key_never_invokes_network_model(self) -> None:
        planner = NativeToolCallingPlanner(
            None,
            model_name="deepseek-chat",
            api_key_configured=False,
        )

        plan = planner.plan("周三有什么课？", (), TOOL_DESCRIPTIONS)

        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        metrics = planner.metrics()
        self.assertFalse(metrics["api_key_configured"])
        self.assertEqual(metrics["api"]["calls"], 0)
        self.assertEqual(metrics["fallback_count"], 1)


class BarrierTool(Tool):
    name = "barrier_tool"
    description = "测试两个互不依赖的工具调用能否并发执行。"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def __init__(self) -> None:
        self._barrier = threading.Barrier(2)

    def run(self, **arguments: Any) -> str:
        self._barrier.wait(timeout=2)
        return str(arguments["value"])


class TwoToolPlanner:
    def plan(self, query, history, tool_descriptions):
        return Plan(
            summary="并行执行两个调用",
            tool_calls=(
                ToolCall("parallel-1", "barrier_tool", {"value": "first"}),
                ToolCall("parallel-2", "barrier_tool", {"value": "second"}),
            ),
        )

    def compose(self, query: str, results: Sequence[ToolResult]) -> str:
        return " ".join(result.output for result in results)


class ParallelToolExecutionTests(unittest.TestCase):
    def test_independent_tool_calls_execute_in_parallel_and_keep_order(self) -> None:
        agent = build_default_agent(
            planner=TwoToolPlanner(),
            extra_tools=(BarrierTool(),),
        )

        response = agent.ask("并行测试")

        self.assertEqual(response.answer, "first second")
        tool_events = [
            event for event in response.events if event.event_type == "tool_result"
        ]
        self.assertEqual(len(tool_events), 2)
        self.assertTrue(all("成功" in event.detail for event in tool_events))


if __name__ == "__main__":
    unittest.main()
