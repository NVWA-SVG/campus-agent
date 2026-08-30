from __future__ import annotations

import json
import unittest
import urllib.error
from typing import Any

from campus_agent.agent import build_default_agent
from campus_agent.deepseek import DeepSeekChatClient, DeepSeekConfig
from campus_agent.deepseek_planner import DeepSeekPlanner
from campus_agent.domain import ToolResult


class FakeJsonClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.call_count = 0

    def complete_json(self, messages):
        self.call_count += 1
        return next(self.responses)

    def metrics(self) -> dict[str, int]:
        return {"calls": self.call_count}


class FakeHttpResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


TOOLS = (
    {
        "name": "query_courses",
        "description": "按星期查询课程",
        "parameters": {
            "type": "object",
            "properties": {
                "weekday": {"type": "string"},
                "keyword": {"type": "string"},
            },
            "required": ["weekday"],
        },
    },
    {
        "name": "search_campus_knowledge",
        "description": "查询校园办事知识",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
)


class DeepSeekPlannerTests(unittest.TestCase):
    def test_config_rejects_non_official_or_insecure_base_url(self) -> None:
        for base_url in (
            "http://api.deepseek.com",
            "https://example.com",
            "https://api.deepseek.com.evil.example",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "官方HTTPS地址"):
                    DeepSeekConfig(api_key="test-key", base_url=base_url)

    def test_model_plan_is_converted_to_tool_calls(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "查询周三课程",
                    "clarification_question": None,
                    "tool_calls": [
                        {
                            "name": "query_courses",
                            "arguments": {"weekday": "周三", "keyword": ""},
                        }
                    ],
                }
            ]
        )
        plan = DeepSeekPlanner(client).plan("周三有什么课？", (), TOOLS)
        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertEqual(plan.tool_calls[0].arguments["weekday"], "周三")

    def test_model_can_plan_multiple_tools(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "查询课程和校园卡",
                    "clarification_question": None,
                    "tool_calls": [
                        {
                            "name": "query_courses",
                            "arguments": {"weekday": "周一", "keyword": ""},
                        },
                        {
                            "name": "search_campus_knowledge",
                            "arguments": {"query": "校园卡丢了怎么补办"},
                        },
                    ],
                }
            ]
        )
        plan = DeepSeekPlanner(client).plan("周一有什么课，校园卡怎么办？", (), TOOLS)
        self.assertEqual(len(plan.tool_calls), 2)

    def test_multiple_knowledge_calls_fall_back_to_one_safe_query(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "错误地拆成两次知识检索",
                    "clarification_question": None,
                    "tool_calls": [
                        {
                            "name": "search_campus_knowledge",
                            "arguments": {"query": "校园卡补办"},
                        },
                        {
                            "name": "search_campus_knowledge",
                            "arguments": {"query": "图书馆借书"},
                        },
                    ],
                }
            ]
        )
        planner = DeepSeekPlanner(client)

        plan = planner.plan("校园卡怎么补办？", (), TOOLS)

        self.assertLessEqual(
            sum(call.name == "search_campus_knowledge" for call in plan.tool_calls),
            1,
        )
        self.assertEqual(planner.metrics()["fallback_count"], 1)

    def test_unknown_model_tool_falls_back_to_rule_planner(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "错误工具",
                    "clarification_question": None,
                    "tool_calls": [{"name": "browse_web", "arguments": {}}],
                }
            ]
        )
        planner = DeepSeekPlanner(client)
        plan = planner.plan("周三有什么课？", (), TOOLS)
        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertIn("回退规则规划", plan.summary)
        self.assertEqual(planner.metrics()["fallback_count"], 1)

    def test_missing_api_key_uses_offline_fallback_without_network(self) -> None:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("没有API Key时不应访问网络")

        client = DeepSeekChatClient(
            DeepSeekConfig(api_key=None),
            opener=fail_if_called,
        )
        plan = DeepSeekPlanner(client).plan("周三有什么课？", (), TOOLS)
        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertIn("DeepSeek不可用", plan.summary)

    def test_composer_uses_model_answer(self) -> None:
        client = FakeJsonClient([{"answer": "周三有两门课程。"}])
        answer = DeepSeekPlanner(client).compose(
            "周三有什么课？",
            [ToolResult("1", "query_courses", True, "原始课程结果")],
        )
        self.assertEqual(answer, "周三有两门课程。")

    def test_invalid_composer_output_returns_original_tool_result(self) -> None:
        client = FakeJsonClient([{"wrong_key": "invalid"}])
        answer = DeepSeekPlanner(client).compose(
            "周三有什么课？",
            [ToolResult("1", "query_courses", True, "原始课程结果")],
        )
        self.assertEqual(answer, "原始课程结果")

    def test_http_client_calls_only_configured_deepseek_endpoint(self) -> None:
        captured: dict[str, Any] = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeHttpResponse(
                {
                    "choices": [
                        {"message": {"content": '{"summary":"ok"}'}}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            )

        client = DeepSeekChatClient(
            DeepSeekConfig(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                timeout_seconds=9,
            ),
            opener=opener,
        )
        result = client.complete_json([{"role": "user", "content": "test"}])
        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(
            captured["url"], "https://api.deepseek.com/chat/completions"
        )
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(
            captured["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(
            captured["payload"]["thinking"], {"type": "disabled"}
        )
        self.assertNotIn("api_key", captured["payload"])
        self.assertEqual(client.metrics()["total_tokens"], 12)

    def test_http_client_retries_transient_network_error_once(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def opener(request, timeout):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise urllib.error.URLError("temporary")
            return FakeHttpResponse(
                {
                    "choices": [
                        {"message": {"content": '{"answer":"ok"}'}}
                    ]
                }
            )

        client = DeepSeekChatClient(
            DeepSeekConfig(api_key="test-key", max_retries=1),
            opener=opener,
            sleeper=sleeps.append,
        )
        self.assertEqual(
            client.complete_json([{"role": "user", "content": "test"}]),
            {"answer": "ok"},
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(client.metrics()["retries"], 1)
        self.assertEqual(client.metrics()["network_attempts"], 2)
        self.assertEqual(len(sleeps), 1)

    def test_full_agent_loop_can_use_deepseek_plan_and_answer(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "查询周三课程",
                    "clarification_question": None,
                    "tool_calls": [
                        {
                            "name": "query_courses",
                            "arguments": {"weekday": "周三", "keyword": ""},
                        }
                    ],
                },
                {"answer": "周三有机器学习基础和人工智能导论。"},
            ]
        )
        agent = build_default_agent(DeepSeekPlanner(client))
        response = agent.ask("周三有什么课？")
        self.assertIn("人工智能导论", response.answer)
        self.assertEqual(client.call_count, 2)

    def test_clarification_is_allowed_but_not_factual_direct_answer(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "clarify",
                    "summary": "课程查询缺少星期",
                    "clarification_question": "你想查询星期几的课程？",
                    "tool_calls": [],
                }
            ]
        )
        plan = DeepSeekPlanner(client).plan("有什么课？", (), TOOLS)
        self.assertEqual(plan.direct_answer, "你想查询星期几的课程？")
        self.assertEqual(plan.tool_calls, ())

    def test_unsupported_answer_is_fixed_by_local_code(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "unsupported",
                    "summary": "超出能力边界",
                    "clarification_question": None,
                    "tool_calls": [],
                }
            ]
        )
        plan = DeepSeekPlanner(client).plan("帮我写诗", (), TOOLS)
        self.assertIn("目前可以查询", plan.direct_answer or "")

    def test_missing_required_argument_falls_back(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "decision": "tools",
                    "summary": "参数不完整",
                    "clarification_question": None,
                    "tool_calls": [
                        {"name": "query_courses", "arguments": {}}
                    ],
                }
            ]
        )
        planner = DeepSeekPlanner(client)
        plan = planner.plan("周三有什么课？", (), TOOLS)
        self.assertEqual(plan.tool_calls[0].name, "query_courses")
        self.assertEqual(planner.metrics()["fallback_count"], 1)

    def test_empty_input_does_not_call_api(self) -> None:
        client = FakeJsonClient([])
        plan = DeepSeekPlanner(client).plan("  ", (), TOOLS)
        self.assertEqual(client.call_count, 0)
        self.assertIn("请输入", plan.direct_answer or "")


if __name__ == "__main__":
    unittest.main()
