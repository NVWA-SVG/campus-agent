from __future__ import annotations

import unittest

from campus_agent.agent import build_default_agent
from campus_agent.domain import Message, ToolCall
from campus_agent.memory import ConversationMemory
from campus_agent.tooling import ToolRegistry
from campus_agent.tools import CampusKnowledgeTool, CourseQueryTool


class CourseQueryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CourseQueryTool()

    def test_query_wednesday_courses(self) -> None:
        result = self.tool.run(weekday="星期三")
        self.assertIn("共找到2门课程", result)
        self.assertIn("机器学习基础", result)
        self.assertIn("人工智能导论", result)

    def test_query_with_keyword(self) -> None:
        result = self.tool.run(weekday="Wednesday", keyword="人工智能")
        self.assertIn("人工智能导论", result)
        self.assertNotIn("机器学习基础", result)

    def test_invalid_weekday(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的星期"):
            self.tool.run(weekday="星期八")


class ToolRegistryTests(unittest.TestCase):
    def test_unknown_tool_returns_failure_instead_of_crashing(self) -> None:
        registry = ToolRegistry()
        result = registry.execute(ToolCall("1", "missing_tool", {}))
        self.assertFalse(result.ok)
        self.assertIn("未知工具", result.error or "")

    def test_invalid_arguments_are_contained(self) -> None:
        registry = ToolRegistry()
        registry.register(CourseQueryTool())
        result = registry.execute(
            ToolCall("1", "query_courses", {"weekday": "星期八"})
        )
        self.assertFalse(result.ok)
        self.assertIn("参数或数据错误", result.error or "")


class ConversationMemoryTests(unittest.TestCase):
    def test_oldest_session_is_evicted_when_capacity_is_reached(self) -> None:
        memory = ConversationMemory(max_sessions=2)
        memory.add("oldest", Message(role="user", content="first"))
        memory.add("recent", Message(role="user", content="second"))
        memory.add("new", Message(role="user", content="third"))

        self.assertEqual(memory.get("oldest"), ())
        self.assertEqual(memory.get("recent")[0].content, "second")
        self.assertEqual(memory.get("new")[0].content, "third")


class CampusAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = build_default_agent()

    def test_agent_executes_course_tool(self) -> None:
        response = self.agent.ask("周三有什么课程？")
        self.assertIn("共找到2门课程", response.answer)
        self.assertTrue(any(event.event_type == "tool_result" for event in response.events))

    def test_agent_executes_knowledge_tool(self) -> None:
        response = self.agent.ask("校园卡丢了怎么补办？")
        self.assertIn("校园卡服务指南 - 挂失与补办", response.answer)
        self.assertIn("挂失", response.answer)

    def test_agent_can_call_multiple_tools(self) -> None:
        response = self.agent.ask("周一有什么课，校园卡丢了怎么补办？")
        self.assertIn("Python程序设计", response.answer)
        self.assertIn("校园卡服务指南 - 挂失与补办", response.answer)
        tool_events = [
            event for event in response.events if event.event_type == "tool_result"
        ]
        self.assertEqual(len(tool_events), 2)

    def test_follow_up_uses_conversation_memory(self) -> None:
        self.agent.ask("周三有什么课程？", session_id="student-1")
        response = self.agent.ask("在哪里上？", session_id="student-1")
        self.assertIn("教五-305", response.answer)
        self.assertIn("教四-102", response.answer)

    def test_unmatched_request_explains_supported_scope(self) -> None:
        response = self.agent.ask("帮我写一首诗")
        self.assertIn("目前可以查询", response.answer)

    def test_history_is_separated_by_session(self) -> None:
        self.agent.ask("周一有什么课？", session_id="a")
        self.assertEqual(len(self.agent.history("a")), 2)
        self.assertEqual(self.agent.history("b"), ())

    def test_failed_turn_does_not_leave_partial_history(self) -> None:
        class ExplodingPlanner:
            def plan(self, query, history, tool_descriptions):
                raise RuntimeError("planner failed")

            def compose(self, query, results):
                return "unused"

        agent = build_default_agent(planner=ExplodingPlanner())
        with self.assertRaisesRegex(RuntimeError, "planner failed"):
            agent.ask("周三有什么课？", session_id="failed-turn")
        self.assertEqual(agent.history("failed-turn"), ())


class KnowledgeToolTests(unittest.TestCase):
    def test_search_library_process(self) -> None:
        result = CampusKnowledgeTool().run(query="我应该怎么去图书馆借书？")
        self.assertIn("图书馆服务指南 - 借书与还书", result)
        self.assertIn("来源", result)


if __name__ == "__main__":
    unittest.main()
