from __future__ import annotations

import unittest

from campus_agent.agent import CampusAgent
from campus_agent.domain import Plan, ToolCall
from campus_agent.graph.workflow import CampusGraphAgent
from campus_agent.memory import ConversationMemory
from campus_agent.tooling import Tool, ToolRegistry


class RecordingTool(Tool):
    name = "record"
    description = "记录调用顺序"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, **arguments):
        value = str(arguments["value"])
        self.calls.append(value)
        return value


class StaticPlanner:
    def __init__(self, plan: Plan, *, explode_compose: bool = False) -> None:
        self._plan = plan
        self.explode_compose = explode_compose
        self.compose_calls = 0

    def plan(self, query, history, tool_descriptions):
        return self._plan

    def compose(self, query, results):
        self.compose_calls += 1
        if self.explode_compose:
            raise RuntimeError("compose failed")
        return ",".join(result.output for result in results)


class LangGraphWorkflowTests(unittest.TestCase):
    def _agent(self, planner, *, max_tool_calls: int = 4):
        calls: list[str] = []
        registry = ToolRegistry()
        registry.register(RecordingTool(calls))
        memory = ConversationMemory()
        return (
            CampusAgent(
                planner=planner,
                tools=registry,
                memory=memory,
                max_tool_calls=max_tool_calls,
            ),
            calls,
        )

    def test_public_agent_is_backed_by_langgraph(self) -> None:
        planner = StaticPlanner(Plan(summary="直接回答", direct_answer="ok"))
        agent, _ = self._agent(planner)

        self.assertIsInstance(agent, CampusGraphAgent)
        self.assertEqual(
            set(agent._graph.get_graph().nodes),
            {
                "__start__",
                "plan",
                "direct_answer",
                "execute_tools",
                "grade_documents",
                "rewrite_query",
                "retry_knowledge",
                "compose",
                "verify_answer",
                "commit_turn",
                "__end__",
            },
        )

    def test_direct_answer_skips_compose(self) -> None:
        planner = StaticPlanner(Plan(summary="直接回答", direct_answer=""))
        agent, calls = self._agent(planner)

        response = agent.ask("question")

        self.assertEqual(response.answer, "")
        self.assertEqual(planner.compose_calls, 0)
        self.assertEqual(calls, [])
        self.assertEqual([event.event_type for event in response.events], ["plan", "final"])

    def test_empty_tool_plan_still_calls_compose(self) -> None:
        planner = StaticPlanner(Plan(summary="无工具", tool_calls=()))
        agent, _ = self._agent(planner)

        agent.ask("question")

        self.assertEqual(planner.compose_calls, 1)

    def test_max_tool_calls_and_order_are_preserved(self) -> None:
        planner = StaticPlanner(
            Plan(
                summary="多个工具",
                tool_calls=tuple(
                    ToolCall(str(index), "record", {"value": str(index)})
                    for index in range(6)
                ),
            )
        )
        agent, calls = self._agent(planner, max_tool_calls=4)

        response = agent.ask("question")

        self.assertEqual(calls, ["0", "1", "2", "3"])
        self.assertEqual(response.answer, "0,1,2,3")

    def test_compose_failure_does_not_commit_a_partial_turn(self) -> None:
        planner = StaticPlanner(
            Plan(
                summary="调用工具",
                tool_calls=(ToolCall("1", "record", {"value": "x"}),),
            ),
            explode_compose=True,
        )
        agent, _ = self._agent(planner)

        with self.assertRaisesRegex(RuntimeError, "compose failed"):
            agent.ask("question", session_id="failed-compose")

        self.assertEqual(agent.history("failed-compose"), ())


if __name__ == "__main__":
    unittest.main()
