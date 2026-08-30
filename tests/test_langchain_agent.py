from __future__ import annotations

import unittest

from campus_agent.langchain_agent import (
    LANGCHAIN_TOOLS,
    build_langchain_agent,
    final_text,
)
from campus_agent.langchain_demo import build_demo_model


class LangChainAgentTests(unittest.TestCase):
    def test_tools_have_structured_schemas(self) -> None:
        names = {tool.name for tool in LANGCHAIN_TOOLS}
        self.assertEqual(names, {"query_courses", "search_campus_knowledge"})

        course_tool = next(tool for tool in LANGCHAIN_TOOLS if tool.name == "query_courses")
        schema = course_tool.args_schema.model_json_schema()
        self.assertIn("weekday", schema["properties"])
        self.assertIn("keyword", schema["properties"])
        self.assertIn("weekday", schema["required"])

    def test_create_agent_executes_course_tool_without_api_key(self) -> None:
        agent = build_langchain_agent(build_demo_model())
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "周三人工智能课程在哪里上？"}
                ]
            }
        )

        tool_messages = [
            message for message in result["messages"] if message.type == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("人工智能导论", tool_messages[0].content)
        self.assertIn("教四-102", tool_messages[0].content)
        self.assertIn("教四-102", final_text(result))


if __name__ == "__main__":
    unittest.main()
