"""无需 API Key 的 LangChain 工具调用演示。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from campus_agent.langchain_agent import build_langchain_agent, final_text


class ToolCallingFakeChatModel(GenericFakeChatModel):
    """为离线演示补充工具绑定能力的确定性模型。"""

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolCallingFakeChatModel":
        return self


def build_demo_model() -> ToolCallingFakeChatModel:
    responses = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_courses",
                        "args": {"weekday": "周三", "keyword": "人工智能"},
                        "id": "demo-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="周三的人工智能导论在14:30上课，地点是教四-102。"),
        ]
    )
    return ToolCallingFakeChatModel(messages=responses)


def main() -> None:
    agent = build_langchain_agent(build_demo_model())
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "周三人工智能课程在哪里上？"}]}
    )

    print("LangChain Agent 执行轨迹：")
    for message in result["messages"]:
        print(f"- {message.type}: {message.content}")
    print(f"\n最终回答：{final_text(result)}")


if __name__ == "__main__":
    main()

