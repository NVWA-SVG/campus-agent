"""基于 LangChain 标准消息与原生工具调用的模型规划器。

LangGraph 仍负责 Campus 的业务控制流；本模块只负责两个明确的 LLM 环节：

1. 把会话消息和工具 JSON Schema 交给模型，读取 ``AIMessage.tool_calls``；
2. 把工具执行结果作为 ``ToolMessage`` 交回模型，生成最终回答。

这样既使用供应商无关的 LangChain 消息协议，又保留项目自己的 ToolRegistry、
Citation、Grade/Rewrite/Verify 和规则回退边界。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from campus_agent.deepseek import DeepSeekConfig
from campus_agent.domain import Message, Plan, ToolCall, ToolResult
from campus_agent.planner import Planner, RuleBasedPlanner


TOOL_SELECTION_PROMPT = """你是校园事务 Agent 的工具决策节点。
课程、地点、时间、校园办事规则和实时业务状态等事实必须通过工具获取，不能猜测。
根据当前问题和对话上下文选择零到四个工具；可以在一次响应中并列请求多个工具。
每轮最多调用一次 search_campus_knowledge，多个知识问题要合并成一个完整 query。
缺少工具必需参数时直接提出一个简短的澄清问题。
简单问候、能力说明或明显超出能力范围的问题可以直接回答，不要调用无关工具。
不要声称工具已经执行；工具执行由 LangGraph 在下一节点完成。
"""


ANSWER_COMPOSER_PROMPT = """你是校园事务 Agent 的答案生成节点。
只使用紧随其后的工具结果回答当前问题，不得补充工具结果中没有的事实。
工具失败或证据不足时要明确说明，不得猜测。
知识检索来源和业务来源标签必须保留。回答使用简洁中文。
"""


class ToolCallingChatModel(Protocol):
    """规划器所需的最小 LangChain ChatModel 接口，便于离线测试替换。"""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> Any: ...

    def invoke(
        self,
        input: Sequence[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage: ...


class NativeToolCallingPlanner:
    """通过 ``bind_tools`` 产生标准工具调用，同时实现现有 Planner 协议。"""

    def __init__(
        self,
        model: ToolCallingChatModel | None,
        *,
        model_name: str | None = None,
        fallback: Planner | None = None,
        api_key_configured: bool | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._fallback = fallback or RuleBasedPlanner()
        self._api_key_configured = (
            model is not None
            if api_key_configured is None
            else api_key_configured
        )
        self._lock = threading.Lock()
        self._calls = 0
        self._successes = 0
        self._failures = 0
        self._fallback_count = 0
        self._last_fallback_reason: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._total_latency_ms = 0

    @classmethod
    def from_deepseek_environment(cls) -> "NativeToolCallingPlanner":
        """用 ``init_chat_model`` 创建官方 DeepSeek 集成；无 Key 时离线回退。"""

        config = DeepSeekConfig.from_environment()
        if not config.api_key:
            return cls(
                None,
                model_name=config.model,
                api_key_configured=False,
            )

        # 延迟导入让完全离线的 rule 模式不需要初始化任何供应商客户端。
        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            config.model,
            model_provider="deepseek",
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0,
            max_tokens=1200,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            streaming=False,
        )
        return cls(
            model,
            model_name=config.model,
            api_key_configured=True,
        )

    def plan(
        self,
        query: str,
        history: Sequence[Message],
        tool_descriptions: Sequence[dict[str, object]],
    ) -> Plan:
        """调用绑定工具后的模型，并把标准 ToolCall 转为稳定领域对象。"""

        if not query.strip():
            return self._fallback.plan(query, history, tool_descriptions)

        try:
            schemas = self._tool_schemas(tool_descriptions)
            model = self._require_model()
            model_with_tools = model.bind_tools(
                schemas,
                tool_choice="auto",
                parallel_tool_calls=True,
            )
            response = self._invoke(
                model_with_tools,
                [
                    SystemMessage(content=TOOL_SELECTION_PROMPT),
                    *self._history_messages(history),
                    HumanMessage(content=query),
                ],
                component="plan",
            )
            calls = self._parse_tool_calls(response, tool_descriptions)
            if calls:
                names = "、".join(call.name for call in calls)
                return Plan(
                    summary=f"模型选择工具：{names}",
                    tool_calls=tuple(calls),
                )

            answer = self._message_text(response)
            if not answer:
                raise ValueError("模型既没有返回文本，也没有返回工具调用")

            # 如果规则安全网判断这个问题明显需要事实工具，不能接受模型绕过工具
            # 直接作答；改走确定性的规则 Plan，避免无依据事实进入 direct 分支。
            safety_plan = self._fallback.plan(query, history, tool_descriptions)
            if safety_plan.tool_calls:
                self._record_fallback("模型未按要求调用事实工具")
                return safety_plan
            return Plan(summary="模型直接回答", direct_answer=answer)
        except Exception as error:
            return self._fallback_plan(query, history, tool_descriptions, error)

    def compose(self, query: str, results: Sequence[ToolResult]) -> str:
        """兼容旧 Planner 协议；新图节点会优先调用 ``compose_with_context``。"""

        return self.compose_with_context(query, (), (), results, ())

    def compose_with_context(
        self,
        query: str,
        history: Sequence[Message],
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
        tool_descriptions: Sequence[dict[str, object]],
    ) -> str:
        """使用 AIMessage → ToolMessage → AIMessage 的标准工具消息序列。"""

        if not results:
            return self._fallback.compose(query, results)

        try:
            model = self._require_model()
            calls_by_id = {call.call_id: call for call in calls}
            # 正常路径中 calls 与 results 一一对应。兼容测试/旧调用方时，用结果
            # 补出最小 ToolCall，确保每个 ToolMessage 都有对应的 call id。
            normalized_calls = [
                calls_by_id.get(
                    result.call_id,
                    ToolCall(
                        call_id=result.call_id,
                        name=result.tool_name,
                        arguments={},
                    ),
                )
                for result in results
            ]
            assistant_tool_request = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": call.name,
                        "args": dict(call.arguments),
                        "id": call.call_id,
                        "type": "tool_call",
                    }
                    for call in normalized_calls
                ],
            )
            tool_messages = [
                ToolMessage(
                    content=result.output if result.ok else (result.error or "工具执行失败"),
                    tool_call_id=result.call_id,
                    name=result.tool_name,
                    status="success" if result.ok else "error",
                    artifact={
                        "citations": [
                            citation.as_dict() for citation in result.citations
                        ],
                        "data": dict(result.data),
                    },
                )
                for result in results
            ]
            # 后续调用继续绑定同一份工具 Schema，使发送给供应商的消息保持完整
            # 的原生 tool-calling 会话；tool_choice=none 明确结束本轮工具决策。
            response = self._invoke(
                (
                    model.bind_tools(
                        self._tool_schemas(tool_descriptions),
                        tool_choice="none",
                    )
                    if tool_descriptions
                    else model
                ),
                [
                    SystemMessage(content=ANSWER_COMPOSER_PROMPT),
                    *self._history_messages(history),
                    HumanMessage(content=query),
                    assistant_tool_request,
                    *tool_messages,
                ],
                component="compose",
            )
            answer = self._message_text(response)
            if not answer:
                raise ValueError("模型回答为空")
            return self._preserve_required_labels(answer, results)
        except Exception as error:
            self._record_fallback(self._safe_reason(error))
            return self._fallback.compose(query, results)

    def metrics(self) -> dict[str, object]:
        with self._lock:
            return {
                "planner": "deepseek",
                "model": self._model_name,
                "integration": "langchain-deepseek",
                "tool_protocol": "native_tool_calling",
                "api_key_configured": self._api_key_configured,
                "fallback_count": self._fallback_count,
                "last_fallback_reason": self._last_fallback_reason,
                "api": {
                    "calls": self._calls,
                    "successes": self._successes,
                    "failures": self._failures,
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": self._total_tokens,
                    "total_latency_ms": self._total_latency_ms,
                },
            }

    def _invoke(
        self,
        runnable: Any,
        messages: Sequence[BaseMessage],
        *,
        component: str,
    ) -> AIMessage:
        started = time.perf_counter()
        with self._lock:
            self._calls += 1
        try:
            response = runnable.invoke(
                list(messages),
                config={
                    "tags": ["campus-agent", "llm"],
                    "metadata": {"component": component},
                },
            )
            if not isinstance(response, AIMessage):
                raise TypeError("ChatModel 必须返回 AIMessage")
            with self._lock:
                self._successes += 1
                self._record_usage(response)
            return response
        except Exception:
            with self._lock:
                self._failures += 1
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                self._total_latency_ms += round(elapsed * 1000)

    def _require_model(self) -> ToolCallingChatModel:
        if self._model is None:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY")
        return self._model

    def _fallback_plan(
        self,
        query: str,
        history: Sequence[Message],
        tool_descriptions: Sequence[dict[str, object]],
        error: Exception,
    ) -> Plan:
        self._record_fallback(self._safe_reason(error))
        return self._fallback.plan(query, history, tool_descriptions)

    def _record_fallback(self, reason: str) -> None:
        with self._lock:
            self._fallback_count += 1
            self._last_fallback_reason = reason

    def _record_usage(self, response: AIMessage) -> None:
        usage = response.usage_metadata or {}
        self._prompt_tokens += self._non_negative_int(usage.get("input_tokens"))
        self._completion_tokens += self._non_negative_int(usage.get("output_tokens"))
        self._total_tokens += self._non_negative_int(usage.get("total_tokens"))

    @staticmethod
    def _non_negative_int(value: object) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _safe_reason(error: Exception) -> str:
        if str(error) == "未设置 DEEPSEEK_API_KEY":
            return str(error)
        if isinstance(error, ValueError):
            return "模型工具调用未通过本地校验"
        return f"模型调用失败：{type(error).__name__}"

    @staticmethod
    def _history_messages(history: Sequence[Message]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in history[-8:]:
            if message.role == "user":
                converted.append(HumanMessage(content=message.content[:500]))
            else:
                converted.append(AIMessage(content=message.content[:500]))
        return converted

    @staticmethod
    def _tool_schemas(
        descriptions: Sequence[dict[str, object]],
    ) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for description in descriptions:
            name = description.get("name")
            text = description.get("description")
            parameters = description.get("parameters")
            if not isinstance(name, str) or not name:
                raise ValueError("工具描述缺少名称")
            if not isinstance(text, str) or not isinstance(parameters, dict):
                raise ValueError(f"工具 {name} 的描述或参数 Schema 无效")
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": text,
                        "parameters": parameters,
                    },
                }
            )
        return schemas

    @classmethod
    def _parse_tool_calls(
        cls,
        response: AIMessage,
        descriptions: Sequence[dict[str, object]],
    ) -> list[ToolCall]:
        if len(response.tool_calls) > 4:
            raise ValueError("单轮最多允许四个工具调用")
        available = {
            str(description.get("name")): description
            for description in descriptions
        }
        calls: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        seen_ids: set[str] = set()
        for index, raw_call in enumerate(response.tool_calls, start=1):
            name = raw_call.get("name")
            arguments = raw_call.get("args")
            if not isinstance(name, str) or name not in available:
                raise ValueError(f"模型请求了未知工具：{name}")
            if not isinstance(arguments, dict):
                raise ValueError(f"工具 {name} 的参数必须是对象")
            cls._validate_arguments(name, arguments, available[name])
            signature = (
                name,
                json.dumps(arguments, ensure_ascii=False, sort_keys=True),
            )
            if signature in seen:
                raise ValueError(f"模型生成了重复工具调用：{name}")
            seen.add(signature)
            raw_call_id = raw_call.get("id")
            call_id = (
                raw_call_id
                if isinstance(raw_call_id, str) and raw_call_id
                else f"model-call-{index}"
            )
            if call_id in seen_ids:
                raise ValueError(f"模型生成了重复 call id：{call_id}")
            seen_ids.add(call_id)
            calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=dict(arguments),
                )
            )
        if sum(call.name == "search_campus_knowledge" for call in calls) > 1:
            raise ValueError("单轮最多允许一次知识检索")
        return calls

    @staticmethod
    def _validate_arguments(
        tool_name: str,
        arguments: dict[str, Any],
        description: dict[str, object],
    ) -> None:
        schema = description.get("parameters")
        if not isinstance(schema, dict):
            raise ValueError(f"工具 {tool_name} 的参数 Schema 无效")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError(f"工具 {tool_name} 的参数 Schema 无效")
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ValueError(f"工具 {tool_name} 包含未知参数：{sorted(unknown)}")
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"工具 {tool_name} 缺少参数：{missing}")

        python_types: dict[object, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for name, value in arguments.items():
            property_schema = properties.get(name)
            if not isinstance(property_schema, dict):
                continue
            expected = property_schema.get("type")
            expected_type = python_types.get(expected)
            if expected_type is not None and not isinstance(value, expected_type):
                raise ValueError(f"工具 {tool_name} 参数 {name} 应为 {expected}")

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        return message.text.strip()

    @staticmethod
    def _preserve_required_labels(
        answer: str,
        results: Sequence[ToolResult],
    ) -> str:
        business_outputs = [
            result.output
            for result in results
            if result.ok and result.tool_name == "query_campus_service_status"
        ]
        required = [
            label
            for label in ("[模拟数据]", "[官方业务API]", "缓存旧数据")
            if any(label in output for output in business_outputs)
            and label not in answer
        ]
        return f"{' '.join(required)} {answer}" if required else answer
