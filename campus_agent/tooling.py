"""工具协议、工具注册表以及统一的安全执行入口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from campus_agent.domain import ToolCall, ToolOutput, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Expected tool failure with details safe to expose to a user or model."""

    def __init__(self, code: str, public_message: str, *, retryable: bool = False) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def run(self, **arguments: Any) -> str:
        """执行工具并返回适合 Agent 使用的文本结果。"""

    def invoke(self, **arguments: Any) -> ToolOutput:
        """结构化执行入口；旧工具只实现 ``run`` 也可继续使用。"""

        output = self.run(**arguments)
        if not isinstance(output, str):
            raise TypeError("工具 run 必须返回字符串")
        return ToolOutput(text=output)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def descriptions(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        )

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error=f"未知工具：{call.name}",
            )

        try:
            output = tool.invoke(**call.arguments)
        except ToolExecutionError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error=f"{exc.code}：{exc.public_message}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error=f"参数或数据错误：{exc}",
            )
        except Exception as exc:  # Agent 边界必须阻止单个工具拖垮整个流程。
            # HTTP 客户端异常可能携带 URL、响应正文或请求头。这里只记录
            # 工具白名单名称和异常类型，既便于定位，也不会把凭据写入日志。
            logger.error(
                "Unexpected failure in tool %s (type=%s)",
                call.name,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error="工具执行失败，请稍后重试",
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            ok=True,
            output=output.text,
            citations=output.citations,
            data=dict(output.data),
        )
