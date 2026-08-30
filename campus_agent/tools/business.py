"""Agent tool for the read-only campus business API."""

from __future__ import annotations

from typing import Any

from campus_agent.business_api import BusinessAPIError, CampusBusinessGateway
from campus_agent.domain import ToolOutput
from campus_agent.tooling import Tool, ToolExecutionError

_STATUS = {
    "open": "开放",
    "closed": "关闭",
    "busy": "繁忙",
    "maintenance": "维护中",
    "unknown": "未知",
}


class CampusServiceStatusTool(Tool):
    name = "query_campus_service_status"
    description = "查询校园服务当前开放状态、地点、时间和排队情况。只读。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "service_code": {
                "type": "string",
                "enum": ["campus_card", "registrar", "library", "student_affairs"],
            },
            "campus": {
                "type": "string",
                "enum": ["main", "east", "west"],
                "default": "main",
            },
        },
        "required": ["service_code"],
        "additionalProperties": False,
    }

    def __init__(self, gateway: CampusBusinessGateway) -> None:
        self._gateway = gateway

    def run(self, **arguments: Any) -> str:
        return self.invoke(**arguments).text

    def invoke(self, **arguments: Any) -> ToolOutput:
        try:
            result = self._gateway.query_service_status(
                arguments.get("service_code"), arguments.get("campus", "main")
            )
        except BusinessAPIError as exc:
            raise ToolExecutionError(
                exc.code, exc.public_message, retryable=exc.retryable
            ) from exc

        label = "[模拟数据]" if result.provider == "mock" else "[官方业务API]"
        if result.stale:
            label += " [缓存旧数据：上游暂不可用]"
        queue = (
            f"；排队 {result.queue_count} 人"
            if result.queue_count is not None
            else ""
        )
        if result.estimated_wait_minutes is not None:
            queue += f"，预计等待 {result.estimated_wait_minutes} 分钟"
        text = (
            f"{label} {result.service_name}当前状态：{_STATUS[result.status]}；"
            f"地点：{result.location}；今日时间：{result.today_hours}{queue}；"
            f"更新时间：{result.updated_at}。"
        )
        return ToolOutput(text=text, data=result.as_dict())

    def close(self) -> None:
        self._gateway.close()
