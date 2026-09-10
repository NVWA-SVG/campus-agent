"""Native desktop process boundary for Campus Agent.

The Rust application communicates with this module over newline-delimited JSON
(NDJSON).  It deliberately reuses ``WebAgentService`` without starting FastAPI,
Uvicorn, a browser, or a listening socket, so every existing LangGraph, RAG,
checkpoint, business-tool, and LangSmith behavior stays identical to the web UI.
"""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, TextIO

from fastapi import HTTPException

from campus_agent.rag.knowledge_base import KnowledgeBaseError
from campus_agent.web import ChatRequest, KnowledgeFilterRequest, WebAgentService


PROTOCOL_VERSION = "campus-agent-desktop/v1"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SUPPORTED_COMMANDS = frozenset(
    {
        "status",
        "chat",
        "history",
        "clear_history",
        "list_knowledge",
        "upload_knowledge",
        "delete_knowledge",
        "rebuild_knowledge",
        "shutdown",
    }
)


class DesktopService(Protocol):
    def status(self) -> dict[str, object]: ...

    def stream_chat(self, request: ChatRequest) -> Iterable[str]: ...

    def history(self, session_id: str) -> list[dict[str, str]]: ...

    def clear(self, session_id: str) -> None: ...

    def knowledge_documents(self) -> dict[str, object]: ...

    def upload_knowledge(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str,
        domain: str = "uploaded",
        category: str = "general",
        version: str = "1",
    ) -> dict[str, object]: ...

    def delete_knowledge(self, document_id: str) -> dict[str, object]: ...

    def rebuild_knowledge(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class DesktopProtocolError(ValueError):
    """A safe validation error that may be returned to the local UI."""


class DesktopProtocolServer:
    """Serve one trusted local desktop client over stdin/stdout."""

    def __init__(
        self,
        service: DesktopService,
        *,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
    ) -> None:
        self._service = service
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._running = True

    def serve(self) -> None:
        self.emit("desktop", "ready", self._safe_status())
        try:
            for raw_line in self._stdin:
                if not self._running:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                self._handle_line(line)
                if not self._running:
                    break
        finally:
            self._service.close()

    def _handle_line(self, line: str) -> None:
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise DesktopProtocolError("请求必须是 JSON 对象")
            request_id = self._request_id(request)
            command = request.get("command")
            if not isinstance(command, str) or command not in SUPPORTED_COMMANDS:
                raise DesktopProtocolError("不支持的桌面命令")
            payload = request.get("payload", {})
            if not isinstance(payload, dict):
                raise DesktopProtocolError("payload 必须是 JSON 对象")
            self._dispatch(request_id, command, payload)
        except (DesktopProtocolError, KnowledgeBaseError, HTTPException) as error:
            self.emit(request_id, "error", {"message": self._safe_error(error)})
        except (OSError, UnicodeError) as error:
            self.emit(
                request_id,
                "error",
                {"message": f"本地文件操作失败：{type(error).__name__}"},
            )
        except Exception:
            traceback.print_exc(file=self._stderr)
            self.emit(request_id, "error", {"message": "Agent 执行失败，请查看本地日志"})

    def _dispatch(
        self,
        request_id: str,
        command: str,
        payload: dict[str, Any],
    ) -> None:
        if command == "status":
            self.emit(request_id, "status", self._safe_status())
            return
        if command == "chat":
            self._chat(request_id, payload)
            return
        if command == "history":
            session_id = self._session_id(payload)
            self.emit(
                request_id,
                "history",
                {"messages": self._service.history(session_id)},
            )
            return
        if command == "clear_history":
            session_id = self._session_id(payload)
            self._service.clear(session_id)
            self.emit(request_id, "history_cleared", {"session_id": session_id})
            return
        if command == "list_knowledge":
            self.emit(request_id, "knowledge", self._service.knowledge_documents())
            return
        if command == "upload_knowledge":
            self._upload(request_id, payload)
            return
        if command == "delete_knowledge":
            document_id = self._required_string(payload, "document_id", max_length=64)
            result = self._service.delete_knowledge(document_id)
            self.emit(request_id, "knowledge_changed", result)
            return
        if command == "rebuild_knowledge":
            self.emit(
                request_id,
                "knowledge_changed",
                {"stats": self._service.rebuild_knowledge()},
            )
            return
        if command == "shutdown":
            self.emit(request_id, "shutdown", {"ok": True})
            self._running = False
            return
        raise DesktopProtocolError("不支持的桌面命令")

    def _chat(self, request_id: str, payload: dict[str, Any]) -> None:
        query = self._required_string(payload, "query", max_length=2000)
        session_id = self._session_id(payload)
        planner = payload.get("planner", "rule")
        if planner not in {"rule", "deepseek"}:
            raise DesktopProtocolError("planner 只能是 rule 或 deepseek")
        raw_filters = payload.get("filters")
        if raw_filters is not None and not isinstance(raw_filters, dict):
            raise DesktopProtocolError("filters 必须是 JSON 对象")
        request = ChatRequest(
            query=query,
            session_id=session_id,
            planner=planner,
            filters=(
                KnowledgeFilterRequest.model_validate(raw_filters)
                if raw_filters
                else None
            ),
        )
        for sse_message in self._service.stream_chat(request):
            event_type, event_payload = parse_sse_message(sse_message)
            if event_type == "done":
                self.emit(request_id, "done", event_payload)
            else:
                self.emit(request_id, event_type, event_payload)

    def _upload(self, request_id: str, payload: dict[str, Any]) -> None:
        raw_path = self._required_string(payload, "path", max_length=4096)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise DesktopProtocolError("请选择有效的本地资料文件")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        result = self._service.upload_knowledge(
            filename=path.name,
            data=path.read_bytes(),
            content_type=content_type,
            domain=self._optional_string(payload, "domain", "uploaded", 64),
            category=self._optional_string(payload, "category", "general", 64),
            version=self._optional_string(payload, "version", "1", 64),
        )
        self.emit(request_id, "knowledge_changed", result)

    def _safe_status(self) -> dict[str, object]:
        # CSRF belongs to HTTP/browser transport and is intentionally absent here.
        return {
            key: value
            for key, value in self._service.status().items()
            if key != "csrf_token"
        }

    @staticmethod
    def _request_id(request: dict[str, Any]) -> str:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise DesktopProtocolError("request_id 格式无效")
        protocol = request.get("protocol", PROTOCOL_VERSION)
        if protocol != PROTOCOL_VERSION:
            raise DesktopProtocolError("桌面协议版本不兼容")
        return request_id

    @classmethod
    def _session_id(cls, payload: dict[str, Any]) -> str:
        return cls._required_string(payload, "session_id", max_length=64)

    @staticmethod
    def _required_string(
        payload: dict[str, Any],
        name: str,
        *,
        max_length: int,
    ) -> str:
        value = payload.get(name)
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized or len(normalized) > max_length:
            raise DesktopProtocolError(f"{name} 不能为空且不能超过 {max_length} 个字符")
        return normalized

    @classmethod
    def _optional_string(
        cls,
        payload: dict[str, Any],
        name: str,
        default: str,
        max_length: int,
    ) -> str:
        if name not in payload:
            return default
        return cls._required_string(payload, name, max_length=max_length)

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if isinstance(error, HTTPException):
            return str(error.detail)
        return str(error)

    def emit(self, request_id: str, event_type: str, payload: object) -> None:
        message = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": event_type,
            "payload": payload,
        }
        # Keep the wire representation ASCII-only. This makes the local NDJSON
        # protocol immune to Windows ANSI code pages even if a frozen Python
        # runtime or launcher ignores its requested UTF-8 stream encoding.
        # json.loads on the Rust side restores every escaped Unicode character.
        self._stdout.write(json.dumps(message, ensure_ascii=True, separators=(",", ":")))
        self._stdout.write("\n")
        self._stdout.flush()


def parse_sse_message(message: str) -> tuple[str, dict[str, Any]]:
    """Decode the trusted SSE records produced by ``WebAgentService``."""

    event_type = "message"
    data_lines: list[str] = []
    for line in message.splitlines():
        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if event_type not in {"trace", "result", "done", "error"} or not data_lines:
        raise DesktopProtocolError("Agent 流式消息格式无效")
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        raise DesktopProtocolError("Agent 流式消息 payload 无效")
    return event_type, payload


def main(
    service_factory: Callable[[], DesktopService] = WebAgentService.from_environment,
) -> None:
    # Pipes do not always inherit UTF-8 on Windows (notably in a PyInstaller
    # executable started outside a terminal).  The desktop protocol is explicitly
    # UTF-8 in both directions, independent of the machine's ANSI code page.
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8", errors="backslashreplace")

    # Reserve the original stdout for NDJSON. Some native/model dependencies print
    # progress messages during initialization; redirect those diagnostics to stderr
    # so they can never corrupt the trusted desktop protocol.
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        service = service_factory()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        message = {
            "protocol": PROTOCOL_VERSION,
            "request_id": "desktop",
            "type": "fatal",
            "payload": {"message": "Campus Agent 初始化失败，请查看本地日志"},
        }
        protocol_stdout.write(
            json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n"
        )
        protocol_stdout.flush()
        raise SystemExit(1) from None
    DesktopProtocolServer(service, stdout=protocol_stdout).serve()


if __name__ == "__main__":
    main()
