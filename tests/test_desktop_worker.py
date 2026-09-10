from __future__ import annotations

import io
import json

from campus_agent.desktop_worker import (
    PROTOCOL_VERSION,
    DesktopProtocolServer,
    parse_sse_message,
)


class FakeDesktopService:
    def __init__(self) -> None:
        self.closed = False
        self.cleared: list[str] = []

    def status(self):
        return {
            "status": "ok",
            "deepseek_configured": False,
            "csrf_token": "must-not-cross-desktop-boundary",
        }

    def stream_chat(self, request):
        yield 'event: trace\ndata: {"type":"plan","detail":"调用知识工具"}\n\n'
        yield (
            'event: result\ndata: {"answer":"请先挂失。","events":[],"citations":[],'
            '"metrics":{"planner":"rule"},"planner":"rule","elapsed_ms":12}\n\n'
        )
        yield 'event: done\ndata: {"ok":true}\n\n'

    def history(self, session_id):
        return [{"role": "user", "content": session_id}]

    def clear(self, session_id):
        self.cleared.append(session_id)

    def knowledge_documents(self):
        return {"documents": [], "stats": {"chunk_count": 5}}

    def upload_knowledge(self, **kwargs):
        return {"document": {"display_name": kwargs["filename"]}}

    def delete_knowledge(self, document_id):
        return {"deleted": True, "document_id": document_id}

    def rebuild_knowledge(self):
        return {"chunk_count": 5}

    def close(self):
        self.closed = True


def _request(command: str, payload: dict[str, object] | None = None) -> str:
    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": f"req-{command}",
            "command": command,
            "payload": payload or {},
        },
        ensure_ascii=False,
    )


def _serve(*requests: str):
    service = FakeDesktopService()
    stdout = io.StringIO()
    server = DesktopProtocolServer(
        service,
        stdin=io.StringIO("\n".join(requests) + "\n"),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    server.serve()
    return service, [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_parse_sse_message_decodes_event_and_payload():
    event_type, payload = parse_sse_message(
        'event: trace\ndata: {"type":"plan","detail":"ok"}\n\n'
    )
    assert event_type == "trace"
    assert payload == {"type": "plan", "detail": "ok"}


def test_protocol_wire_is_ascii_only_and_round_trips_unicode():
    service = FakeDesktopService()
    stdout = io.StringIO()
    server = DesktopProtocolServer(
        service,
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    server.emit("req-unicode", "result", {"answer": "校园卡丢了怎么办？"})

    wire = stdout.getvalue()
    assert wire.isascii()
    assert json.loads(wire)["payload"]["answer"] == "校园卡丢了怎么办？"


def test_protocol_serves_chat_stream_without_exposing_csrf():
    service, messages = _serve(
        _request(
            "chat",
            {
                "query": "校园卡丢了怎么办？",
                "session_id": "desktop-default",
                "planner": "rule",
            },
        ),
        _request("shutdown"),
    )

    assert service.closed is True
    assert messages[0]["type"] == "ready"
    assert "csrf_token" not in messages[0]["payload"]
    assert [message["type"] for message in messages[1:5]] == [
        "trace",
        "result",
        "done",
        "shutdown",
    ]
    assert messages[2]["payload"]["answer"] == "请先挂失。"


def test_shutdown_does_not_wait_for_another_input_line():
    service, messages = _serve(
        _request("shutdown"),
        _request("history", {"session_id": "must-not-run"}),
    )

    assert service.closed is True
    assert [message["type"] for message in messages] == ["ready", "shutdown"]


def test_protocol_supports_history_clear_and_knowledge():
    service, messages = _serve(
        _request("history", {"session_id": "desktop-default"}),
        _request("clear_history", {"session_id": "desktop-default"}),
        _request("list_knowledge"),
    )

    assert service.cleared == ["desktop-default"]
    assert [message["type"] for message in messages[1:]] == [
        "history",
        "history_cleared",
        "knowledge",
    ]


def test_protocol_rejects_unknown_commands_without_internal_details():
    _, messages = _serve(_request("run_arbitrary_code"))

    assert messages[1]["type"] == "error"
    assert messages[1]["payload"]["message"] == "不支持的桌面命令"
