from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from campus_agent.mobile_api import (
    MAX_JSON_BODY_BYTES,
    create_mobile_app,
    mobile_api_token_from_environment,
    server_allowed_hosts_from_environment,
)


TEST_TOKEN = "mobile-api-test-token-with-32-characters"


class FakeAgentService:
    def __init__(self) -> None:
        self.stream_requests = []
        self.history_requests: list[str] = []
        self.clear_requests: list[str] = []

    def stream_chat(self, request):
        self.stream_requests.append(request)
        return iter(
            (
                'event: trace\ndata: {"type":"plan","detail":"本地规则"}\n\n',
                'event: result\ndata: {"answer":"周三有机器学习基础"}\n\n',
                'event: done\ndata: {"ok":true}\n\n',
            )
        )

    def history(self, session_id: str) -> list[dict[str, str]]:
        self.history_requests.append(session_id)
        return [{"role": "assistant", "content": "历史回答"}]

    def clear(self, session_id: str) -> None:
        self.clear_requests.append(session_id)


class MobileApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeAgentService()
        self.app = create_mobile_app(
            self.service,  # type: ignore[arg-type]
            api_token=TEST_TOKEN,
            allowed_hosts=["testserver", "127.0.0.1"],
            max_concurrent_chats=1,
        )

    def test_health_is_public_and_sanitized(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "campus-agent",
                "api_version": "v1",
            },
        )
        self.assertNotIn(TEST_TOKEN, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_protected_routes_require_exact_bearer_token(self) -> None:
        with TestClient(self.app) as client:
            missing = client.get(
                "/api/v1/history", params={"session_id": "mobile-1"}
            )
            wrong = client.get(
                "/api/v1/history",
                params={"session_id": "mobile-1"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            valid = client.get(
                "/api/v1/history",
                params={"session_id": "mobile-1"},
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.json()["messages"][0]["content"], "历史回答")
        self.assertEqual(self.service.history_requests, ["mobile-1"])

    def test_stream_and_clear_delegate_to_existing_service(self) -> None:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        with TestClient(self.app) as client:
            stream = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={
                    "query": "周三有什么课程？",
                    "session_id": "mobile-stream",
                    "planner": "rule",
                },
            )
            cleared = client.delete(
                "/api/v1/history",
                headers=headers,
                params={"session_id": "mobile-stream"},
            )

        self.assertEqual(stream.status_code, 200)
        self.assertTrue(stream.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: result", stream.text)
        self.assertIn("周三有机器学习基础", stream.text)
        self.assertEqual(self.service.stream_requests[0].session_id, "mobile-stream")
        self.assertEqual(cleared.json(), {"cleared": True, "session_id": "mobile-stream"})
        self.assertEqual(self.service.clear_requests, ["mobile-stream"])

    def test_body_size_and_chat_concurrency_are_bounded(self) -> None:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        with TestClient(self.app) as client:
            oversized = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                content=b"x" * (MAX_JSON_BODY_BYTES + 1),
            )
            oversized_chunked = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                content=iter((b"x" * MAX_JSON_BODY_BYTES, b"x")),
            )

            self.assertTrue(client.app.state.chat_limiter.acquire(blocking=False))
            try:
                busy = client.post(
                    "/api/v1/chat/stream",
                    headers=headers,
                    json={
                        "query": "测试",
                        "session_id": "busy-session",
                        "planner": "rule",
                    },
                )
            finally:
                client.app.state.chat_limiter.release()

        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized_chunked.status_code, 413)
        self.assertEqual(busy.status_code, 429)
        self.assertEqual(busy.headers["retry-after"], "1")

    def test_token_and_allowed_host_configuration_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "32至256位"):
            create_mobile_app(
                self.service,  # type: ignore[arg-type]
                api_token="too-short",
                allowed_hosts=["testserver"],
            )
        with self.assertRaisesRegex(ValueError, "通配符"):
            create_mobile_app(
                self.service,  # type: ignore[arg-type]
                api_token=TEST_TOKEN,
                allowed_hosts=["*"],
            )

        with patch.dict(os.environ, {"CAMPUS_MOBILE_API_TOKEN": "short"}):
            with self.assertRaisesRegex(ValueError, "32至256位"):
                mobile_api_token_from_environment()

        with patch.dict(
            os.environ,
            {"CAMPUS_SERVER_ALLOWED_HOSTS": "api.example.com"},
        ):
            self.assertEqual(
                server_allowed_hosts_from_environment(),
                ["api.example.com", "127.0.0.1", "localhost", "testserver"],
            )

        with patch.dict(os.environ, {"CAMPUS_SERVER_ALLOWED_HOSTS": "*"}):
            with self.assertRaisesRegex(ValueError, "通配符"):
                server_allowed_hosts_from_environment()


if __name__ == "__main__":
    unittest.main()
