from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from campus_agent.agent import build_default_agent
from campus_agent.deepseek import DeepSeekChatClient, DeepSeekConfig
from campus_agent.deepseek_planner import DeepSeekPlanner
from campus_agent.memory import ConversationMemory
from campus_agent.rag.knowledge_base import KnowledgeBaseService
from campus_agent.tools import CampusKnowledgeTool
from campus_agent.web import WebAgentService, create_app


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("无 API Key 的 Web 测试不应访问网络")

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.knowledge_base = KnowledgeBaseService(
            storage_dir=Path(self.temporary_directory.name)
        )
        knowledge_tool = CampusKnowledgeTool(self.knowledge_base.rag)
        memory = ConversationMemory()
        service = WebAgentService(
            rule_agent=build_default_agent(
                memory=memory,
                knowledge_tool=knowledge_tool,
            ),
            deepseek_agent=build_default_agent(
                planner=DeepSeekPlanner(
                    DeepSeekChatClient(
                        DeepSeekConfig(api_key=None),
                        opener=fail_if_called,
                    )
                ),
                memory=memory,
                knowledge_tool=knowledge_tool,
            ),
            knowledge_base=self.knowledge_base,
        )
        self.client = TestClient(create_app(service))
        self.addCleanup(self.client.close)
        health = self.client.get("/api/health")
        self.csrf_token = health.json()["csrf_token"]
        self.client.headers.update({"X-CSRF-Token": self.csrf_token})

    def test_index_and_local_assets_are_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Campus Agent", response.text)
        self.assertIn('id="message-input"', response.text)
        self.assertNotIn("https://", response.text)

        css = self.client.get("/assets/styles.css")
        javascript = self.client.get("/assets/app.js")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(javascript.status_code, 200)
        self.assertNotIn("https://", javascript.text)
        self.assertNotIn("http://", javascript.text)
        self.assertEqual(self.client.get("/assets/favicon.svg").status_code, 200)

        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("Campus Agent 接口", docs.text)
        self.assertNotIn("https://", docs.text)
        self.assertNotIn("cdn.", docs.text)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)
        self.assertIn("connect-src 'self'", response.headers["content-security-policy"])

    def test_rule_chat_returns_answer_trace_and_metrics(self) -> None:
        response = self.client.post(
            "/api/chat",
            json={
                "query": "周三有什么课程？",
                "session_id": "student-web-1",
                "planner": "rule",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("机器学习基础", payload["answer"])
        self.assertEqual(payload["planner"], "rule")
        self.assertGreaterEqual(payload["elapsed_ms"], 0)
        self.assertEqual(
            [event["type"] for event in payload["events"]],
            ["plan", "tool_result", "verify", "final"],
        )
        self.assertEqual(payload["metrics"]["network_calls"], 0)

    def test_chat_returns_structured_citations_and_accepts_server_filters(self) -> None:
        response = self.client.post(
            "/api/chat",
            json={
                "query": "校园卡丢了怎么补办？",
                "session_id": "citation-session",
                "planner": "rule",
            },
        )

        self.assertEqual(response.status_code, 200)
        citation = response.json()["citations"][0]
        self.assertEqual(citation["source"], "campus_card.md")
        self.assertEqual(citation["document_id"], "builtin-campus_card")
        self.assertNotIn(str(Path.cwd()), json.dumps(citation, ensure_ascii=False))

        filtered = self.client.post(
            "/api/chat",
            json={
                "query": "校园卡丢了怎么补办？",
                "session_id": "filtered-session",
                "planner": "rule",
                "filters": {"domain": "academic"},
            },
        )
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["citations"], [])

        invalid = self.client.post(
            "/api/chat",
            json={
                "query": "校园卡",
                "session_id": "invalid-filter",
                "filters": {"visibility": "private"},
            },
        )
        self.assertEqual(invalid.status_code, 422)

    def test_deepseek_without_key_falls_back_without_network(self) -> None:
        response = self.client.post(
            "/api/chat",
            json={
                "query": "周三有什么课程？",
                "session_id": "deepseek-offline",
                "planner": "deepseek",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("机器学习基础", payload["answer"])
        self.assertEqual(payload["metrics"]["planner"], "deepseek")
        self.assertFalse(payload["metrics"]["api_key_configured"])
        self.assertEqual(payload["metrics"]["api"]["network_attempts"], 0)
        self.assertGreaterEqual(payload["metrics"]["fallback_count"], 1)

    def test_history_context_and_clear_are_session_scoped(self) -> None:
        for session_id in ("session-a", "session-b"):
            self.client.post(
                "/api/chat",
                json={
                    "query": "周三有什么课程？",
                    "session_id": session_id,
                    "planner": "rule",
                },
            )

        follow_up = self.client.post(
            "/api/chat",
            json={
                "query": "在哪里上？",
                "session_id": "session-a",
                "planner": "rule",
            },
        )
        self.assertIn("教五-305", follow_up.json()["answer"])

        history_a = self.client.get(
            "/api/history", params={"session_id": "session-a"}
        ).json()["messages"]
        history_b = self.client.get(
            "/api/history", params={"session_id": "session-b"}
        ).json()["messages"]
        self.assertEqual(len(history_a), 4)
        self.assertEqual(len(history_b), 2)

        cleared = self.client.delete(
            "/api/history", params={"session_id": "session-a"}
        )
        self.assertTrue(cleared.json()["cleared"])
        self.assertEqual(
            self.client.get(
                "/api/history", params={"session_id": "session-a"}
            ).json()["messages"],
            [],
        )
        self.assertEqual(
            len(
                self.client.get(
                    "/api/history", params={"session_id": "session-b"}
                ).json()["messages"]
            ),
            2,
        )

    def test_invalid_input_and_session_id_are_rejected(self) -> None:
        empty = self.client.post(
            "/api/chat",
            json={
                "query": "   ",
                "session_id": "valid-session",
                "planner": "rule",
            },
        )
        invalid_session = self.client.post(
            "/api/chat",
            json={
                "query": "周三有什么课？",
                "session_id": "invalid/session",
                "planner": "rule",
            },
        )
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(invalid_session.status_code, 422)

    def test_health_never_exposes_api_key(self) -> None:
        secret = "test-secret-must-not-leak"
        memory = ConversationMemory()
        service = WebAgentService(
            rule_agent=build_default_agent(memory=memory),
            deepseek_agent=build_default_agent(
                planner=DeepSeekPlanner(
                    DeepSeekChatClient(DeepSeekConfig(api_key=secret))
                ),
                memory=memory,
            ),
        )
        response = TestClient(create_app(service)).get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deepseek_configured"])
        self.assertNotIn(secret, response.text)

    def test_csrf_protects_mutating_api_and_keeps_security_headers(self) -> None:
        request_body = {
            "query": "周三有什么课程？",
            "session_id": "csrf-check",
            "planner": "rule",
        }
        missing = self.client.post(
            "/api/chat",
            json=request_body,
            headers={"X-CSRF-Token": ""},
        )
        wrong = self.client.post(
            "/api/chat",
            json=request_body,
            headers={"X-CSRF-Token": "wrong-token"},
        )
        foreign_origin = self.client.post(
            "/api/chat",
            json=request_body,
            headers={
                "X-CSRF-Token": self.csrf_token,
                "Origin": "https://attacker.example",
            },
        )
        same_origin = self.client.post(
            "/api/chat",
            json=request_body,
            headers={
                "X-CSRF-Token": self.csrf_token,
                "Origin": "http://testserver",
            },
        )

        self.assertEqual(missing.status_code, 403)
        self.assertEqual(wrong.status_code, 403)
        self.assertEqual(foreign_origin.status_code, 403)
        self.assertEqual(same_origin.status_code, 200)
        self.assertEqual(missing.headers["x-content-type-options"], "nosniff")
        self.assertIn("connect-src 'self'", missing.headers["content-security-policy"])

    def test_uploaded_document_is_shared_and_has_full_api_lifecycle(self) -> None:
        markdown = (
            "# 夜间校车指南\n\n"
            "## 星河专线\n\n"
            "星河号每周五22:30从紫荆门发车，预约码为CAMPUS-8842。"
        ).encode("utf-8")
        initial = self.client.get("/api/knowledge/documents").json()
        initial_version = initial["stats"]["index_version"]
        self.assertEqual(initial["documents"], [])

        uploaded = self.client.post(
            "/api/knowledge/documents",
            params={
                "filename": "night-bus.md",
                "domain": "transport",
                "category": "night_bus",
                "version": "2026.1",
            },
            content=markdown,
            headers={"Content-Type": "text/markdown"},
        )
        self.assertEqual(uploaded.status_code, 201)
        upload_payload = uploaded.json()
        document_id = upload_payload["document"]["document_id"]
        self.assertGreater(upload_payload["stats"]["index_version"], initial_version)
        self.assertEqual(upload_payload["stats"]["uploaded_documents"], 1)
        self.assertEqual(upload_payload["document"]["domain"], "transport")
        self.assertEqual(upload_payload["document"]["category"], "night_bus")
        self.assertEqual(upload_payload["document"]["version"], "2026.1")

        listing = self.client.get("/api/knowledge/documents")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["documents"][0]["display_name"], "night-bus.md")
        self.assertNotIn(str(Path(self.temporary_directory.name)), listing.text)
        self.assertNotIn("紫荆门发车", listing.text)

        for planner in ("rule", "deepseek"):
            with self.subTest(planner=planner):
                answer = self.client.post(
                    "/api/chat",
                    json={
                        "query": "星河号几点从哪里发车？",
                        "session_id": f"uploaded-{planner}",
                        "planner": planner,
                    },
                )
                self.assertEqual(answer.status_code, 200)
                self.assertIn("22:30", answer.json()["answer"])
                self.assertIn("night-bus.md", answer.json()["answer"])

        filtered = self.client.post(
            "/api/chat",
            json={
                "query": "星河号几点从哪里发车？",
                "session_id": "uploaded-filtered",
                "planner": "rule",
                "filters": {
                    "domain": "transport",
                    "category": "night_bus",
                    "version": "2026.1",
                },
            },
        )
        self.assertEqual(filtered.status_code, 200)
        self.assertIn("22:30", filtered.json()["answer"])

        mixed = self.client.post(
            "/api/chat",
            json={
                "query": "周三有什么课程，另外星河号几点发车？",
                "session_id": "uploaded-mixed-tools",
                "planner": "rule",
            },
        )
        self.assertEqual(mixed.status_code, 200)
        self.assertIn("机器学习基础", mixed.json()["answer"])
        self.assertIn("22:30", mixed.json()["answer"])
        self.assertEqual(
            [event["type"] for event in mixed.json()["events"]],
            [
                "plan",
                "tool_result",
                "retrieve",
                "tool_result",
                "grade",
                "verify",
                "final",
            ],
        )

        rebuilt = self.client.post("/api/knowledge/rebuild")
        self.assertEqual(rebuilt.status_code, 200)
        self.assertGreater(
            rebuilt.json()["index_version"],
            upload_payload["stats"]["index_version"],
        )

        deleted = self.client.delete(f"/api/knowledge/documents/{document_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["stats"]["uploaded_documents"], 0)
        self.assertEqual(
            self.client.delete(f"/api/knowledge/documents/{document_id}").status_code,
            404,
        )
        after_delete = self.client.post(
            "/api/chat",
            json={
                "query": "星河号几点从哪里发车？",
                "session_id": "uploaded-after-delete",
                "planner": "rule",
            },
        )
        self.assertEqual(after_delete.status_code, 200)
        self.assertNotIn("22:30", after_delete.json()["answer"])

    def test_knowledge_upload_rejects_unsafe_or_invalid_input(self) -> None:
        cases = (
            ("manual.exe", b"hello", "application/octet-stream", 415),
            ("../manual.md", b"# title\ntext", "text/markdown", 422),
            ("manual.md", b"", "text/markdown", 422),
            ("manual.txt", b"\xff\xfe", "text/plain", 422),
            ("manual.pdf", b"not-a-pdf", "application/pdf", 422),
            ("manual.md", b"# title\ntext", "application/pdf", 415),
        )
        for filename, body, content_type, status_code in cases:
            with self.subTest(filename=filename, content_type=content_type):
                response = self.client.post(
                    "/api/knowledge/documents",
                    params={"filename": filename},
                    content=body,
                    headers={"Content-Type": content_type},
                )
                self.assertEqual(response.status_code, status_code)

        first = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "first.md"},
            content=b"# Unique\n\nDuplicate body marker",
            headers={"Content-Type": "text/markdown"},
        )
        duplicate = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "second.md"},
            content=b"# Unique\n\nDuplicate body marker",
            headers={"Content-Type": "text/markdown"},
        )
        self.assertEqual(first.status_code, 201)
        self.assertEqual(duplicate.status_code, 409)

        unsafe_metadata = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "metadata.md", "domain": "../private"},
            content=b"# Metadata\n\nUnsafe label",
            headers={"Content-Type": "text/markdown"},
        )
        self.assertEqual(unsafe_metadata.status_code, 422)

    def test_course_process_document_routes_to_rag_not_timetable_prompt(self) -> None:
        uploaded = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "course-withdrawal.md"},
            content=(
                "# 课程退选指南\n\n## 办理流程\n\n"
                "退选申请须在第2周前提交，办理代码为DROP-202。"
            ).encode("utf-8"),
            headers={"Content-Type": "text/markdown"},
        )
        self.assertEqual(uploaded.status_code, 201)

        response = self.client.post(
            "/api/chat",
            json={
                "query": "课程退选流程是什么？",
                "session_id": "course-process-rag",
                "planner": "rule",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("DROP-202", response.json()["answer"])
        self.assertNotIn("星期几", response.json()["answer"])

    def test_invalid_deepseek_environment_keeps_rule_web_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"DEEPSEEK_TIMEOUT_SECONDS": "invalid"},
        ):
            service = WebAgentService.from_environment(
                knowledge_storage_dir=Path(directory)
            )
            client = TestClient(create_app(service))

            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertFalse(health.json()["deepseek_configured"])
            self.assertIsNotNone(health.json()["deepseek_configuration_error"])
            client.headers.update({"X-CSRF-Token": health.json()["csrf_token"]})

            response = client.post(
                "/api/chat",
                json={
                    "query": "周三有什么课程？",
                    "session_id": "invalid-config-rule",
                    "planner": "rule",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("机器学习基础", response.json()["answer"])
            client.close()
            service.close()


if __name__ == "__main__":
    unittest.main()
