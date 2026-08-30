from __future__ import annotations

import asyncio
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from campus_agent.agent import build_default_agent
from campus_agent.checkpoint import PrunableSqliteSaver
from campus_agent.domain import Plan, ToolCall
from campus_agent.rag.service import LocalRAG
from campus_agent.tools import CampusKnowledgeTool
from campus_agent.web import (
    ChatRequest,
    WebAgentService,
    _closeable_async_iterator,
    create_app,
)


class FailingPlanPlanner:
    def plan(self, query, history, tool_descriptions):
        raise RuntimeError("planner failed")

    def compose(self, query, results):
        raise AssertionError("compose should not run")


class KnowledgeThenFailPlanner:
    def __init__(self, *, fail_compose: bool = True) -> None:
        self.fail_compose = fail_compose

    def plan(self, query, history, tool_descriptions):
        return Plan(
            summary="检索私有测试资料",
            tool_calls=(
                ToolCall("knowledge-1", "search_campus_knowledge", {"query": query}),
            ),
        )

    def compose(self, query, results):
        if self.fail_compose:
            raise RuntimeError("compose failed")
        return results[0].output


class CheckpointAndStreamingTests(unittest.TestCase):
    def _service(self, directory: str) -> WebAgentService:
        return WebAgentService.from_environment(
            knowledge_storage_dir=Path(directory),
        )

    @staticmethod
    def _checkpoint(directory: str):
        path = Path(directory) / "checkpoint.sqlite3"
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("PRAGMA secure_delete=ON")
        saver = PrunableSqliteSaver(
            connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            ),
        )
        saver.setup()
        return path, connection, saver

    def test_sqlite_checkpoint_restores_history_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._service(directory)
            self.assertTrue(first.status()["checkpoint"]["enabled"])
            first.ask(
                ChatRequest(
                    query="周三有什么课程？",
                    session_id="restart-session",
                    planner="rule",
                )
            )
            first.close()

            restarted = self._service(directory)
            history = restarted.history("restart-session")
            self.assertEqual(len(history), 2)
            follow_up = restarted.ask(
                ChatRequest(
                    query="在哪里上？",
                    session_id="restart-session",
                    planner="rule",
                )
            )
            self.assertIn("教五-305", follow_up["answer"])
            self.assertEqual(len(restarted.history("restart-session")), 4)
            self.assertEqual(restarted.history("different-session"), [])
            restarted.close()

    def test_clear_deletes_persistent_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(directory)
            service.ask(
                ChatRequest(
                    query="周一有什么课程？",
                    session_id="clear-persistent",
                )
            )
            service.clear("clear-persistent")
            service.close()

            restarted = self._service(directory)
            self.assertEqual(restarted.history("clear-persistent"), [])
            restarted.close()

    def test_api_key_is_not_written_to_checkpoint_database(self) -> None:
        secret = "p4-secret-key-must-never-be-persisted"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": secret},
        ):
            service = self._service(directory)
            service.ask(
                ChatRequest(
                    query="校园卡丢了怎么补办？",
                    session_id="secret-check",
                    planner="rule",
                )
            )
            service.close()
            database_path = Path(directory) / ".agent-checkpoints.sqlite3"
            self.assertNotIn(secret.encode(), database_path.read_bytes())

            connection = sqlite3.connect(database_path)
            try:
                checkpoint_count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    ("campus:secret-check",),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(checkpoint_count, 1)

    def test_post_sse_streams_trace_then_result_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(directory)
            client = TestClient(create_app(service))
            health = client.get("/api/health").json()
            response = client.post(
                "/api/chat/stream",
                headers={"X-CSRF-Token": health["csrf_token"]},
                json={
                    "query": "校园卡丢了怎么补办？",
                    "session_id": "sse-session",
                    "planner": "rule",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            event_names = re.findall(r"^event: (\w+)$", response.text, re.MULTILINE)
            self.assertEqual(event_names[-2:], ["result", "done"])
            self.assertTrue(all(name == "trace" for name in event_names[:-2]))
            self.assertIn('"type":"retrieve"', response.text)
            self.assertIn('"type":"grade"', response.text)
            self.assertIn('"type":"verify"', response.text)
            self.assertNotIn('"retrieved_documents"', response.text)
            self.assertEqual(len(service.history("sse-session")), 2)
            client.close()
            service.close()

    def test_stream_endpoint_is_csrf_protected_before_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(directory)
            client = TestClient(create_app(service))

            response = client.post(
                "/api/chat/stream",
                json={"query": "周三有什么课？", "session_id": "csrf-stream"},
            )

            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["detail"], "CSRF校验失败")
            client.close()
            service.close()

    def test_failed_turn_rolls_back_and_does_not_persist_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, connection, saver = self._checkpoint(directory)
            good_agent = build_default_agent(checkpointer=saver)
            good_agent.ask("周一有什么课程？", session_id="atomic")
            before_id = saver.latest_id("campus:atomic")

            knowledge_dir = Path(directory) / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "secret.md").write_text(
                "# 私有资料\n\n## 标记\n\nULTRA-PRIVATE-DOC-77 仅用于测试。",
                encoding="utf-8",
            )
            failing_agent = build_default_agent(
                planner=KnowledgeThenFailPlanner(),
                knowledge_tool=CampusKnowledgeTool(LocalRAG(knowledge_dir)),
                checkpointer=saver,
            )

            with self.assertRaisesRegex(RuntimeError, "compose failed"):
                failing_agent.ask(
                    "FAILED-QUERY-SECRET ULTRA-PRIVATE-DOC-77",
                    session_id="atomic",
                )

            self.assertEqual(saver.latest_id("campus:atomic"), before_id)
            self.assertEqual(len(good_agent.history("atomic")), 2)
            rows = connection.execute(
                "SELECT checkpoint FROM checkpoints UNION ALL SELECT value FROM writes"
            ).fetchall()
            persisted = b"".join(bytes(row[0]) for row in rows if row[0] is not None)
            self.assertNotIn(b"FAILED-QUERY-SECRET", persisted)
            self.assertNotIn(b"ULTRA-PRIVATE-DOC-77", persisted)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    ("campus:atomic",),
                ).fetchone()[0],
                1,
            )
            connection.close()
            self.assertTrue(path.exists())

    def test_first_failed_turn_leaves_no_checkpoint_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, connection, saver = self._checkpoint(directory)
            agent = build_default_agent(
                planner=FailingPlanPlanner(),
                checkpointer=saver,
            )

            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                agent.ask("FIRST-FAIL-SECRET", session_id="first-failure")

            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    ("campus:first-failure",),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM writes WHERE thread_id = ?",
                    ("campus:first-failure",),
                ).fetchone()[0],
                0,
            )
            connection.close()

    def test_closing_stream_mid_turn_rolls_back_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, connection, saver = self._checkpoint(directory)
            knowledge_dir = Path(directory) / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "secret.md").write_text(
                "# 私有资料\n\n## 标记\n\nSTREAM-PRIVATE-DOC-88。",
                encoding="utf-8",
            )
            agent = build_default_agent(
                planner=KnowledgeThenFailPlanner(fail_compose=False),
                knowledge_tool=CampusKnowledgeTool(LocalRAG(knowledge_dir)),
                checkpointer=saver,
            )
            stream = agent.stream(
                "STREAM-PRIVATE-DOC-88 是什么？",
                session_id="disconnect",
            )
            for item in stream:
                event = item.get("event")
                if getattr(event, "event_type", None) == "tool_result":
                    break
            stream.close()

            self.assertEqual(agent.history("disconnect"), ())
            self.assertIsNone(saver.latest_id("campus:disconnect"))
            rows = connection.execute(
                "SELECT checkpoint FROM checkpoints UNION ALL SELECT value FROM writes"
            ).fetchall()
            persisted = b"".join(bytes(row[0]) for row in rows if row[0] is not None)
            self.assertNotIn(b"STREAM-PRIVATE-DOC-88", persisted)
            connection.close()

    def test_many_turns_keep_only_latest_physical_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(directory)
            for index in range(15):
                marker = " OLDEST-PHYSICAL-SECRET-000" if index == 0 else ""
                service.ask(
                    ChatRequest(
                        query=f"周一有什么课程？第{index}次{marker}",
                        session_id="bounded-history",
                    )
                )
            self.assertEqual(len(service.history("bounded-history")), 20)
            connection = sqlite3.connect(
                Path(directory) / ".agent-checkpoints.sqlite3"
            )
            try:
                count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    ("campus:bounded-history",),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)
            current_rows = service._checkpoint_connection.execute(
                "SELECT checkpoint FROM checkpoints UNION ALL SELECT value FROM writes"
            ).fetchall()
            current_state = b"".join(
                bytes(row[0]) for row in current_rows if row[0] is not None
            )
            self.assertNotIn(b"OLDEST-PHYSICAL-SECRET-000", current_state)
            wal_path = Path(directory) / ".agent-checkpoints.sqlite3-wal"
            if wal_path.exists():
                self.assertNotIn(
                    b"OLDEST-PHYSICAL-SECRET-000",
                    wal_path.read_bytes(),
                )
            service.close()

    def test_rule_and_deepseek_share_one_bounded_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": ""},
        ):
            service = self._service(directory)
            service.ask(
                ChatRequest(query="周一有什么课程？", session_id="shared", planner="rule")
            )
            service.ask(
                ChatRequest(query="周三呢？", session_id="shared", planner="deepseek")
            )
            self.assertEqual(len(service.history("shared")), 4)
            connection = sqlite3.connect(
                Path(directory) / ".agent-checkpoints.sqlite3"
            )
            try:
                count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    ("campus:shared",),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)
            service.close()

    def test_owned_app_defers_service_creation_until_lifespan(self) -> None:
        fake_service = MagicMock(spec=WebAgentService)
        fake_service.status.return_value = {"status": "ok"}
        with patch.object(
            WebAgentService,
            "from_environment",
            return_value=fake_service,
        ):
            owned_app = create_app()
            self.assertIsNone(owned_app.state.agent_service)

            with TestClient(owned_app) as client:
                self.assertEqual(client.get("/api/health").json(), {"status": "ok"})

        fake_service.close.assert_called_once_with()

    def test_async_sse_wrapper_closes_sync_iterator_on_disconnect(self) -> None:
        class TrackingIterator:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return "event: trace\ndata: {}\n\n"

            def close(self) -> None:
                self.closed = True

        iterator = TrackingIterator()

        async def consume_one_then_disconnect() -> None:
            stream = _closeable_async_iterator(iterator)
            await anext(stream)
            await stream.aclose()

        asyncio.run(consume_one_then_disconnect())

        self.assertTrue(iterator.closed)

    def test_runtime_sqlite_failure_is_reported_as_unhealthy_503(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(directory)
            assert service._checkpoint_connection is not None
            service._checkpoint_connection.close()

            with self.assertRaises(HTTPException) as captured:
                service.history("broken-storage")

            self.assertEqual(getattr(captured.exception, "status_code", None), 503)
            checkpoint = service.status()["checkpoint"]
            self.assertFalse(checkpoint["healthy"])
            self.assertIn("ProgrammingError", checkpoint["error"])
            service._checkpoint_connection = None
            service.close()


if __name__ == "__main__":
    unittest.main()
