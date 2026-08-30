from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from campus_agent.agent import build_default_agent
from campus_agent.business_api import (
    BusinessAPIError,
    CampusBusinessGateway,
    MockCampusBusinessGateway,
)
from campus_agent.deepseek_planner import DeepSeekPlanner
from campus_agent.domain import ToolResult
from campus_agent.tools import CampusServiceStatusTool
from campus_agent.web import WebAgentService, create_app


class FailingGateway(CampusBusinessGateway):
    def query_service_status(self, service_code: str, campus: str):
        raise BusinessAPIError(
            "upstream_unavailable",
            "校园业务服务暂时不可用",
            retryable=True,
        )


class FakeComposerClient:
    config = None

    def complete_json(self, messages):
        return {"answer": "校园卡中心当前开放。"}

    def metrics(self):
        return {}


class BusinessAgentIntegrationTests(unittest.TestCase):
    def build_agent(self, gateway=None):
        tool = CampusServiceStatusTool(gateway or MockCampusBusinessGateway())
        return build_default_agent(extra_tools=(tool,))

    def test_live_status_uses_business_tool_without_static_rag(self) -> None:
        response = self.build_agent().ask("校园卡服务中心现在开门吗？")

        self.assertIn("[模拟数据]", response.answer)
        self.assertIn("校园卡服务中心", response.answer)
        self.assertEqual(response.citations, ())
        trace = "\n".join(event.detail for event in response.events)
        self.assertIn("query_campus_service_status", trace)
        self.assertNotIn("search_campus_knowledge", trace)

    def test_mixed_process_and_live_status_uses_api_and_rag(self) -> None:
        response = self.build_agent().ask(
            "校园卡丢了怎么办，服务中心现在开门吗？"
        )

        self.assertIn("[模拟数据]", response.answer)
        self.assertIn("挂失", response.answer)
        self.assertEqual(response.citations[0].document_id, "builtin-campus_card")
        trace = "\n".join(event.detail for event in response.events)
        self.assertIn("query_campus_service_status", trace)
        self.assertIn("search_campus_knowledge", trace)

    def test_mixed_query_sends_only_static_clause_to_rag(self) -> None:
        response = self.build_agent().ask(
            "校园卡补办流程是什么，服务中心现在几点开门？"
        )

        self.assertIn("[模拟数据]", response.answer)
        self.assertIn("挂失与补办", response.answer)
        self.assertNotIn("没有提供你询问的具体开放或服务时间", response.answer)

    def test_missing_service_is_clarified_instead_of_guessed(self) -> None:
        response = self.build_agent().ask("现在开门吗？")

        self.assertIn("哪个服务", response.answer)
        self.assertEqual(response.citations, ())

    def test_api_failure_is_explicit_and_never_fabricates_status(self) -> None:
        response = self.build_agent(FailingGateway()).ask(
            "校园卡服务中心现在开门吗？"
        )

        self.assertIn("暂时不可用", response.answer)
        self.assertNotIn("当前状态：开放", response.answer)

    def test_deepseek_composer_cannot_remove_mock_label(self) -> None:
        planner = DeepSeekPlanner(FakeComposerClient())
        answer = planner.compose(
            "校园卡中心现在开门吗？",
            (
                ToolResult(
                    call_id="1",
                    tool_name="query_campus_service_status",
                    ok=True,
                    output="[模拟数据] 校园卡服务中心当前状态：开放。",
                ),
            ),
        )

        self.assertTrue(answer.startswith("[模拟数据]"))

    def test_health_exposes_mode_but_never_business_secret_or_url(self) -> None:
        secret = "official-secret-must-not-leak"
        base_url = "https://authorized-campus.example/api"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "CAMPUS_BUSINESS_API_MODE": "official",
                "CAMPUS_BUSINESS_API_BASE_URL": base_url,
                "CAMPUS_BUSINESS_API_TOKEN": secret,
            },
            clear=False,
        ):
            service = WebAgentService.from_environment(
                knowledge_storage_dir=Path(directory) / "knowledge",
                checkpoint_path=Path(directory) / "checkpoint.sqlite3",
            )
            try:
                with TestClient(create_app(service)) as client:
                    response = client.get("/api/health")
            finally:
                service.close()

        self.assertEqual(response.status_code, 200)
        status = response.json()["campus_business_api"]
        self.assertEqual(status["mode"], "official")
        self.assertTrue(status["network_enabled"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn(base_url, response.text)

    def test_invalid_official_config_does_not_fall_back_to_mock(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "CAMPUS_BUSINESS_API_MODE": "official",
                "CAMPUS_BUSINESS_API_BASE_URL": "",
                "CAMPUS_BUSINESS_API_TOKEN": "",
            },
            clear=False,
        ):
            service = WebAgentService.from_environment(
                knowledge_storage_dir=Path(directory) / "knowledge",
                checkpoint_path=Path(directory) / "checkpoint.sqlite3",
            )
            try:
                status = service.status()["campus_business_api"]
            finally:
                service.close()

        self.assertEqual(status["mode"], "invalid")
        self.assertFalse(status["configured"])


if __name__ == "__main__":
    unittest.main()
