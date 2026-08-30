from __future__ import annotations
import threading
import unittest
from unittest.mock import patch
import httpx
from campus_agent.business_api import (
    BusinessAPIConfig,
    BusinessAPIError,
    CachingCampusBusinessGateway,
    MockCampusBusinessGateway,
    OfficialHttpCampusBusinessGateway,
)
from campus_agent.business_api.gateway import CampusBusinessGateway
from campus_agent.tools.business import CampusServiceStatusTool
from campus_agent.domain import ToolCall
from campus_agent.tooling import Tool, ToolRegistry

def payload(**updates):
    value = {"service_code": "campus_card", "service_name": "校园卡中心", "campus": "main", "status": "open", "location": "行政楼", "today_hours": "08:30-17:30", "updated_at": "2026-08-29T09:00:00+08:00", "queue_count": 3, "estimated_wait_minutes": 6, "request_id": "req-1"}
    value.update(updates)
    return value

class OfficialGatewayTests(unittest.TestCase):
    def make(self, handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        return OfficialHttpCampusBusinessGateway(base_url="https://campus.example.edu/api", token="secret-token", client=client, sleep=lambda _: None, **kwargs)
    def test_fixed_get_auth_and_canonical_mapping(self):
        seen = {}
        def handler(request):
            seen.update(method=request.method, url=str(request.url), auth=request.headers.get("authorization"))
            return httpx.Response(200, json=payload())
        result = self.make(handler).query_service_status("campus_card", "main")
        self.assertEqual(seen["method"], "GET")
        self.assertEqual(seen["auth"], "Bearer secret-token")
        self.assertTrue(seen["url"].startswith("https://campus.example.edu/api/v1/campus/service-status?"))
        self.assertEqual(result.provider, "official")
        self.assertNotIn("secret", str(result.as_dict()))
    def test_https_required(self):
        with self.assertRaises(ValueError): OfficialHttpCampusBusinessGateway(base_url="http://campus.test", token="x")

    def test_private_hosts_and_hosts_outside_allowlist_are_rejected(self):
        for base_url in (
            "https://localhost",
            "https://127.0.0.1",
            "https://169.254.169.254",
            "https://10.0.0.1",
        ):
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                OfficialHttpCampusBusinessGateway(base_url=base_url, token="x")
        with self.assertRaisesRegex(ValueError, "允许列表"):
            OfficialHttpCampusBusinessGateway(
                base_url="https://campus.example.edu",
                token="x",
                allowed_hosts={"official.example.edu"},
            )
    def test_retries_once_on_503(self):
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=payload())
        self.assertEqual(self.make(handler).query_service_status("campus_card", "main").status, "open")
        self.assertEqual(len(calls), 2)

    def test_429_retry_after_is_capped_without_real_sleep(self):
        calls = []
        sleeps = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after": "999"})
            return httpx.Response(200, json=payload())

        client = httpx.Client(transport=httpx.MockTransport(handler))
        gateway = OfficialHttpCampusBusinessGateway(
            base_url="https://campus.example.edu/api",
            token="secret-token",
            client=client,
            sleep=sleeps.append,
        )
        gateway.query_service_status("campus_card", "main")
        self.assertEqual(sleeps, [gateway.MAX_RETRY_AFTER_SECONDS])

    def test_updated_at_requires_iso8601_timezone(self):
        gateway = self.make(
            lambda request: httpx.Response(
                200, json=payload(updated_at="2026-08-29T09:00:00")
            )
        )
        with self.assertRaisesRegex(BusinessAPIError, "数据不完整"):
            gateway.query_service_status("campus_card", "main")
    def test_does_not_retry_auth_failure_or_fallback_to_mock(self):
        calls = []
        gateway = self.make(lambda request: (calls.append(1), httpx.Response(401))[1])
        with self.assertRaisesRegex(BusinessAPIError, "鉴权失败"):
            gateway.query_service_status("campus_card", "main")
        self.assertEqual(len(calls), 1)
    def test_redirect_is_rejected(self):
        with self.assertRaisesRegex(BusinessAPIError, "重定向"):
            self.make(lambda request: httpx.Response(302, headers={"location": "https://evil.test"})).query_service_status("campus_card", "main")
    def test_content_type_size_and_schema_are_enforced(self):
        cases = [
            (lambda r: httpx.Response(200, text="not json"), {}),
            (lambda r: httpx.Response(200, content=b"x" * 20, headers={"content-type": "application/json"}), {"max_response_bytes": 10}),
            (lambda r: httpx.Response(200, json=payload(service_code="library")), {}),
        ]
        for handler, kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(BusinessAPIError):
                self.make(handler, **kwargs).query_service_status(
                    "campus_card", "main"
                )

    def test_compressed_response_is_rejected_before_body_use(self):
        gateway = self.make(
            lambda request: httpx.Response(
                200,
                stream=httpx.ByteStream(b"compressed"),
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                },
            )
        )
        with self.assertRaisesRegex(BusinessAPIError, "压缩"):
            gateway.query_service_status("campus_card", "main")

    def test_remote_protocol_error_is_retried_and_normalized(self):
        calls = []

        def handler(request):
            calls.append(1)
            raise httpx.RemoteProtocolError("broken response")

        with self.assertRaisesRegex(BusinessAPIError, "暂时不可用"):
            self.make(handler).query_service_status("campus_card", "main")
        self.assertEqual(len(calls), 2)

    def test_monotonic_deadline_limits_whole_operation(self):
        now = [0.0]

        def handler(request):
            now[0] = 5.0
            return httpx.Response(200, json=payload())

        client = httpx.Client(transport=httpx.MockTransport(handler))
        gateway = OfficialHttpCampusBusinessGateway(
            base_url="https://campus.example.edu/api",
            token="secret-token",
            client=client,
            sleep=lambda delay: None,
            clock=lambda: now[0],
        )
        with self.assertRaisesRegex(BusinessAPIError, "超时"):
            gateway.query_service_status("campus_card", "main")

    def test_request_id_must_be_a_safe_string(self):
        gateway = self.make(
            lambda request: httpx.Response(200, json=payload(request_id={"secret": 1}))
        )
        with self.assertRaisesRegex(BusinessAPIError, "数据不完整"):
            gateway.query_service_status("campus_card", "main")

class CacheAndToolTests(unittest.TestCase):
    def test_mock_is_explicit_and_cache_hits(self):
        now = [0.0]
        cache = CachingCampusBusinessGateway(
            MockCampusBusinessGateway(), clock=lambda: now[0]
        )
        tool = CampusServiceStatusTool(cache)
        first = tool.invoke(service_code="campus_card")
        second = tool.invoke(service_code="campus_card")
        self.assertIn("[模拟数据]", first.text)
        self.assertFalse(first.data["cache_hit"])
        self.assertTrue(second.data["cache_hit"])
    def test_enum_validation(self):
        with self.assertRaises(ValueError): MockCampusBusinessGateway().query_service_status("payments", "main")
    def test_stale_only_for_retryable_failure(self):
        class Flaky(CampusBusinessGateway):
            calls = 0
            def query_service_status(self, service_code, campus):
                self.calls += 1
                if self.calls > 1:
                    raise BusinessAPIError(
                        "upstream_unavailable", "down", retryable=True
                    )
                return MockCampusBusinessGateway().query_service_status(service_code, campus)
        now = [0.0]
        cache = CachingCampusBusinessGateway(
            Flaky(), ttl_seconds=1, stale_seconds=10, clock=lambda: now[0]
        )
        cache.query_service_status("library", "main")
        now[0] = 2
        self.assertTrue(cache.query_service_status("library", "main").stale)
    def test_singleflight(self):
        class Blocking(CampusBusinessGateway):
            calls = 0
            def query_service_status(self, service_code, campus):
                self.calls += 1
                gate.wait(1)
                return MockCampusBusinessGateway().query_service_status(
                    service_code, campus
                )

        gate = threading.Event()
        source = Blocking()
        cache = CachingCampusBusinessGateway(source)
        threads = [threading.Thread(target=lambda: cache.query_service_status("library", "main")) for _ in range(4)]
        for thread in threads: thread.start()
        gate.set()
        for thread in threads: thread.join()
        self.assertEqual(source.calls, 1)

    def test_cache_constructor_rejects_non_finite_durations(self):
        for ttl, stale in ((float("nan"), 300), (30, float("inf"))):
            with self.subTest(ttl=ttl, stale=stale):
                with self.assertRaises(ValueError):
                    CachingCampusBusinessGateway(
                        MockCampusBusinessGateway(),
                        ttl_seconds=ttl,
                        stale_seconds=stale,
                    )
    def test_registry_redacts_unexpected_exception(self):
        class Bad(Tool):
            name = "bad"
            description = "bad"
            parameters = {}

            def run(self, **arguments):
                raise RuntimeError("token=super-secret")

        registry = ToolRegistry()
        registry.register(Bad())
        result = registry.execute(ToolCall("1", "bad", {}))
        self.assertFalse(result.ok)
        self.assertNotIn("super-secret", result.error or "")


class BusinessAPIConfigTests(unittest.TestCase):
    def test_numeric_cache_environment_is_parsed(self):
        env = {
            "CAMPUS_BUSINESS_API_MODE": "mock",
            "CAMPUS_BUSINESS_API_CACHE_TTL": "12.5",
            "CAMPUS_BUSINESS_API_CACHE_STALE": "60",
        }
        with patch.dict("os.environ", env, clear=True):
            config = BusinessAPIConfig.from_env()
        self.assertEqual(config.cache_ttl_seconds, 12.5)
        self.assertEqual(config.cache_stale_seconds, 60)

    def test_invalid_numeric_cache_environment_is_rejected(self):
        invalid_environments = [
            {"CAMPUS_BUSINESS_API_CACHE_TTL": "not-a-number"},
            {"CAMPUS_BUSINESS_API_CACHE_TTL": "nan"},
            {"CAMPUS_BUSINESS_API_CACHE_TTL": "inf"},
            {
                "CAMPUS_BUSINESS_API_CACHE_TTL": "30",
                "CAMPUS_BUSINESS_API_CACHE_STALE": "10",
            },
        ]
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                with patch.dict("os.environ", environment, clear=True):
                    with self.assertRaises(ValueError):
                        BusinessAPIConfig.from_env()

    def test_base_url_and_token_are_stripped(self):
        with patch.dict(
            "os.environ",
            {
                "CAMPUS_BUSINESS_API_MODE": "official",
                "CAMPUS_BUSINESS_API_BASE_URL": "  https://campus.example.edu  ",
                "CAMPUS_BUSINESS_API_TOKEN": "  secret  ",
            },
            clear=True,
        ):
            config = BusinessAPIConfig.from_env()
        self.assertEqual(config.base_url, "https://campus.example.edu")
        self.assertEqual(config.token, "secret")

    def test_official_allowed_hosts_are_normalized(self):
        with patch.dict(
            "os.environ",
            {
                "CAMPUS_BUSINESS_API_MODE": "official",
                "CAMPUS_BUSINESS_API_BASE_URL": "https://Api.Example.edu.cn",
                "CAMPUS_BUSINESS_API_TOKEN": "secret",
                "CAMPUS_BUSINESS_API_ALLOWED_HOSTS": (
                    " API.EXAMPLE.EDU.CN., backup.example.edu.cn "
                ),
            },
            clear=True,
        ):
            config = BusinessAPIConfig.from_env()

        self.assertEqual(
            config.allowed_hosts,
            ("api.example.edu.cn", "backup.example.edu.cn"),
        )

if __name__ == "__main__": unittest.main()
