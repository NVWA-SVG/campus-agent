"""Hardened adapter for an authorized official read-only API."""
from __future__ import annotations

import ipaddress
import json
import re
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from .gateway import BusinessAPIError, CampusBusinessGateway
from .models import (
    CampusServiceStatus,
    STATUSES,
    utc_now_iso,
    validate_query,
    validate_timezone_iso8601,
)


class OfficialHttpCampusBusinessGateway(CampusBusinessGateway):
    MAX_RETRY_AFTER_SECONDS = 1.0
    TOTAL_BUDGET_SECONDS = 4.0
    _REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,100}")

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        client: httpx.Client | None = None,
        max_response_bytes: int = 131_072,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        allowed_hosts: set[str] | frozenset[str] | None = None,
    ) -> None:
        base_url = base_url.strip()
        token = token.strip()
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("official base_url 必须是固定 HTTPS 地址")
        if not token.strip() or max_response_bytes <= 0:
            raise ValueError("official 配置无效")
        hostname = parsed.hostname.rstrip(".").lower()
        self._validate_public_hostname(hostname)
        normalized_allowed = {
            host.strip().rstrip(".").lower() for host in (allowed_hosts or {hostname})
        }
        if hostname not in normalized_allowed:
            raise ValueError("official base_url 主机不在允许列表")
        self._url = base_url.rstrip("/") + "/v1/campus/service-status"
        self._token = token
        self._max_bytes = max_response_bytes
        self._sleep = sleep
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(2.5, connect=1.5, write=1.0, pool=1.0),
            follow_redirects=False,
            verify=True,
            trust_env=False,
        )
    def query_service_status(
        self, service_code: str, campus: str
    ) -> CampusServiceStatus:
        service_code, campus = validate_query(service_code, campus)
        deadline = self._clock() + self.TOTAL_BUDGET_SECONDS
        for attempt in range(2):
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise BusinessAPIError(
                    "timeout", "校园业务服务请求超时", retryable=True
                )
            try:
                with self._client.stream(
                    "GET",
                    self._url,
                    params={"service_code": service_code, "campus": campus},
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    timeout=min(2.5, remaining),
                ) as response:
                    self._validate_response_envelope(response)
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        if self._clock() >= deadline:
                            raise BusinessAPIError(
                                "timeout", "校园业务服务请求超时", retryable=True
                            )
                        body.extend(chunk)
                        if len(body) > self._max_bytes:
                            raise BusinessAPIError(
                                "invalid_response", "校园业务服务响应过大"
                            )
                    return self._decode(response, bytes(body), service_code, campus)
            except BusinessAPIError as exc:
                if attempt == 0 and exc.retryable:
                    delay = min(
                        exc.retry_after_seconds or 0.05,
                        max(0.0, deadline - self._clock()),
                    )
                    if delay:
                        self._sleep(delay)
                    continue
                raise
            except httpx.TransportError as exc:
                if attempt == 0:
                    delay = min(0.05, max(0.0, deadline - self._clock()))
                    if delay:
                        self._sleep(delay)
                    continue
                raise BusinessAPIError(
                    "upstream_unavailable",
                    "校园业务服务暂时不可用",
                    retryable=True,
                ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _validate_public_hostname(hostname: str) -> None:
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            raise ValueError("official base_url 不允许本地主机")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not address.is_global:
            raise ValueError("official base_url 不允许私有或保留 IP")

    def _validate_response_envelope(self, response: httpx.Response) -> None:
        encoding = (
            response.headers.get("content-encoding", "identity").lower().strip()
        )
        if encoding not in {"", "identity"}:
            raise BusinessAPIError(
                "invalid_response", "校园业务服务响应使用了不允许的压缩"
            )
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self._max_bytes
            except ValueError as exc:
                raise BusinessAPIError(
                    "invalid_response", "校园业务服务响应长度无效"
                ) from exc
            if too_large:
                raise BusinessAPIError("invalid_response", "校园业务服务响应过大")

    def _decode(
        self,
        response: httpx.Response,
        body: bytes,
        service_code: str,
        campus: str,
    ) -> CampusServiceStatus:
        if response.is_redirect:
            raise BusinessAPIError(
                "invalid_response", "校园业务服务返回了不允许的重定向"
            )
        if response.status_code in {408, 429, 502, 503, 504}:
            retry_after = (
                self._bounded_retry_after(response)
                if response.status_code == 429
                else None
            )
            raise BusinessAPIError(
                "upstream_unavailable",
                "校园业务服务暂时不可用",
                retryable=True,
                retry_after_seconds=retry_after,
            )
        if response.status_code in {401, 403}:
            raise BusinessAPIError("unauthorized", "校园业务服务鉴权失败")
        if response.status_code == 404:
            raise BusinessAPIError("not_found", "未找到该校园服务")
        if response.status_code >= 400:
            raise BusinessAPIError("invalid_request", "校园业务服务拒绝了请求")
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            raise BusinessAPIError(
                "invalid_response", "校园业务服务返回格式不正确"
            )
        try:
            return self._map(json.loads(body), service_code, campus)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BusinessAPIError(
                "invalid_response", "校园业务服务返回数据不完整"
            ) from exc

    def _bounded_retry_after(self, response: httpx.Response) -> float:
        try:
            value = float(response.headers.get("retry-after", "0.05"))
        except ValueError:
            value = 0.05
        return min(max(value, 0.0), self.MAX_RETRY_AFTER_SECONDS)

    @staticmethod
    def _map(
        payload: Any, expected_service: str, expected_campus: str
    ) -> CampusServiceStatus:
        if not isinstance(payload, dict):
            raise TypeError("payload")
        service, campus, status = (
            payload["service_code"],
            payload["campus"],
            payload["status"],
        )
        if (
            service != expected_service
            or campus != expected_campus
            or status not in STATUSES
        ):
            raise ValueError("identity/status mismatch")

        def text_field(name: str, limit: int = 200) -> str:
            value = payload[name]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(name)
            return value.strip()

        def optional_int(name: str) -> int | None:
            value = payload.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name)
            return value

        updated_at = validate_timezone_iso8601(text_field("updated_at", 64))
        request_id = payload.get("request_id")
        if request_id is not None and (
            not isinstance(request_id, str)
            or not OfficialHttpCampusBusinessGateway._REQUEST_ID_PATTERN.fullmatch(
                request_id
            )
        ):
            raise ValueError("request_id")
        return CampusServiceStatus(
            "1.0",
            "official",
            service,
            text_field("service_name"),
            campus,
            status,
            text_field("location"),
            text_field("today_hours"),
            updated_at,
            utc_now_iso(),
            optional_int("queue_count"),
            optional_int("estimated_wait_minutes"),
            request_id=request_id,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
