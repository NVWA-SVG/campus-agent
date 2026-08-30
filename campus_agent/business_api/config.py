"""Server-side provider configuration; secrets are never serialized."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class BusinessAPIConfig:
    mode: str = "mock"
    base_url: str | None = None
    token: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    cache_ttl_seconds: float = 30
    cache_stale_seconds: float = 300

    @classmethod
    def from_env(cls) -> "BusinessAPIConfig":
        mode = os.getenv("CAMPUS_BUSINESS_API_MODE", "mock").strip().lower()
        if mode not in {"mock", "official"}:
            raise ValueError("CAMPUS_BUSINESS_API_MODE 必须是 mock 或 official")
        try:
            ttl = float(os.getenv("CAMPUS_BUSINESS_API_CACHE_TTL", "30"))
            stale = float(os.getenv("CAMPUS_BUSINESS_API_CACHE_STALE", "300"))
        except ValueError as exc:
            raise ValueError("业务 API 缓存时间必须是数字") from exc
        if (
            not math.isfinite(ttl)
            or not math.isfinite(stale)
            or ttl <= 0
            or stale < ttl
            or ttl > 3_600
            or stale > 86_400
        ):
            raise ValueError("业务 API 缓存时间范围无效")
        base_url = (os.getenv("CAMPUS_BUSINESS_API_BASE_URL") or "").strip() or None
        token = (os.getenv("CAMPUS_BUSINESS_API_TOKEN") or "").strip() or None
        raw_allowed_hosts = os.getenv("CAMPUS_BUSINESS_API_ALLOWED_HOSTS", "")
        allowed_hosts = tuple(
            dict.fromkeys(
                host.strip().rstrip(".").lower()
                for host in raw_allowed_hosts.split(",")
                if host.strip()
            )
        )
        if mode == "official" and base_url and not allowed_hosts:
            parsed_host = urlsplit(base_url).hostname
            if parsed_host:
                allowed_hosts = (parsed_host.rstrip(".").lower(),)
        config = cls(
            mode=mode,
            base_url=base_url,
            token=token,
            allowed_hosts=allowed_hosts,
            cache_ttl_seconds=ttl,
            cache_stale_seconds=stale,
        )
        if mode == "official" and (not config.base_url or not config.token):
            raise ValueError("official 模式缺少业务 API 地址或令牌")
        return config

    def public_status(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "configured": self.mode == "mock" or bool(self.base_url and self.token),
            "network_enabled": self.mode == "official",
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }
