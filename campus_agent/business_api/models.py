"""Canonical, allow-listed business API models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

SERVICE_CODES = frozenset({"campus_card", "registrar", "library", "student_affairs"})
CAMPUS_CODES = frozenset({"main", "east", "west"})
STATUSES = frozenset({"open", "closed", "busy", "maintenance", "unknown"})


@dataclass(frozen=True, slots=True)
class CampusServiceStatus:
    schema_version: str
    provider: str
    service_code: str
    service_name: str
    campus: str
    status: str
    location: str
    today_hours: str
    updated_at: str
    fetched_at: str
    queue_count: int | None = None
    estimated_wait_minutes: int | None = None
    stale: bool = False
    cache_hit: bool = False
    request_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_cache(self, *, hit: bool, stale: bool = False) -> "CampusServiceStatus":
        return replace(self, cache_hit=hit, stale=stale)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_timezone_iso8601(value: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp timezone")
    return value


def validate_query(service_code: object, campus: object) -> tuple[str, str]:
    if not isinstance(service_code, str) or service_code not in SERVICE_CODES:
        supported_services = ", ".join(sorted(SERVICE_CODES))
        raise ValueError(f"service_code 必须是以下之一：{supported_services}")
    if not isinstance(campus, str) or campus not in CAMPUS_CODES:
        supported_campuses = ", ".join(sorted(CAMPUS_CODES))
        raise ValueError(f"campus 必须是以下之一：{supported_campuses}")
    return service_code, campus
