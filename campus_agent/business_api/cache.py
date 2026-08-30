"""Bounded, thread-safe TTL cache with stale-if-error support."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from .gateway import BusinessAPIError, CampusBusinessGateway
from .models import CampusServiceStatus, validate_query


@dataclass(slots=True)
class _Entry:
    value: CampusServiceStatus
    created: float


class CachingCampusBusinessGateway(CampusBusinessGateway):
    def __init__(
        self,
        upstream: CampusBusinessGateway,
        *,
        ttl_seconds: float = 30,
        stale_seconds: float = 300,
        max_entries: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not math.isfinite(ttl_seconds)
            or not math.isfinite(stale_seconds)
            or ttl_seconds <= 0
            or stale_seconds < ttl_seconds
            or ttl_seconds > 3_600
            or stale_seconds > 86_400
            or max_entries <= 0
        ):
            raise ValueError("缓存配置无效")
        self._upstream = upstream
        self._ttl = ttl_seconds
        self._stale = stale_seconds
        self._max = max_entries
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._condition = threading.Condition()
        self._loading: set[tuple[str, str]] = set()

    def query_service_status(
        self, service_code: str, campus: str
    ) -> CampusServiceStatus:
        key = validate_query(service_code, campus)
        with self._condition:
            while True:
                entry = self._entries.get(key)
                now = self._clock()
                if entry and now - entry.created <= self._ttl:
                    self._entries.move_to_end(key)
                    return entry.value.with_cache(hit=True)
                if key not in self._loading:
                    self._loading.add(key)
                    stale_entry = entry
                    break
                self._condition.wait()

        try:
            value = self._upstream.query_service_status(*key)
        except BusinessAPIError as exc:
            if (
                exc.retryable
                and stale_entry
                and self._clock() - stale_entry.created <= self._stale
            ):
                return stale_entry.value.with_cache(hit=True, stale=True)
            raise
        finally:
            with self._condition:
                self._loading.discard(key)
                self._condition.notify_all()

        with self._condition:
            self._entries[key] = _Entry(value, self._clock())
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return value

    def close(self) -> None:
        self._upstream.close()
