from __future__ import annotations

from abc import ABC, abstractmethod

from .models import CampusServiceStatus


class BusinessAPIError(RuntimeError):
    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class CampusBusinessGateway(ABC):
    @abstractmethod
    def query_service_status(
        self, service_code: str, campus: str
    ) -> CampusServiceStatus:
        ...

    def close(self) -> None:
        pass
