"""DeepSeek Chat Completions客户端、配置和调用指标。"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekError(RuntimeError):
    """DeepSeek请求、响应或配置错误。"""


@dataclass(frozen=True, slots=True)
class DeepSeekConfig:
    api_key: str | None
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout_seconds: float = 30.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        normalized_base_url = self.base_url.rstrip("/")
        object.__setattr__(self, "base_url", normalized_base_url)
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip() or None)
        if not self.model.strip():
            raise ValueError("model不能为空")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须大于0")
        if self.max_retries < 0:
            raise ValueError("max_retries不能小于0")

        parsed = urllib.parse.urlparse(normalized_base_url)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("DEEPSEEK_BASE_URL端口无效") from error
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.deepseek.com"
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/v1"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "DEEPSEEK_BASE_URL只允许官方HTTPS地址https://api.deepseek.com"
            )

    @classmethod
    def from_environment(cls) -> "DeepSeekConfig":
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            base_url=os.getenv(
                "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
            ).rstrip("/"),
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "1")),
        )


@dataclass(slots=True)
class DeepSeekMetrics:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "network_attempts": self.calls + self.retries,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
        }


OpenUrl = Callable[..., Any]


class DeepSeekChatClient:
    """使用标准库调用DeepSeek，并要求返回JSON对象。"""

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        opener: OpenUrl = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._opener = opener
        self._sleeper = sleeper
        self._clock = clock
        self._metrics = DeepSeekMetrics()
        self._metrics_lock = threading.Lock()

    def complete_json(
        self,
        messages: Sequence[dict[str, str]],
    ) -> dict[str, Any]:
        """调用Chat Completions并解析assistant返回的JSON对象。"""

        if not self.config.api_key:
            with self._metrics_lock:
                self._metrics.failures += 1
            raise DeepSeekError("未设置DEEPSEEK_API_KEY")
        with self._metrics_lock:
            self._metrics.calls += 1

        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": 1200,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = self._clock()
        try:
            body = self._request_with_retry(request)
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(self._strip_code_fence(content))
            if not isinstance(parsed, dict):
                raise DeepSeekError("模型输出必须是JSON对象")
            self._record_usage(body.get("usage", {}))
            with self._metrics_lock:
                self._metrics.successes += 1
            return parsed
        except DeepSeekError:
            with self._metrics_lock:
                self._metrics.failures += 1
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            with self._metrics_lock:
                self._metrics.failures += 1
            raise DeepSeekError("DeepSeek返回了无法解析的JSON响应") from error
        finally:
            elapsed = max(0.0, self._clock() - started)
            with self._metrics_lock:
                self._metrics.total_latency_ms += round(elapsed * 1000)

    def metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            return self._metrics.as_dict()

    def _request_with_retry(
        self,
        request: urllib.request.Request,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with self._opener(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    if not isinstance(body, dict):
                        raise DeepSeekError("DeepSeek响应主体不是JSON对象")
                    return body
            except urllib.error.HTTPError as error:
                error.read()
                last_error = DeepSeekError(f"DeepSeek HTTP {error.code}")
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt >= self.config.max_retries:
                    break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = DeepSeekError(f"DeepSeek网络请求失败：{error}")
                if attempt >= self.config.max_retries:
                    break
            except json.JSONDecodeError as error:
                raise DeepSeekError("DeepSeek HTTP响应不是合法JSON") from error

            with self._metrics_lock:
                self._metrics.retries += 1
            self._sleeper(min(0.5 * (2**attempt), 2.0))

        if isinstance(last_error, DeepSeekError):
            raise last_error
        raise DeepSeekError("DeepSeek请求失败")

    def _record_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        with self._metrics_lock:
            self._metrics.prompt_tokens += self._safe_int(
                usage.get("prompt_tokens")
            )
            self._metrics.completion_tokens += self._safe_int(
                usage.get("completion_tokens")
            )
            self._metrics.total_tokens += self._safe_int(usage.get("total_tokens"))

    @staticmethod
    def _safe_int(value: object) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _strip_code_fence(content: object) -> str:
        if not isinstance(content, str):
            raise DeepSeekError("DeepSeek消息content不是字符串")
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped
