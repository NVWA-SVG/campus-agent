"""面向移动端或内网客户端的最小 Campus Agent HTTP 服务。"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Final

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from campus_agent.web import ChatRequest, WebAgentService, _closeable_async_iterator


API_PREFIX: Final = "/api/v1"
DEFAULT_ALLOWED_HOSTS: Final = ("127.0.0.1", "localhost", "testserver")
MAX_JSON_BODY_BYTES: Final = 16 * 1024
_HOST_LABEL_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


class RequestBodyLimitMiddleware:
    """同时限制带 Content-Length 和 chunked 传输的请求体。"""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") not in {"POST", "PUT", "PATCH"}
        ):
            await self.app(scope, receive, send)
            return

        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        declared_length = headers.get(b"content-length")
        if declared_length is not None:
            try:
                content_length = int(declared_length)
            except ValueError:
                await self._error_response(400, "Content-Length无效", scope, receive, send)
                return
            if content_length < 0:
                await self._error_response(400, "Content-Length无效", scope, receive, send)
                return
            if content_length > self.max_body_bytes:
                await self._error_response(
                    413,
                    "请求体不能超过16 KiB",
                    scope,
                    receive,
                    send,
                )
                return

        received_bytes = 0
        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="请求体不能超过16 KiB",
                    )
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _error_response(
        status_code: int,
        detail: str,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)


def mobile_api_token_from_environment() -> str:
    """读取共享访问令牌；过短令牌会在启动阶段直接拒绝。"""

    token = os.environ.get("CAMPUS_MOBILE_API_TOKEN", "")
    return _validated_api_token(token)


def server_allowed_hosts_from_environment() -> list[str]:
    """返回明确的 Host 白名单；始终保留本机探活地址。"""

    raw_hosts = os.environ.get("CAMPUS_SERVER_ALLOWED_HOSTS", "").strip()
    if not raw_hosts:
        return list(DEFAULT_ALLOWED_HOSTS)

    allowed_hosts: list[str] = []
    for raw_host in raw_hosts.split(","):
        host = raw_host.strip().lower()
        if not host:
            continue
        if host == "*":
            raise ValueError("CAMPUS_SERVER_ALLOWED_HOSTS 不允许使用全局通配符 *")

        hostname = host[2:] if host.startswith("*.") else host
        labels = hostname.split(".")
        if (
            len(hostname) > 253
            or not labels
            or any(not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels)
        ):
            raise ValueError(
                "CAMPUS_SERVER_ALLOWED_HOSTS 只能包含逗号分隔的 IPv4 或域名，"
                "不能包含协议、端口或路径"
            )
        if host not in allowed_hosts:
            allowed_hosts.append(host)

    if not allowed_hosts:
        raise ValueError("CAMPUS_SERVER_ALLOWED_HOSTS 至少需要一个有效 Host")
    for local_host in DEFAULT_ALLOWED_HOSTS:
        if local_host not in allowed_hosts:
            allowed_hosts.append(local_host)
    return allowed_hosts


def _validated_api_token(token: str) -> str:
    if len(token) < 32 or len(token) > 256 or token.strip() != token:
        raise ValueError(
            "CAMPUS_MOBILE_API_TOKEN 必须是32至256位且首尾不含空白的随机字符串"
        )
    return token


def _max_concurrent_chats_from_environment() -> int:
    raw_value = os.environ.get("CAMPUS_SERVER_MAX_CONCURRENT_CHATS", "4").strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "CAMPUS_SERVER_MAX_CONCURRENT_CHATS 必须是1至32之间的整数"
        ) from error
    if not 1 <= value <= 32:
        raise ValueError("CAMPUS_SERVER_MAX_CONCURRENT_CHATS 必须是1至32之间的整数")
    return value


def _release_on_close(
    iterator: Iterator[str],
    limiter: threading.BoundedSemaphore,
) -> Iterator[str]:
    try:
        yield from iterator
    finally:
        close_iterator = getattr(iterator, "close", None)
        if callable(close_iterator):
            close_iterator()
        limiter.release()


def create_mobile_app(
    service: WebAgentService | None = None,
    *,
    api_token: str | None = None,
    allowed_hosts: list[str] | None = None,
    max_concurrent_chats: int | None = None,
) -> FastAPI:
    """创建独立于本地浏览器 CSRF 接口的移动端 ASGI 应用。"""

    owns_service = service is None
    configured_token = _validated_api_token(api_token) if api_token is not None else None
    configured_hosts = (
        server_allowed_hosts_from_environment()
        if allowed_hosts is None
        else allowed_hosts
    )
    if not configured_hosts:
        raise ValueError("allowed_hosts 至少需要一个有效 Host")
    if "*" in configured_hosts:
        raise ValueError("allowed_hosts 不允许使用全局通配符 *")
    configured_limit = (
        _max_concurrent_chats_from_environment()
        if max_concurrent_chats is None
        else max_concurrent_chats
    )
    if not 1 <= configured_limit <= 32:
        raise ValueError("max_concurrent_chats 必须是1至32之间的整数")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.api_token = (
            configured_token or mobile_api_token_from_environment()
        )
        application.state.chat_limiter = threading.BoundedSemaphore(configured_limit)
        if owns_service:
            application.state.agent_service = WebAgentService.from_environment()
        try:
            yield
        finally:
            if owns_service:
                application.state.agent_service.close()

    application = FastAPI(
        title="Campus Agent Mobile API",
        version="1.0.0",
        description="面向移动端内测客户端的最小 API",
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    application.state.agent_service = service
    application.state.api_token = configured_token
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(configured_hosts),
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=MAX_JSON_BODY_BYTES,
    )

    @application.middleware("http")
    async def mobile_boundary(request: Request, call_next):
        response = None
        path = request.url.path
        if path.startswith(API_PREFIX) and path != f"{API_PREFIX}/health":
            authorization = request.headers.get("authorization", "")
            scheme, separator, credential = authorization.partition(" ")
            expected_token = request.app.state.api_token
            if (
                separator != " "
                or scheme.lower() != "bearer"
                or not credential
                or not secrets.compare_digest(credential, expected_token)
            ):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "访问令牌无效"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.get(f"{API_PREFIX}/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "campus-agent",
            "api_version": "v1",
        }

    @application.post(f"{API_PREFIX}/chat/stream")
    def chat_stream(chat_request: ChatRequest) -> StreamingResponse:
        limiter = application.state.chat_limiter
        if not limiter.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="同时处理的聊天请求过多，请稍后重试",
                headers={"Retry-After": "1"},
            )
        try:
            iterator = application.state.agent_service.stream_chat(chat_request)
        except Exception:
            limiter.release()
            raise
        return StreamingResponse(
            _closeable_async_iterator(_release_on_close(iterator, limiter)),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-store",
            },
        )

    @application.get(f"{API_PREFIX}/history")
    def history(
        session_id: str = Query(min_length=1, max_length=64),
    ) -> dict[str, object]:
        return {
            "session_id": session_id,
            "messages": application.state.agent_service.history(session_id),
        }

    @application.delete(f"{API_PREFIX}/history")
    def clear_history(
        session_id: str = Query(min_length=1, max_length=64),
    ) -> dict[str, object]:
        application.state.agent_service.clear(session_id)
        return {"cleared": True, "session_id": session_id}

    return application


app = create_mobile_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Campus Agent 移动端 API")
    parser.add_argument(
        "--host",
        default=os.environ.get("CAMPUS_SERVER_HOST", "127.0.0.1"),
        help="监听地址，默认仅本机",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CAMPUS_SERVER_PORT", "8000")),
        help="监听端口",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="代码变化后自动重载，仅用于开发",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        mobile_api_token_from_environment()
    except ValueError as error:
        parser.error(str(error))
    target = "campus_agent.mobile_api:app" if args.reload else app
    uvicorn.run(target, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
