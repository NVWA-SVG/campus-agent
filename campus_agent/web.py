"""Campus Agent 的本地 Web 服务入口。"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from anyio import CancelScope
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from campus_agent.agent import CampusAgent, build_default_agent
from campus_agent.business_api import (
    BusinessAPIConfig,
    CachingCampusBusinessGateway,
    MockCampusBusinessGateway,
    OfficialHttpCampusBusinessGateway,
)
from campus_agent.checkpoint import PrunableSqliteSaver
from campus_agent.deepseek import (
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekChatClient,
    DeepSeekConfig,
)
from campus_agent.deepseek_planner import DeepSeekPlanner
from campus_agent.domain import AgentEvent, AgentResponse
from campus_agent.memory import ConversationMemory
from campus_agent.rag.knowledge_base import (
    MAX_UPLOAD_BYTES,
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
    KnowledgeBaseError,
    KnowledgeBaseService,
    UnsupportedDocumentError,
)
from campus_agent.tools import CampusKnowledgeTool, CampusServiceStatusTool


STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class KnowledgeFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str | None = Field(default=None, min_length=1, max_length=64)
    domain: str | None = Field(default=None, min_length=1, max_length=64)
    category: str | None = Field(default=None, min_length=1, max_length=64)
    version: str | None = Field(default=None, min_length=1, max_length=64)
    origin: Literal["built_in", "uploaded"] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=64)
    planner: Literal["rule", "deepseek"] = "rule"
    filters: KnowledgeFilterRequest | None = None


class WebAgentService:
    """为 HTTP 层管理规划器选择和共享会话记忆。"""

    def __init__(
        self,
        rule_agent: CampusAgent,
        deepseek_agent: CampusAgent,
        deepseek_configuration_error: str | None = None,
        knowledge_base: KnowledgeBaseService | None = None,
        checkpoint_connection: sqlite3.Connection | None = None,
        checkpoint_saver: PrunableSqliteSaver | None = None,
        persistence_error: str | None = None,
        business_tool: CampusServiceStatusTool | None = None,
        business_api_status: dict[str, object] | None = None,
    ) -> None:
        self._agents = {
            "rule": rule_agent,
            "deepseek": deepseek_agent,
        }
        self._deepseek_configuration_error = deepseek_configuration_error
        self._knowledge_base = knowledge_base
        self._checkpoint_connection = checkpoint_connection
        self._checkpoint_saver = checkpoint_saver
        self._persistence_error = persistence_error
        self._business_tool = business_tool
        self._business_api_status = business_api_status or {
            "mode": "disabled",
            "configured": False,
            "network_enabled": False,
            "cache_ttl_seconds": 0,
            "error": None,
        }
        self._closing = False
        self._closed = False
        self._csrf_token = secrets.token_urlsafe(32)
        self._session_locks: weakref.WeakValueDictionary[
            str, threading.Lock
        ] = weakref.WeakValueDictionary()
        self._locks_guard = threading.Lock()
        self._activity_condition = threading.Condition()
        self._active_operations = 0

    @classmethod
    def from_environment(
        cls,
        *,
        knowledge_storage_dir: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> "WebAgentService":
        knowledge_base = KnowledgeBaseService(storage_dir=knowledge_storage_dir)
        knowledge_tool = CampusKnowledgeTool(knowledge_base.rag)
        checkpoint_connection = None
        checkpointer = None
        persistence_error = None
        resolved_checkpoint_path = (
            checkpoint_path.expanduser().absolute()
            if checkpoint_path is not None
            else knowledge_base.store.storage_dir / ".agent-checkpoints.sqlite3"
        )
        try:
            resolved_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if resolved_checkpoint_path.exists() and resolved_checkpoint_path.is_symlink():
                raise sqlite3.DatabaseError("checkpoint 文件不能是符号链接")
            checkpoint_connection = sqlite3.connect(
                resolved_checkpoint_path,
                check_same_thread=False,
            )
            checkpoint_connection.execute("PRAGMA busy_timeout=5000")
            checkpoint_connection.execute("PRAGMA secure_delete=ON")
            checkpointer = PrunableSqliteSaver(
                checkpoint_connection,
                serde=JsonPlusSerializer(
                    pickle_fallback=False,
                    allowed_msgpack_modules=None,
                ),
            )
            checkpointer.setup()
            checkpointer.startup_prune(keep_per_thread=1)
            checkpointer.truncate_wal()
        except (OSError, sqlite3.Error) as error:
            if checkpoint_connection is not None:
                checkpoint_connection.close()
            checkpoint_connection = None
            checkpointer = None
            persistence_error = f"SQLite checkpoint 不可用：{type(error).__name__}"
        memory = ConversationMemory() if checkpointer is None else None
        configuration_error = None
        try:
            deepseek_planner = DeepSeekPlanner.from_environment()
        except (TypeError, ValueError):
            # 在线配置错误不能阻止完全离线的规则模式启动。
            configuration_error = "DeepSeek 环境变量无效，请检查模型、超时、重试和接口地址配置"
            deepseek_planner = DeepSeekPlanner(
                DeepSeekChatClient(DeepSeekConfig(api_key=None))
            )
        business_tool = None
        try:
            business_config = BusinessAPIConfig.from_env()
            if business_config.mode == "official":
                assert business_config.base_url is not None
                assert business_config.token is not None
                business_upstream = OfficialHttpCampusBusinessGateway(
                    base_url=business_config.base_url,
                    token=business_config.token,
                    allowed_hosts=frozenset(business_config.allowed_hosts),
                )
            else:
                business_upstream = MockCampusBusinessGateway()
            business_gateway = CachingCampusBusinessGateway(
                business_upstream,
                ttl_seconds=business_config.cache_ttl_seconds,
                stale_seconds=business_config.cache_stale_seconds,
            )
            business_tool = CampusServiceStatusTool(business_gateway)
            business_status = {
                **business_config.public_status(),
                "error": None,
            }
        except (TypeError, ValueError):
            # 真实模式配置错误时保持禁用，绝不悄悄回退成模拟数据。
            business_status = {
                "mode": "invalid",
                "configured": False,
                "network_enabled": False,
                "cache_ttl_seconds": 0,
                "error": "校园业务 API 配置无效",
            }
        extra_tools = (business_tool,) if business_tool is not None else ()
        return cls(
            rule_agent=build_default_agent(
                memory=memory,
                knowledge_tool=knowledge_tool,
                checkpointer=checkpointer,
                extra_tools=extra_tools,
            ),
            deepseek_agent=build_default_agent(
                planner=deepseek_planner,
                memory=memory,
                knowledge_tool=knowledge_tool,
                checkpointer=checkpointer,
                extra_tools=extra_tools,
            ),
            deepseek_configuration_error=configuration_error,
            knowledge_base=knowledge_base,
            checkpoint_connection=checkpoint_connection,
            checkpoint_saver=checkpointer,
            persistence_error=persistence_error,
            business_tool=business_tool,
            business_api_status=business_status,
        )

    def ask(self, request: ChatRequest) -> dict[str, object]:
        session_id, query, retrieval_filters = self._request_context(request)
        agent = self._agents[request.planner]
        started_at = time.perf_counter()
        try:
            with self._operation(), self._lock_for(session_id):
                response = agent.ask(
                    query,
                    session_id=session_id,
                    retrieval_filters=retrieval_filters,
                )
        except sqlite3.Error as error:
            self._record_checkpoint_error(error)
            raise HTTPException(
                status_code=503,
                detail="会话存储暂不可用，请检查 SQLite 后重试",
            ) from error
        return self._response_payload(response, agent, request.planner, started_at)

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        """返回 POST SSE 迭代器；只发送白名单化 Trace 与最终结果。"""

        session_id, query, retrieval_filters = self._request_context(request)
        agent = self._agents[request.planner]
        started_at = time.perf_counter()

        def generate() -> Iterator[str]:
            graph_stream: Iterator[dict[str, object]] | None = None
            try:
                with self._operation(), self._lock_for(session_id):
                    graph_stream = agent.stream(
                        query,
                        session_id=session_id,
                        retrieval_filters=retrieval_filters,
                    )
                    for item in graph_stream:
                        if item.get("kind") == "trace":
                            event = item.get("event")
                            if isinstance(event, AgentEvent):
                                yield _encode_sse(
                                    "trace",
                                    {"type": event.event_type, "detail": event.detail},
                                )
                        elif item.get("kind") == "result":
                            response = item.get("response")
                            if isinstance(response, AgentResponse):
                                yield _encode_sse(
                                    "result",
                                    self._response_payload(
                                        response,
                                        agent,
                                        request.planner,
                                        started_at,
                                    ),
                                )
                yield _encode_sse("done", {"ok": True})
            except sqlite3.Error as error:
                self._record_checkpoint_error(error)
                yield _encode_sse(
                    "error",
                    {"detail": "会话存储暂不可用，请检查 SQLite 后重试"},
                )
            except Exception:
                # 流已经开始后不能再改变 HTTP 状态码，也不能泄漏内部异常。
                yield _encode_sse(
                    "error",
                    {"detail": "Agent 执行失败，请检查服务日志后重试"},
                )
            finally:
                close_stream = getattr(graph_stream, "close", None)
                if callable(close_stream):
                    close_stream()

        return generate()

    @staticmethod
    def _response_payload(
        response: AgentResponse,
        agent: CampusAgent,
        planner: Literal["rule", "deepseek"],
        started_at: float,
    ) -> dict[str, object]:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "answer": response.answer,
            "events": [
                {"type": event.event_type, "detail": event.detail}
                for event in response.events
            ],
            "citations": [citation.as_dict() for citation in response.citations],
            "metrics": agent.planner_metrics(),
            "planner": planner,
            "elapsed_ms": elapsed_ms,
        }

    @staticmethod
    def _request_context(
        request: ChatRequest,
    ) -> tuple[str, str, dict[str, str] | None]:
        session_id = _validated_session_id(request.session_id)
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="问题不能为空")
        retrieval_filters = (
            request.filters.model_dump(exclude_none=True)
            if request.filters is not None
            else None
        )
        return session_id, query, retrieval_filters

    def history(self, session_id: str) -> list[dict[str, str]]:
        validated = _validated_session_id(session_id)
        try:
            with self._operation(), self._lock_for(validated):
                messages = self._agents["rule"].history(validated)
        except sqlite3.Error as error:
            self._record_checkpoint_error(error)
            raise HTTPException(status_code=503, detail="会话存储暂不可用") from error
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    def clear(self, session_id: str) -> None:
        validated = _validated_session_id(session_id)
        try:
            with self._operation(), self._lock_for(validated):
                # 两个 Agent 共享同一个 Memory 或 SQLite thread，清理一次即可。
                self._agents["rule"].clear_history(validated)
        except sqlite3.Error as error:
            self._record_checkpoint_error(error)
            raise HTTPException(status_code=503, detail="会话存储暂不可用") from error

    def metrics(self, planner: Literal["rule", "deepseek"]) -> dict[str, object]:
        return self._agents[planner].planner_metrics()

    def status(self) -> dict[str, object]:
        metrics = self._agents["deepseek"].planner_metrics()
        return {
            "status": "ok",
            "deepseek_configured": bool(metrics.get("api_key_configured", False)),
            "deepseek_model": metrics.get("model") or DEFAULT_DEEPSEEK_MODEL,
            "deepseek_configuration_error": self._deepseek_configuration_error,
            "csrf_token": self._csrf_token,
            "knowledge": (
                self._knowledge_base.stats() if self._knowledge_base is not None else None
            ),
            "checkpoint": {
                "enabled": self._agents["rule"].checkpoint_enabled,
                "healthy": self._persistence_error is None,
                "backend": (
                    "sqlite" if self._agents["rule"].checkpoint_enabled else "memory"
                ),
                "error": self._persistence_error,
            },
            "campus_business_api": dict(self._business_api_status),
            "network_boundary": (
                "Rule 不调用大模型；Mock 业务数据完全本地；Official 模式会由"
                "后端请求获授权的校园业务 API；选择 DeepSeek 且有有效 Key 时"
                "还会请求模型 API。Embedding 运行期只读取本地缓存"
            ),
        }

    @property
    def csrf_token(self) -> str:
        return self._csrf_token

    def knowledge_documents(self) -> dict[str, object]:
        return self._require_knowledge_base().documents()

    def upload_knowledge(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str,
        domain: str = "uploaded",
        category: str = "general",
        version: str = "1",
    ) -> dict[str, object]:
        return self._require_knowledge_base().upload(
            filename=filename,
            data=data,
            content_type=content_type,
            domain=domain,
            category=category,
            version=version,
        )

    def delete_knowledge(self, document_id: str) -> dict[str, object]:
        return self._require_knowledge_base().delete(document_id)

    def rebuild_knowledge(self) -> dict[str, object]:
        return self._require_knowledge_base().rebuild()

    def close(self) -> None:
        with self._activity_condition:
            if self._closed:
                return
            self._closing = True
            deadline = time.monotonic() + 5.0
            while self._active_operations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 最后一个 operation 会在退出时完成关闭，避免竞态和句柄泄漏。
                    return
                self._activity_condition.wait(timeout=remaining)
            self._finish_close_locked()

    def _require_knowledge_base(self) -> KnowledgeBaseService:
        if self._knowledge_base is None:
            raise HTTPException(status_code=503, detail="动态知识库没有初始化")
        return self._knowledge_base

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    @contextmanager
    def _operation(self):
        with self._activity_condition:
            if self._closing or self._closed:
                raise RuntimeError("WebAgentService 已关闭")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._activity_condition:
                self._active_operations -= 1
                if self._active_operations == 0:
                    if self._closing:
                        self._finish_close_locked()
                    elif self._checkpoint_saver is not None:
                        try:
                            # 此时没有运行中的 Graph；当前 thread 已在 Agent 内压缩，
                            # 这里只用单事务执行全局 session 上限，避免逐 thread 扫描。
                            self._checkpoint_saver.enforce_thread_limit()
                            self._checkpoint_saver.truncate_wal()
                        except (OSError, sqlite3.Error) as error:
                            self._persistence_error = (
                                "SQLite checkpoint 清理失败："
                                f"{type(error).__name__}"
                            )
                self._activity_condition.notify_all()

    def _finish_close_locked(self) -> None:
        if self._closed:
            return
        if self._checkpoint_connection is not None:
            if self._checkpoint_saver is not None:
                try:
                    self._checkpoint_saver.truncate_wal()
                except (OSError, sqlite3.Error):
                    pass
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
        if self._business_tool is not None:
            try:
                self._business_tool.close()
            finally:
                self._business_tool = None
        self._closed = True

    def _record_checkpoint_error(self, error: BaseException) -> None:
        self._persistence_error = (
            f"SQLite checkpoint 运行失败：{type(error).__name__}"
        )


def _validated_session_id(session_id: str) -> str:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id 只能包含字母、数字、下划线和连字符",
        )
    return session_id


def _encode_sse(event: str, payload: dict[str, object]) -> str:
    if event not in {"trace", "result", "done", "error"}:
        raise ValueError("不支持的 SSE 事件")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


async def _read_limited_body(request: Request) -> bytes:
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="单个文件不能超过10 MiB")
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length无效") from None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="单个文件不能超过10 MiB")
        chunks.append(chunk)
    return b"".join(chunks)


def _raise_knowledge_http_error(error: KnowledgeBaseError) -> None:
    if isinstance(error, DocumentTooLargeError):
        status_code = 413
    elif isinstance(error, UnsupportedDocumentError):
        status_code = 415
    elif isinstance(error, DuplicateDocumentError):
        status_code = 409
    elif isinstance(error, DocumentNotFoundError):
        status_code = 404
    elif isinstance(error, InvalidDocumentError):
        status_code = 422
    else:
        status_code = 500
    raise HTTPException(status_code=status_code, detail=str(error)) from error


def create_app(service: WebAgentService | None = None) -> FastAPI:
    owns_service = service is None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if owns_service:
            application.state.agent_service = WebAgentService.from_environment()
        try:
            yield
        finally:
            if owns_service:
                application.state.agent_service.close()

    app = FastAPI(
        title="Campus Agent API",
        version="0.9.0",
        description="校园课程与办事知识问答 Agent 的本地 Web API",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    # 注入型 service 由调用方管理；拥有型 service 到 ASGI startup 才创建，
    # 避免普通 import 或 uvicorn --reload 父进程提前持有 SQLite 句柄。
    app.state.agent_service = service
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.middleware("http")
    async def add_response_headers(request, call_next):
        response = None
        if request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD"}:
            expected_token = app.state.agent_service.csrf_token
            supplied_token = request.headers.get("x-csrf-token", "")
            if not supplied_token or not secrets.compare_digest(
                supplied_token,
                expected_token,
            ):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF校验失败"},
                )

            origin = request.headers.get("origin")
            if response is None and origin:
                expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
                if origin.rstrip("/") != expected_origin.rstrip("/"):
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "请求来源无效"},
                    )

        if response is None:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self'; font-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/docs", include_in_schema=False)
    def local_api_docs() -> FileResponse:
        # FastAPI 默认 Swagger 页面依赖公共 CDN；本项目改用完全本地的说明页。
        return FileResponse(STATIC_DIR / "docs.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return app.state.agent_service.status()

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        return app.state.agent_service.ask(request)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            _closeable_async_iterator(app.state.agent_service.stream_chat(request)),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/history")
    def history(
        session_id: str = Query(min_length=1, max_length=64),
    ) -> dict[str, object]:
        return {
            "session_id": session_id,
            "messages": app.state.agent_service.history(session_id),
        }

    @app.delete("/api/history")
    def clear_history(
        session_id: str = Query(min_length=1, max_length=64),
    ) -> dict[str, object]:
        app.state.agent_service.clear(session_id)
        return {"cleared": True, "session_id": session_id}

    @app.get("/api/metrics")
    def metrics(
        planner: Literal["rule", "deepseek"] = "rule",
    ) -> dict[str, object]:
        return app.state.agent_service.metrics(planner)

    @app.get("/api/knowledge/documents")
    def knowledge_documents() -> dict[str, object]:
        return app.state.agent_service.knowledge_documents()

    @app.post("/api/knowledge/documents", status_code=201)
    async def upload_knowledge_document(
        request: Request,
        filename: str = Query(min_length=1, max_length=120),
        domain: str = Query(default="uploaded", min_length=1, max_length=64),
        category: str = Query(default="general", min_length=1, max_length=64),
        version: str = Query(default="1", min_length=1, max_length=64),
    ) -> dict[str, object]:
        data = await _read_limited_body(request)
        try:
            return await run_in_threadpool(
                app.state.agent_service.upload_knowledge,
                filename=filename,
                data=data,
                content_type=request.headers.get(
                    "content-type", "application/octet-stream"
                ),
                domain=domain,
                category=category,
                version=version,
            )
        except KnowledgeBaseError as error:
            _raise_knowledge_http_error(error)

    @app.delete("/api/knowledge/documents/{document_id}")
    def delete_knowledge_document(document_id: str) -> dict[str, object]:
        try:
            return app.state.agent_service.delete_knowledge(document_id)
        except KnowledgeBaseError as error:
            _raise_knowledge_http_error(error)

    @app.post("/api/knowledge/rebuild")
    def rebuild_knowledge() -> dict[str, object]:
        try:
            return app.state.agent_service.rebuild_knowledge()
        except KnowledgeBaseError as error:
            _raise_knowledge_http_error(error)

    return app


async def _closeable_async_iterator(iterator: Iterator[str]):
    """在线程池迭代同步 Graph 流，并在 ASGI 断线/取消时显式 close。"""

    sentinel = object()

    def next_item():
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    try:
        while True:
            item = await run_in_threadpool(next_item)
            if item is sentinel:
                break
            yield item
    finally:
        close_iterator = getattr(iterator, "close", None)
        if callable(close_iterator):
            # StreamingResponse 在 http.disconnect 时会取消发送任务；shield 确保
            # 底层同步生成器仍能 close，从而回滚 checkpoint 并释放 session 锁。
            with CancelScope(shield=True):
                await run_in_threadpool(close_iterator)


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Campus Agent Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="代码变化后自动重载，仅用于开发",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = "campus_agent.web:app" if args.reload else app
    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
