"""RAG服务：检索、构造上下文并生成带引用的保守回答。"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from campus_agent.rag.chunking import load_markdown_chunks
from campus_agent.rag.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider_from_environment,
)
from campus_agent.domain import Citation
from campus_agent.rag.models import (
    DocumentChunk,
    RAGAnswer,
    RetrievalFilter,
    RetrievalHit,
)
from campus_agent.rag.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from campus_agent.rag.vector_cache import (
    VectorCache,
    VectorCacheStats,
    build_vector_cache_from_environment,
)


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
ExtraChunkLoader = Callable[[], Iterable[DocumentChunk]]

PRECISE_FACT_RULES = (
    (
        "费用或金额",
        re.compile(r"多少钱|多少元|收费|费用|金额|罚款|价格"),
        re.compile(r"(?:\d+(?:\.\d+)?\s*(?:元|万元))|免费|不收费"),
    ),
    (
        "开放或服务时间",
        re.compile(r"几点|开门|关门|开放时间|营业时间|服务时间"),
        re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d"),
    ),
    (
        "联系电话",
        re.compile(r"联系电话|电话号码|电话是多少|联系方式"),
        re.compile(r"(?:0\d{2,3}[- ]?)?\d{7,11}"),
    ),
    (
        "办理或审核时长",
        re.compile(
            r"多久|几天|多少天|几日|多少日|几个工作日|几小时|多少小时|"
            r"几分钟|多少分钟|多长时间"
        ),
        re.compile(r"\d+(?:\.\d+)?\s*(?:天|日|工作日|小时|分钟)"),
    ),
    (
        "数量或次数",
        re.compile(r"多少本|几本|几次|多少次|多少份|几份|最多.*(?:本|次|份)"),
        re.compile(r"\d+\s*(?:本|册|次|份)"),
    ),
    (
        "赔偿倍数",
        re.compile(r"几倍|多少倍|赔偿倍数"),
        re.compile(r"\d+(?:\.\d+)?\s*倍"),
    ),
    (
        "分数或门槛",
        re.compile(r"绩点|多少分|最低.*分|信用分"),
        re.compile(r"\d+(?:\.\d+)?\s*(?:分|绩点)?"),
    ),
    (
        "截止日期",
        re.compile(r"截止日期|截止到哪|哪一天截止|什么时候截止"),
        re.compile(r"(?:\d{4}[-年/.])?\d{1,2}[-月/.]\d{1,2}日?"),
    ),
    (
        "具体楼栋或房间号",
        re.compile(r"房间号|楼栋|具体地址|哪些房间|具体有哪些房间"),
        re.compile(r"(?:[A-Za-z]?\d{2,4}\s*室)|(?:\d+\s*号?楼)"),
    ),
)


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    version: int
    built_at: str
    chunks: tuple[DocumentChunk, ...]
    lexical_retriever: BM25Retriever | None
    vector_retriever: VectorRetriever | None
    hybrid_retriever: HybridRetriever | None
    vector_status: str
    vector_degraded_reason: str | None
    vector_cache_stats: VectorCacheStats | None


class LocalRAG:
    """使用不可变快照的线程安全本地检索服务。"""

    def __init__(
        self,
        knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
        *,
        extra_chunk_loader: ExtraChunkLoader | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_cache: VectorCache | None = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self._extra_chunk_loader = extra_chunk_loader
        self._state_lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._provider_degraded_reason: str | None = None
        self._runtime_vector_degraded_reason: str | None = None
        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
        else:
            try:
                self._embedding_provider = build_embedding_provider_from_environment()
            except Exception as error:
                # 语义模型依赖缺失或本地模型未缓存时，Web/CLI 仍应能够启动。
                # Hashing Provider 完全离线，并让 BM25 继续充当可靠的词法底座。
                self._embedding_provider = HashingEmbeddingProvider()
                self._provider_degraded_reason = type(error).__name__
        if vector_cache is not None:
            self._vector_cache = vector_cache
        else:
            try:
                self._vector_cache = build_vector_cache_from_environment()
            except Exception:
                # 缓存配置或目录不可用只影响启动性能，不影响向量与词法检索。
                self._vector_cache = None
        self._snapshot = self._build_snapshot(version=1)

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return self._current_snapshot().chunks

    @property
    def version(self) -> int:
        return self._current_snapshot().version

    @property
    def built_at(self) -> str:
        return self._current_snapshot().built_at

    @property
    def embedding_provider_name(self) -> str:
        return self._embedding_provider.name

    @property
    def embedding_model_name(self) -> str:
        return str(
            getattr(
                self._embedding_provider,
                "model_name",
                self._embedding_provider.name,
            )
        )

    @property
    def embedding_revision(self) -> str | None:
        revision = getattr(self._embedding_provider, "revision", None)
        return str(revision) if revision else None

    @property
    def embedding_fingerprint(self) -> str:
        raw = str(self._embedding_provider.fingerprint)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def embedding_device(self) -> str:
        return str(getattr(self._embedding_provider, "device", "unknown"))

    @property
    def embedding_runtime_versions(self) -> dict[str, str]:
        raw = getattr(self._embedding_provider, "runtime_versions", {})
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in sorted(raw.items())}

    @property
    def embedding_minimum_similarity(self) -> float:
        return float(self._embedding_provider.minimum_similarity)

    @property
    def embedding_query_prompt(self) -> str:
        return str(getattr(self._embedding_provider, "query_prompt", ""))

    @property
    def vector_status(self) -> str:
        with self._state_lock:
            if (
                self._provider_degraded_reason is not None
                or self._runtime_vector_degraded_reason is not None
            ):
                return "degraded"
            return self._snapshot.vector_status

    @property
    def vector_degraded_reason(self) -> str | None:
        with self._state_lock:
            return (
                self._runtime_vector_degraded_reason
                or self._provider_degraded_reason
                or self._snapshot.vector_degraded_reason
            )

    @property
    def vector_cache_stats(self) -> dict[str, object] | None:
        stats = self._current_snapshot().vector_cache_stats
        return stats.as_dict() if stats is not None else None

    def reload(self) -> IndexSnapshot:
        """先完整构建候选快照，成功后再原子替换旧索引。"""

        with self._rebuild_lock:
            next_version = self._current_snapshot().version + 1
            candidate = self._build_snapshot(version=next_version)
            with self._state_lock:
                self._snapshot = candidate
                self._runtime_vector_degraded_reason = None
            return candidate

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        strategy: str = "hybrid",
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须至少为 1")
        snapshot = self._current_snapshot()
        retrievers = {
            "bm25": snapshot.lexical_retriever,
            "vector": snapshot.vector_retriever,
            "hybrid": snapshot.hybrid_retriever or snapshot.lexical_retriever,
        }
        if strategy not in retrievers:
            raise ValueError("strategy 只支持 bm25、vector 或 hybrid")
        retriever = retrievers[strategy]
        if retriever is None:
            return ()
        try:
            hits = retriever.search(query, top_k=top_k, filters=filters)
        except Exception as error:
            if strategy not in {"vector", "hybrid"}:
                raise
            # 文档向量可能构建成功，但查询向量仍可能因模型、设备或维度异常失败。
            # 查询期同样必须降级，不能让整个 RAG 工具不可用。
            with self._state_lock:
                self._runtime_vector_degraded_reason = type(error).__name__
            lexical = snapshot.lexical_retriever
            if lexical is None:
                return ()
            return lexical.search(query, top_k=top_k, filters=filters)
        if strategy in {"vector", "hybrid"}:
            with self._state_lock:
                self._runtime_vector_degraded_reason = None
        return hits

    def build_context(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: RetrievalFilter | None = None,
    ) -> str:
        hits = self.retrieve(query, top_k=top_k, filters=filters)
        if not hits:
            return "没有检索到可用资料。"
        sections = []
        for hit in hits:
            sections.append(
                f"[资料{hit.rank}] {hit.chunk.title}\n"
                f"{hit.chunk.content}\n"
                f"来源：{hit.chunk.source}"
            )
        return "\n\n".join(sections)

    def answer(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: RetrievalFilter | None = None,
        strategy: str = "hybrid",
    ) -> RAGAnswer:
        """离线保守回答；真实LLM通过 `build_context` 获取增强上下文。"""
        hits = self.retrieve(
            query,
            top_k=top_k,
            filters=filters,
            strategy=strategy,
        )
        if not hits:
            return RAGAnswer(
                query=query,
                answer=(
                    "知识库中暂未找到相关信息，请联系对应业务部门确认。"
                    "我目前可以查询课程和已经收录的校园办事资料。"
                ),
                hits=(),
                citations=(),
            )

        citations = tuple(
            Citation(
                citation_id=f"C{hit.rank}",
                document_id=hit.chunk.metadata.document_id,
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.metadata.source_name or hit.chunk.source.split("#", 1)[0],
                title=hit.chunk.title,
                score=round(hit.score, 6),
                retrieval_method=hit.retrieval_method,
                # Citation 只携带定位信息；正文已经存在于最终回答，不再复制进 checkpoint。
                snippet="",
            )
            for hit in hits[:1]
        )
        missing_fact = self.missing_precise_fact(query, hits)
        if missing_fact is not None:
            return RAGAnswer(
                query=query,
                answer=(
                    f"检索到了相关主题资料，但资料中没有提供你询问的具体{missing_fact}。"
                    "为避免编造，请联系对应业务部门或查看当期官方通知确认。"
                ),
                hits=hits,
                citations=citations,
            )

        best = hits[0].chunk
        answer = f"{best.title}：{best.content}\n来源：{best.source}"
        return RAGAnswer(
            query=query,
            answer=answer,
            hits=hits,
            citations=citations,
        )

    @staticmethod
    def missing_precise_fact(
        query: str,
        hits: Iterable[RetrievalHit],
    ) -> str | None:
        """判断问题要求的精确事实是否确实存在于本轮证据中。"""

        evidence = "\n".join(
            # 离线回答只使用最佳片段。不能因为 Top-k 中另一个主题恰好含有
            # 数字，就把它误当成当前主题的金额、时间或数量证据。
            f"{hit.chunk.title}\n{hit.chunk.content}"
            for hit in tuple(hits)[:1]
        )
        for label, query_pattern, evidence_pattern in PRECISE_FACT_RULES:
            if query_pattern.search(query) and not evidence_pattern.search(evidence):
                return label
        return None

    def _build_snapshot(self, *, version: int) -> IndexSnapshot:
        chunks = list(load_markdown_chunks(self.knowledge_dir))
        if self._extra_chunk_loader is not None:
            chunks.extend(self._extra_chunk_loader())
        frozen_chunks = tuple(chunks)
        # BM25 是跨 Embedding 实验的固定词法对照，不能随向量 Provider 改变。
        lexical = BM25Retriever(frozen_chunks) if frozen_chunks else None
        vector = None
        vector_status = "empty"
        vector_degraded_reason = None
        if frozen_chunks:
            try:
                vector = VectorRetriever(
                    frozen_chunks,
                    self._embedding_provider,
                    vector_cache=self._vector_cache,
                )
                vector_status = "ready"
            except Exception as error:
                # BM25 始终可独立工作；模型缺失或向量异常不能让整个知识库下线。
                vector_status = "degraded"
                vector_degraded_reason = type(error).__name__
        return IndexSnapshot(
            version=version,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            chunks=frozen_chunks,
            lexical_retriever=lexical,
            vector_retriever=vector,
            hybrid_retriever=(
                HybridRetriever(lexical, vector)
                if lexical is not None and vector is not None
                else None
            ),
            vector_status=vector_status,
            vector_degraded_reason=vector_degraded_reason,
            vector_cache_stats=(vector.cache_stats if vector is not None else None),
        )

    def _current_snapshot(self) -> IndexSnapshot:
        with self._state_lock:
            return self._snapshot
