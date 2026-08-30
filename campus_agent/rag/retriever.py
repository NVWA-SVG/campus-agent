"""支持中英文文本的轻量 BM25 检索器。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from campus_agent.rag.models import DocumentChunk, RetrievalFilter, RetrievalHit

if TYPE_CHECKING:
    from campus_agent.rag.embeddings import EmbeddingProvider
    from campus_agent.rag.vector_cache import VectorCache, VectorCacheStats


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
IDENTIFIER_PATTERN = re.compile(
    r"\b(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*\d)"
    r"[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
GENERIC_BIGRAMS = {
    "如何",
    "怎么",
    "应该",
    "办理",
    "申请",
    "什么",
    "哪里",
    "可以",
}
DEFAULT_BM25_MINIMUM_INFORMATIVE_MATCHES = 2
DEFAULT_HYBRID_LEXICAL_WEIGHT = 0.55
DEFAULT_HYBRID_VECTOR_WEIGHT = 0.45
DEFAULT_HYBRID_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """英文按单词切分，中文同时产生单字与二元字串。"""
    tokens: list[str] = []
    for part in TOKEN_PATTERN.findall(text.lower()):
        if part.isascii():
            tokens.append(part)
            continue
        characters = list(part)
        tokens.extend(characters)
        tokens.extend(
            characters[index] + characters[index + 1]
            for index in range(len(characters) - 1)
        )
    return tokens


def exact_facts(text: str) -> set[str]:
    """提取应该由词法通道锁定的编号与时间，不对普通自然语言做硬排序。"""

    return {
        match.group(0).casefold()
        for pattern in (IDENTIFIER_PATTERN, TIME_PATTERN)
        for match in pattern.finditer(text)
    }


class BM25Retriever:
    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        minimum_informative_matches: int = DEFAULT_BM25_MINIMUM_INFORMATIVE_MATCHES,
    ) -> None:
        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("BM25Retriever 至少需要一个文档片段")
        self._k1 = k1
        self._b = b
        if minimum_informative_matches < 1:
            raise ValueError("BM25 最少信息词命中数必须至少为1")
        self._minimum_informative_matches = minimum_informative_matches
        self._term_frequencies = [
            Counter(tokenize(f"{chunk.title} {chunk.title} {chunk.title} {chunk.content}"))
            for chunk in self._chunks
        ]
        self._lengths = [sum(counter.values()) for counter in self._term_frequencies]

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须至少为 1")

        query_terms = tokenize(query)
        informative_query_terms = {
            term
            for term in query_terms
            if (
                (term.isascii() and len(term) >= 2)
                or (len(term) == 2 and term not in GENERIC_BIGRAMS)
            )
        }
        if not informative_query_terms:
            return ()
        minimum_informative_matches = min(
            self._minimum_informative_matches,
            len(informative_query_terms),
        )
        scored: list[tuple[float, DocumentChunk]] = []
        active_filter = filters or RetrievalFilter()
        eligible_indices = [
            index
            for index, chunk in enumerate(self._chunks)
            if active_filter.matches(chunk.metadata)
        ]
        if not eligible_indices:
            return ()
        document_count = len(eligible_indices)
        average_length = sum(self._lengths[index] for index in eligible_indices) / (
            document_count
        )
        document_frequencies: Counter[str] = Counter()
        for index in eligible_indices:
            document_frequencies.update(self._term_frequencies[index].keys())

        for index in eligible_indices:
            chunk = self._chunks[index]
            frequencies = self._term_frequencies[index]
            length = self._lengths[index]
            informative_matches = informative_query_terms & frequencies.keys()
            # 单个偶然二元词（例如“领取”）不应把完全无关的问题强行匹配到
            # 校园资料；短查询则要求其有限的信息词全部匹配。
            if len(informative_matches) < minimum_informative_matches:
                continue
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = document_frequencies[term]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / average_length
                )
                score += inverse_document_frequency * (
                    frequency * (self._k1 + 1) / denominator
                )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                score=score,
                rank=index + 1,
                lexical_score=score,
                retrieval_method="bm25",
                lexical_rank=index + 1,
            )
            for index, (score, chunk) in enumerate(scored[:top_k])
        )


class VectorRetriever:
    """在不可变 float32 文档矩阵上执行余弦相似度检索。"""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedding_provider: "EmbeddingProvider",
        *,
        vector_cache: "VectorCache | None" = None,
    ) -> None:
        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("VectorRetriever 至少需要一个文档片段")
        self._embedding_provider = embedding_provider
        texts = tuple(
            f"{chunk.title}\n{chunk.title}\n{chunk.content}"
            for chunk in self._chunks
        )
        if vector_cache is not None and isinstance(
            getattr(embedding_provider, "fingerprint", None),
            str,
        ):
            raw_vectors = vector_cache.load_or_encode(embedding_provider, texts)
            self._cache_stats = vector_cache.last_stats
        else:
            raw_vectors = embedding_provider.embed_documents(texts)
            self._cache_stats = None
        self._vectors = self._normalized_matrix(raw_vectors, rows=len(self._chunks))
        self._dimension = int(self._vectors.shape[1])
        if self._dimension != int(embedding_provider.dimension):
            raise ValueError("文档向量维度与 Provider 声明不一致")
        # 快照构建完成后矩阵不可变，避免并发检索期间被意外修改。
        self._vectors.setflags(write=False)

    @property
    def cache_stats(self) -> "VectorCacheStats | None":
        return self._cache_stats

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须至少为 1")
        raw_query_vector = np.asarray(
            self._embedding_provider.embed_query(query),
            dtype=np.float32,
        )
        if raw_query_vector.ndim != 1 or raw_query_vector.shape[0] != self._dimension:
            raise ValueError("查询向量和文档向量维度不一致")
        if not np.isfinite(raw_query_vector).all():
            raise ValueError("查询向量包含非有限数值")
        query_norm = float(np.linalg.norm(raw_query_vector))
        if query_norm == 0:
            return ()
        query_vector = np.ascontiguousarray(
            raw_query_vector / query_norm,
            dtype=np.float32,
        )
        active_filter = filters or RetrievalFilter()
        eligible_indices = np.asarray(
            [
                index
                for index, chunk in enumerate(self._chunks)
                if active_filter.matches(chunk.metadata)
            ],
            dtype=np.intp,
        )
        if eligible_indices.size == 0:
            return ()

        # 文档矩阵已归一化，因此矩阵乘法结果就是余弦相似度。
        similarities = self._vectors[eligible_indices] @ query_vector
        scored: list[tuple[float, DocumentChunk]] = []
        for original_index, raw_score in zip(
            eligible_indices.tolist(),
            similarities.tolist(),
            strict=True,
        ):
            score = float(raw_score)
            if score >= self._embedding_provider.minimum_similarity:
                scored.append((score, self._chunks[original_index]))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                score=score,
                rank=index + 1,
                vector_score=score,
                retrieval_method="vector",
                vector_rank=index + 1,
            )
            for index, (score, chunk) in enumerate(scored[:top_k])
        )

    @staticmethod
    def _normalized_matrix(
        vectors: object,
        *,
        rows: int,
    ) -> NDArray[np.float32]:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("文档向量无法转换为 float32 矩阵") from error
        if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1:
            raise ValueError("向量器返回数量或文档向量维度不一致")
        if not np.isfinite(matrix).all():
            raise ValueError("文档向量包含非有限数值")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("文档向量不能是零向量")
        return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)


class HybridRetriever:
    """使用加权 Reciprocal Rank Fusion 融合 BM25 与向量排序。"""

    def __init__(
        self,
        lexical: BM25Retriever,
        vector: VectorRetriever,
        *,
        lexical_weight: float = DEFAULT_HYBRID_LEXICAL_WEIGHT,
        vector_weight: float = DEFAULT_HYBRID_VECTOR_WEIGHT,
        rrf_k: int = DEFAULT_HYBRID_RRF_K,
    ) -> None:
        if lexical_weight <= 0 or vector_weight <= 0:
            raise ValueError("混合检索权重必须大于0")
        if rrf_k < 1:
            raise ValueError("rrf_k 必须至少为1")
        total = lexical_weight + vector_weight
        self._lexical = lexical
        self._vector = vector
        self._lexical_weight = lexical_weight / total
        self._vector_weight = vector_weight / total
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if top_k < 1:
            raise ValueError("top_k 必须至少为 1")
        candidate_count = max(20, top_k * 4)
        lexical_hits = self._lexical.search(
            query,
            top_k=candidate_count,
            filters=filters,
        )
        vector_hits = self._vector.search(
            query,
            top_k=candidate_count,
            filters=filters,
        )
        candidates: dict[str, dict[str, object]] = {}

        for hit in lexical_hits:
            candidates[hit.chunk.chunk_id] = {
                "chunk": hit.chunk,
                "fused": self._lexical_weight / (self._rrf_k + hit.rank),
                "lexical_score": hit.score,
                "vector_score": 0.0,
                "lexical_rank": hit.rank,
                "vector_rank": None,
            }
        for hit in vector_hits:
            candidate = candidates.setdefault(
                hit.chunk.chunk_id,
                {
                    "chunk": hit.chunk,
                    "fused": 0.0,
                    "lexical_score": 0.0,
                    "vector_score": 0.0,
                    "lexical_rank": None,
                    "vector_rank": None,
                },
            )
            candidate["fused"] = float(candidate["fused"]) + (
                self._vector_weight / (self._rrf_k + hit.rank)
            )
            candidate["vector_score"] = hit.score
            candidate["vector_rank"] = hit.rank

        maximum = (
            self._lexical_weight + self._vector_weight
        ) / (self._rrf_k + 1)
        query_facts = exact_facts(query)
        for item in candidates.values():
            chunk = item["chunk"]
            chunk_facts = exact_facts(f"{chunk.title}\n{chunk.content}")
            item["exact_fact_match"] = bool(query_facts & chunk_facts)
            item["normalized_score"] = float(item["fused"]) / maximum + (
                1.0 if item["exact_fact_match"] else 0.0
            )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item["normalized_score"]),
                str(getattr(item["chunk"], "chunk_id")),
            ),
        )
        return tuple(
            RetrievalHit(
                chunk=item["chunk"],
                score=float(item["normalized_score"]),
                rank=index + 1,
                lexical_score=float(item["lexical_score"]),
                vector_score=float(item["vector_score"]),
                retrieval_method=(
                    "hybrid"
                    if item["lexical_rank"] is not None
                    and item["vector_rank"] is not None
                    else "bm25"
                    if item["lexical_rank"] is not None
                    else "vector"
                ),
                lexical_rank=(
                    int(item["lexical_rank"])
                    if item["lexical_rank"] is not None
                    else None
                ),
                vector_rank=(
                    int(item["vector_rank"])
                    if item["vector_rank"] is not None
                    else None
                ),
            )
            for index, item in enumerate(ordered[:top_k])
        )
