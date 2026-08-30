from __future__ import annotations

from dataclasses import dataclass, field

from campus_agent.domain import Citation


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    document_id: str = ""
    source_name: str = ""
    domain: str = "campus_service"
    category: str = "general"
    visibility: str = "public"
    version: str = "1"
    origin: str = "built_in"
    chunk_index: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "source_name": self.source_name,
            "domain": self.domain,
            "category": self.category,
            "visibility": self.visibility,
            "version": self.version,
            "origin": self.origin,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    document_id: str | None = None
    domain: str | None = None
    category: str | None = None
    version: str | None = None
    origin: str | None = None
    visibility: str = "public"

    @classmethod
    def from_mapping(cls, value: object) -> "RetrievalFilter":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("filters 必须是对象")
        allowed = {"document_id", "domain", "category", "version", "origin"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"filters 包含未知字段：{sorted(unknown)}")
        normalized: dict[str, str | None] = {}
        for key in allowed:
            raw = value.get(key)
            if raw is not None and (not isinstance(raw, str) or not raw.strip()):
                raise ValueError(f"filters.{key} 必须是非空字符串")
            normalized[key] = raw.strip() if isinstance(raw, str) else None
        return cls(**normalized)

    def matches(self, metadata: ChunkMetadata) -> bool:
        return all(
            expected is None or getattr(metadata, field_name) == expected
            for field_name, expected in (
                ("document_id", self.document_id),
                ("domain", self.domain),
                ("category", self.category),
                ("version", self.version),
                ("origin", self.origin),
            )
        ) and metadata.visibility == self.visibility


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    title: str
    content: str
    source: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    rank: int
    lexical_score: float = 0.0
    vector_score: float = 0.0
    retrieval_method: str = "bm25"
    lexical_rank: int | None = None
    vector_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    query: str
    answer: str
    hits: tuple[RetrievalHit, ...]
    citations: tuple[Citation, ...] = ()
