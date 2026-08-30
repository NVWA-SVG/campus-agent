"""按模型指纹与文本内容增量复用文档向量。

缓存使用单个不含 pickle 的 NPZ 文件。每次更新先写入同目录临时文件并 fsync，
最后用 ``os.replace`` 原子提交；缓存缺失、版本不匹配或损坏时都会安全重建。
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from campus_agent.rag.embeddings import EmbeddingProvider


CACHE_SCHEMA_VERSION = 1
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class VectorCacheStats:
    total: int = 0
    hits: int = 0
    misses: int = 0
    rebuilt: bool = False
    write_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_vector_cache_dir() -> Path:
    configured = os.getenv("CAMPUS_VECTOR_CACHE_DIR")
    path = (
        Path(configured)
        if configured
        else Path.cwd() / ".campus_agent_data" / "vector_cache"
    )
    return path.expanduser().absolute()


def build_vector_cache_from_environment() -> "VectorCache | None":
    raw = os.getenv("CAMPUS_VECTOR_CACHE_ENABLED", "true").strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError("CAMPUS_VECTOR_CACHE_ENABLED 只能是 true 或 false")
    return VectorCache(default_vector_cache_dir()) if raw == "true" else None


class VectorCache:
    """线程安全的内容寻址向量缓存。"""

    def __init__(self, cache_dir: Path) -> None:
        expanded = cache_dir.expanduser().absolute()
        if expanded.exists() and expanded.is_symlink():
            raise ValueError("向量缓存目录不能是符号链接")
        self.cache_dir = expanded
        self._lock = threading.RLock()
        self._last_stats = VectorCacheStats()

    @property
    def last_stats(self) -> VectorCacheStats:
        with self._lock:
            return self._last_stats

    def load_or_encode(
        self,
        provider: "EmbeddingProvider",
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """返回与 ``texts`` 顺序一致的 float32 矩阵，只编码缓存缺失项。"""

        fingerprint = provider.fingerprint
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Embedding Provider 必须提供非空 fingerprint")
        dimension = int(provider.dimension)
        if dimension < 1:
            raise ValueError("Embedding Provider 维度无效")
        normalized_texts = tuple(str(text) for text in texts)
        if not normalized_texts:
            with self._lock:
                self._last_stats = VectorCacheStats()
            return np.empty((0, dimension), dtype=np.float32)

        content_hashes = tuple(self._content_hash(text) for text in normalized_texts)
        with self._lock:
            cache_path = self._cache_path(fingerprint)
            cached, rebuilt = self._read_cache(
                cache_path,
                fingerprint=fingerprint,
                dimension=dimension,
            )
            unique_texts: dict[str, str] = {}
            for digest, text in zip(content_hashes, normalized_texts, strict=True):
                unique_texts.setdefault(digest, text)

            missing_hashes = [digest for digest in unique_texts if digest not in cached]
            if missing_hashes:
                missing_vectors = self._validated_matrix(
                    provider.embed_documents(
                        tuple(unique_texts[digest] for digest in missing_hashes)
                    ),
                    rows=len(missing_hashes),
                    dimension=dimension,
                )
                for index, digest in enumerate(missing_hashes):
                    cached[digest] = missing_vectors[index]

            matrix = np.stack([cached[digest] for digest in content_hashes]).astype(
                np.float32,
                copy=False,
            )

            # 只持久化当前索引仍使用的唯一内容，删除文档后不会留下永久孤儿。
            active_hashes = tuple(unique_texts)
            active_vectors = np.stack([cached[digest] for digest in active_hashes])
            write_error = None
            cache_contents_changed = set(cached) != set(active_hashes)
            if (
                missing_hashes
                or rebuilt
                or cache_contents_changed
                or not cache_path.exists()
            ):
                try:
                    self._write_cache(
                        cache_path,
                        fingerprint=fingerprint,
                        hashes=active_hashes,
                        vectors=active_vectors,
                    )
                except Exception as error:
                    # 缓存只是性能优化；只读目录或磁盘故障不能让检索下线。
                    write_error = type(error).__name__

            missing_set = set(missing_hashes)
            hit_count = len(content_hashes) - sum(
                1 for digest in content_hashes if digest in missing_set
            )
            self._last_stats = VectorCacheStats(
                total=len(content_hashes),
                hits=hit_count,
                misses=len(content_hashes) - hit_count,
                rebuilt=rebuilt,
                write_error=write_error,
            )
            return matrix

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, fingerprint: str) -> Path:
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def _read_cache(
        self,
        path: Path,
        *,
        fingerprint: str,
        dimension: int,
    ) -> tuple[dict[str, NDArray[np.float32]], bool]:
        if not path.exists():
            return {}, False
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("缓存路径类型无效")
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {
                    "schema_version",
                    "fingerprint",
                    "hashes",
                    "vectors",
                }:
                    raise ValueError("缓存字段不完整")
                schema_version = int(payload["schema_version"].item())
                stored_fingerprint = str(payload["fingerprint"].item())
                hashes = payload["hashes"]
                vectors = payload["vectors"]
                if schema_version != CACHE_SCHEMA_VERSION:
                    raise ValueError("缓存Schema版本不匹配")
                if stored_fingerprint != fingerprint:
                    raise ValueError("缓存模型指纹不匹配")
                if hashes.ndim != 1 or vectors.ndim != 2:
                    raise ValueError("缓存矩阵形状无效")
                if len(hashes) != vectors.shape[0] or vectors.shape[1] != dimension:
                    raise ValueError("缓存向量维度无效")
                normalized_hashes = tuple(str(value) for value in hashes.tolist())
                if len(set(normalized_hashes)) != len(normalized_hashes) or any(
                    HASH_PATTERN.fullmatch(value) is None for value in normalized_hashes
                ):
                    raise ValueError("缓存内容哈希无效")
                matrix = self._validated_matrix(
                    vectors,
                    rows=len(normalized_hashes),
                    dimension=dimension,
                )
                return {
                    digest: matrix[index]
                    for index, digest in enumerate(normalized_hashes)
                }, False
        except Exception:
            # 不信任持久化缓存；任何异常都按未命中处理，随后原子重建。
            return {}, True

    def _write_cache(
        self,
        path: Path,
        *,
        fingerprint: str,
        hashes: Sequence[str],
        vectors: NDArray[np.float32],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir.is_symlink():
            raise OSError("向量缓存目录不能是符号链接")
        temporary = self.cache_dir / f".{path.stem}-{uuid.uuid4().hex}.tmp.npz"
        try:
            with temporary.open("xb") as stream:
                np.savez(
                    stream,
                    schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
                    fingerprint=np.asarray(fingerprint),
                    hashes=np.asarray(tuple(hashes), dtype="<U64"),
                    vectors=np.asarray(vectors, dtype=np.float32),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _validated_matrix(
        vectors: object,
        *,
        rows: int,
        dimension: int,
    ) -> NDArray[np.float32]:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("向量无法转换为 float32 矩阵") from error
        if matrix.ndim != 2 or matrix.shape != (rows, dimension):
            raise ValueError("向量数量或维度与文档不一致")
        if not np.isfinite(matrix).all():
            raise ValueError("向量包含非有限数值")
        if rows and np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("文档向量不能是零向量")
        return np.ascontiguousarray(matrix, dtype=np.float32)
