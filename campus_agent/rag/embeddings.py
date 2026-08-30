"""可替换、默认离线的文本向量接口。

Hashing Provider 是无需第三方模型的安全底座。真实中文语义检索可以显式切换到
Sentence Transformers；运行期默认只读取本地缓存，模型下载必须由独立准备脚本触发。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections import Counter
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from campus_agent.rag.retriever import tokenize


DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_SEMANTIC_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_BGE_QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："
DEFAULT_BATCH_SIZE = 32


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def resolve_model_revision(model_name: str, revision: str | None) -> str:
    """为默认BGE锁定revision，并拒绝不可复现的自定义模型配置。"""

    normalized_model = model_name.strip()
    normalized_revision = revision.strip() if revision else ""
    if normalized_revision:
        return normalized_revision
    if normalized_model == DEFAULT_SEMANTIC_MODEL:
        return DEFAULT_SEMANTIC_REVISION
    raise ValueError(
        "使用自定义 Embedding 模型时必须显式设置 CAMPUS_EMBEDDING_REVISION"
    )


class EmbeddingProvider(Protocol):
    name: str
    dimension: int
    minimum_similarity: float
    device: str
    runtime_versions: dict[str, str]
    runtime_metadata: dict[str, object]

    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class HashingEmbeddingProvider:
    """将中英文词元映射到固定维度向量的纯本地实现。"""

    name = "local-hashing"
    # 特征哈希存在碰撞；较保守的阈值避免无关问题被强制匹配到资料。
    minimum_similarity = 0.22

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 64:
            raise ValueError("哈希向量维度必须至少为64")
        self.dimension = dimension
        self.device = "cpu"
        self.runtime_versions = {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
        }
        self.runtime_metadata = {
            "device": self.device,
            "versions": dict(self.runtime_versions),
        }

    @property
    def fingerprint(self) -> str:
        # 算法版本必须进入指纹；将来更改分词或哈希方式时缓存会自动失效。
        return f"local-hashing:v1:{self.dimension}"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    def _embed(self, text: str) -> tuple[float, ...]:
        frequencies = Counter(tokenize(text))
        vector = [0.0] * self.dimension
        for token, frequency in frequencies.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(frequency))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


class SentenceTransformerEmbeddingProvider:
    """Sentence Transformers 适配器；默认只允许读取本机模型缓存。

    新版 Sentence Transformers 的 ``encode_query`` / ``encode_document`` 会被
    优先使用。较旧版本没有这两个接口时，会安全回退到 ``encode``。
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_folder: Path | str | None = None,
        local_files_only: bool = True,
        minimum_similarity: float = 0.48,
        query_prompt: str | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("CAMPUS_EMBEDDING_MODEL 不能为空")
        if batch_size < 1:
            raise ValueError("Embedding batch size 必须至少为1")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("Embedding 相似度阈值必须位于 -1 到 1 之间")
        normalized_revision = resolve_model_revision(model_name, revision)
        normalized_query_prompt = (
            DEFAULT_BGE_QUERY_PROMPT
            if query_prompt is None and model_name == DEFAULT_SEMANTIC_MODEL
            else (query_prompt or "").strip()
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "启用 Sentence Transformers 前请安装 campus-agent[semantic]"
            ) from error

        normalized_device = device.strip() if device else None
        normalized_cache = (
            str(Path(cache_folder).expanduser().absolute()) if cache_folder else None
        )
        # 任何配置下都不执行远程自定义代码。local_files_only 默认为 True，
        # 只有 prepare_embedding_model 脚本会显式传 False。
        self._model = SentenceTransformer(
            model_name,
            revision=normalized_revision,
            device=normalized_device,
            cache_folder=normalized_cache,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if not callable(get_dimension):
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None or int(dimension) < 1:
            raise RuntimeError("Embedding 模型没有返回有效维度")

        self.model_name = model_name
        self.revision = normalized_revision
        self.batch_size = batch_size
        self.cache_folder = normalized_cache
        self.local_files_only = local_files_only
        self.minimum_similarity = float(minimum_similarity)
        self.query_prompt = normalized_query_prompt
        self.dimension = int(dimension)
        model_device = getattr(self._model, "device", None)
        self.device = str(model_device or normalized_device or "auto")
        self.runtime_versions = {
            "python": platform.python_version(),
            "sentence-transformers": _package_version("sentence-transformers"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
        }
        self.runtime_metadata = {
            "device": self.device,
            "versions": dict(self.runtime_versions),
        }

    @property
    def fingerprint(self) -> str:
        """返回只包含向量语义相关配置的稳定指纹。"""

        return json.dumps(
            {
                "provider": self.name,
                "adapter_version": 3,
                "model": self.model_name,
                "revision": self.revision,
                "dimension": self.dimension,
                "normalize": True,
                "query_prompt": self.query_prompt,
                "device": self.device,
                "runtime_versions": self.runtime_versions,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts, kind="document")

    def embed_query(self, text: str) -> tuple[float, ...]:
        rows = self._encode((text,), kind="query")
        if len(rows) != 1:
            raise RuntimeError("Embedding 模型没有返回单条查询向量")
        return rows[0]

    def _encode(
        self,
        texts: Sequence[str],
        *,
        kind: str,
    ) -> tuple[tuple[float, ...], ...]:
        values = [str(text) for text in texts]
        if not values:
            return ()
        kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        specialized = getattr(self._model, f"encode_{kind}", None)
        if callable(specialized):
            if kind == "query" and self.query_prompt:
                kwargs["prompt"] = self.query_prompt
            vectors = specialized(values, **kwargs)
        else:
            # Sentence Transformers < 5 没有非对称编码快捷接口。为保持相同
            # 语义，查询提示词在旧接口回退路径中直接拼接到查询文本。
            if kind == "query" and self.query_prompt:
                values = [f"{self.query_prompt}{text}" for text in values]
            vectors = self._model.encode(values, **kwargs)

        rows = tuple(
            tuple(float(component) for component in vector) for vector in vectors
        )
        if len(rows) != len(values):
            raise RuntimeError("Embedding 模型返回的向量数量不正确")
        if any(len(row) != self.dimension for row in rows):
            raise RuntimeError("Embedding 模型返回的向量维度不正确")
        return rows


def _environment_boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} 只能是 true 或 false")
    return normalized == "true"


def _optional_environment(name: str) -> str | None:
    value = os.getenv(name)
    normalized = value.strip() if value else ""
    return normalized or None


def build_embedding_provider_from_environment() -> EmbeddingProvider:
    """从环境变量创建 Provider，调用本身不会允许模型联网下载。"""

    provider = os.getenv("CAMPUS_EMBEDDING_PROVIDER", "hashing").strip().lower()
    if provider in {"hash", "hashing", "local"}:
        return HashingEmbeddingProvider()
    if provider in {"sentence-transformers", "sentence_transformers", "semantic"}:
        try:
            batch_size = int(
                os.getenv("CAMPUS_EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
            )
            threshold = float(os.getenv("CAMPUS_EMBEDDING_MINIMUM_SIMILARITY", "0.48"))
        except ValueError as error:
            raise ValueError("Embedding batch size 或相似度阈值格式错误") from error
        cache_folder = _optional_environment("CAMPUS_EMBEDDING_MODEL_CACHE_DIR")
        runtime_local_only = _environment_boolean(
            "CAMPUS_EMBEDDING_LOCAL_ONLY",
            True,
        )
        if not runtime_local_only:
            raise ValueError(
                "运行期禁止联网下载 Embedding；请先执行 "
                "python -m scripts.prepare_embedding_model --allow-download"
            )
        model_name = os.getenv(
            "CAMPUS_EMBEDDING_MODEL",
            DEFAULT_SEMANTIC_MODEL,
        ).strip()
        return SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            revision=_optional_environment("CAMPUS_EMBEDDING_REVISION"),
            device=_optional_environment("CAMPUS_EMBEDDING_DEVICE"),
            batch_size=batch_size,
            cache_folder=Path(cache_folder) if cache_folder else None,
            # Web/CLI 运行期始终离线；联网下载只能由显式准备脚本触发。
            local_files_only=True,
            minimum_similarity=threshold,
            query_prompt=(
                os.environ["CAMPUS_EMBEDDING_QUERY_PROMPT"]
                if "CAMPUS_EMBEDDING_QUERY_PROMPT" in os.environ
                else None
            ),
        )
    raise ValueError("CAMPUS_EMBEDDING_PROVIDER 只支持 hashing 或 sentence-transformers")
