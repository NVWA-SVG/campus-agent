"""显式下载并校验本地 Embedding 模型。

正常 Web/CLI 运行不会联网下载模型。只有用户主动执行本模块并传入
``--allow-download`` 时，Sentence Transformers 才会访问模型仓库。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from campus_agent.rag.embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEMANTIC_MODEL,
    SentenceTransformerEmbeddingProvider,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验 Campus Agent 语义模型")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="确认本次命令可以联网下载模型（必填）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CAMPUS_EMBEDDING_MODEL", DEFAULT_SEMANTIC_MODEL),
    )
    parser.add_argument(
        "--revision",
        default=os.getenv("CAMPUS_EMBEDDING_REVISION") or None,
    )
    parser.add_argument(
        "--device",
        default=os.getenv("CAMPUS_EMBEDDING_DEVICE") or None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CAMPUS_EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=(
            Path(os.environ["CAMPUS_EMBEDDING_MODEL_CACHE_DIR"])
            if os.getenv("CAMPUS_EMBEDDING_MODEL_CACHE_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--query-prompt",
        default=(
            os.environ["CAMPUS_EMBEDDING_QUERY_PROMPT"]
            if "CAMPUS_EMBEDDING_QUERY_PROMPT" in os.environ
            else None
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.allow_download:
        parser.error("该命令可能联网；请确认后添加 --allow-download")

    provider_kwargs = {
        "model_name": arguments.model,
        "revision": arguments.revision,
        "device": arguments.device,
        "batch_size": arguments.batch_size,
        "cache_folder": arguments.model_cache_dir,
        "query_prompt": arguments.query_prompt,
    }
    try:
        download_provider = SentenceTransformerEmbeddingProvider(
            **provider_kwargs,
            local_files_only=False,
        )
    except ValueError as error:
        parser.error(str(error))

    def self_test(provider: SentenceTransformerEmbeddingProvider) -> None:
        document_vectors = provider.embed_documents(("校园卡挂失与补办流程",))
        query_vector = provider.embed_query("饭卡丢了应该怎么办？")
        if len(document_vectors) != 1 or len(query_vector) != provider.dimension:
            raise RuntimeError("Embedding 模型向量自检失败")

    # 第一次允许下载并触发两条编码路径，确保所有延迟文件进入缓存。
    self_test(download_provider)
    # 再创建一个严格 local-only 的新实例；只有该实例也能完成编码，才证明
    # 后续 Web/CLI 无网络运行所需文件已经完整落盘。
    offline_provider = SentenceTransformerEmbeddingProvider(
        **provider_kwargs,
        local_files_only=True,
    )
    self_test(offline_provider)
    if download_provider.fingerprint != offline_provider.fingerprint:
        raise RuntimeError("下载实例与离线实例的 Embedding 指纹不一致")

    fingerprint = hashlib.sha256(
        offline_provider.fingerprint.encode("utf-8")
    ).hexdigest()[:16]

    print(
        json.dumps(
            {
                "status": "ready",
                "model": offline_provider.model_name,
                "revision": offline_provider.revision,
                "dimension": offline_provider.dimension,
                "device": offline_provider.device,
                "runtime_versions": offline_provider.runtime_versions,
                "query_prompt": offline_provider.query_prompt,
                "fingerprint": fingerprint,
                "cache_folder": offline_provider.cache_folder,
                "offline_reload_verified": True,
                "trust_remote_code": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
