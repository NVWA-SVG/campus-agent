from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from campus_agent.rag.embeddings import (
    DEFAULT_BGE_QUERY_PROMPT,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_embedding_provider_from_environment,
)
from campus_agent.rag.models import DocumentChunk
from campus_agent.rag.service import LocalRAG
from campus_agent.rag.vector_cache import VectorCache
from scripts.prepare_embedding_model import main as prepare_embedding_model


class CountingEmbeddingProvider:
    name = "counting"
    dimension = 2
    minimum_similarity = -1.0

    def __init__(self, fingerprint: str = "counting:v1") -> None:
        self._fingerprint = fingerprint
        self.document_calls: list[tuple[str, ...]] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed_documents(self, texts):
        values = tuple(texts)
        self.document_calls.append(values)
        return tuple(self._vector(text) for text in values)

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> tuple[float, float]:
        checksum = sum(text.encode("utf-8"))
        return (float(checksum % 17 + 1), float(checksum % 13 + 1))


class ModernSentenceTransformer:
    instances: list["ModernSentenceTransformer"] = []

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.document_calls = []
        self.query_calls = []
        type(self).instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 2

    def encode_document(self, texts, **kwargs):
        self.document_calls.append((tuple(texts), kwargs))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, texts, **kwargs):
        self.query_calls.append((tuple(texts), kwargs))
        return np.asarray([[0.0, 1.0] for _ in texts], dtype=np.float32)


class LegacySentenceTransformer:
    instances: list["LegacySentenceTransformer"] = []

    def __init__(self, model_name: str, **kwargs) -> None:
        self.calls = []
        type(self).instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        self.calls.append((tuple(texts), kwargs))
        return np.asarray([[1.0, 1.0] for _ in texts], dtype=np.float32)


class SentenceTransformerProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        ModernSentenceTransformer.instances.clear()
        LegacySentenceTransformer.instances.clear()

    @staticmethod
    def _sentence_transformers_module(model_class):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = model_class
        return module

    def test_runtime_configuration_is_offline_and_never_trusts_remote_code(self) -> None:
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": self._sentence_transformers_module(
                    ModernSentenceTransformer
                )
            },
        ):
            provider = SentenceTransformerEmbeddingProvider(
                revision="release-1",
                device="cpu",
                batch_size=7,
                cache_folder=Path("model-cache"),
            )

        model = ModernSentenceTransformer.instances[-1]
        self.assertEqual(model.model_name, DEFAULT_SEMANTIC_MODEL)
        self.assertTrue(model.kwargs["local_files_only"])
        self.assertFalse(model.kwargs["trust_remote_code"])
        self.assertEqual(model.kwargs["revision"], "release-1")
        self.assertEqual(model.kwargs["device"], "cpu")
        self.assertTrue(Path(model.kwargs["cache_folder"]).is_absolute())
        self.assertIn("release-1", provider.fingerprint)

    def test_query_and_document_use_distinct_modern_encoders(self) -> None:
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": self._sentence_transformers_module(
                    ModernSentenceTransformer
                )
            },
        ):
            provider = SentenceTransformerEmbeddingProvider(
                batch_size=5,
                query_prompt="检索：",
            )
        self.assertEqual(provider.embed_documents(("资料一", "资料二"))[0], (1.0, 0.0))
        self.assertEqual(provider.embed_query("问题"), (0.0, 1.0))

        model = ModernSentenceTransformer.instances[-1]
        self.assertEqual(model.document_calls[0][0], ("资料一", "资料二"))
        self.assertEqual(model.document_calls[0][1]["batch_size"], 5)
        self.assertNotIn("prompt", model.document_calls[0][1])
        self.assertEqual(model.query_calls[0][0], ("问题",))
        self.assertEqual(model.query_calls[0][1]["prompt"], "检索：")
        self.assertTrue(model.query_calls[0][1]["normalize_embeddings"])

    def test_old_sentence_transformers_falls_back_to_encode(self) -> None:
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": self._sentence_transformers_module(
                    LegacySentenceTransformer
                )
            },
        ):
            provider = SentenceTransformerEmbeddingProvider(query_prompt="检索：")
        provider.embed_documents(("资料",))
        provider.embed_query("问题")

        calls = LegacySentenceTransformer.instances[-1].calls
        self.assertEqual(calls[0][0], ("资料",))
        self.assertEqual(calls[1][0], ("检索：问题",))

    def test_environment_defaults_do_not_enable_downloads(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"CAMPUS_EMBEDDING_PROVIDER": "semantic"},
                clear=True,
            ),
            patch.dict(
                sys.modules,
                {
                    "sentence_transformers": self._sentence_transformers_module(
                        ModernSentenceTransformer
                    )
                },
            ),
        ):
            provider = build_embedding_provider_from_environment()

        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)
        self.assertEqual(provider.model_name, DEFAULT_SEMANTIC_MODEL)
        self.assertEqual(provider.revision, DEFAULT_SEMANTIC_REVISION)
        self.assertEqual(provider.query_prompt, DEFAULT_BGE_QUERY_PROMPT)
        self.assertTrue(provider.local_files_only)

    def test_custom_model_requires_an_explicit_revision(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CAMPUS_EMBEDDING_PROVIDER": "semantic",
                "CAMPUS_EMBEDDING_MODEL": "example/custom-model",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "自定义.*REVISION"):
                build_embedding_provider_from_environment()

    def test_fingerprint_contains_device_and_critical_runtime_versions(self) -> None:
        with (
            patch.dict(
                sys.modules,
                {
                    "sentence_transformers": self._sentence_transformers_module(
                        ModernSentenceTransformer
                    )
                },
            ),
            patch(
                "campus_agent.rag.embeddings._package_version",
                side_effect=lambda name: f"locked-{name}",
            ),
        ):
            provider = SentenceTransformerEmbeddingProvider(device="cpu")

        fingerprint = provider.fingerprint
        self.assertIn('"device":"cpu"', fingerprint)
        self.assertIn(
            '"sentence-transformers":"locked-sentence-transformers"',
            fingerprint,
        )
        self.assertIn('"torch":"locked-torch"', fingerprint)

    def test_runtime_rejects_environment_request_to_download(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CAMPUS_EMBEDDING_PROVIDER": "semantic",
                "CAMPUS_EMBEDDING_LOCAL_ONLY": "false",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "运行期禁止联网下载"):
                build_embedding_provider_from_environment()

    def test_prepare_script_refuses_implicit_download(self) -> None:
        with (
            patch(
                "scripts.prepare_embedding_model.SentenceTransformerEmbeddingProvider"
            ) as provider,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            prepare_embedding_model([])
        provider.assert_not_called()

    def test_prepare_script_downloads_then_reloads_strictly_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(
                    sys.modules,
                    {
                        "sentence_transformers": self._sentence_transformers_module(
                            ModernSentenceTransformer
                        )
                    },
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = prepare_embedding_model(
                    [
                        "--allow-download",
                        "--model-cache-dir",
                        temporary,
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(len(ModernSentenceTransformer.instances), 2)
        self.assertFalse(ModernSentenceTransformer.instances[0].kwargs["local_files_only"])
        self.assertTrue(ModernSentenceTransformer.instances[1].kwargs["local_files_only"])
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["offline_reload_verified"])
        self.assertEqual(payload["revision"], DEFAULT_SEMANTIC_REVISION)
        self.assertEqual(payload["query_prompt"], DEFAULT_BGE_QUERY_PROMPT)


class VectorCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cache_dir = Path(self.temporary_directory.name)

    def test_cache_reuses_unchanged_vectors_and_only_encodes_new_text(self) -> None:
        provider = CountingEmbeddingProvider()
        first = VectorCache(self.cache_dir)
        matrix = first.load_or_encode(provider, ("a", "b"))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(provider.document_calls, [("a", "b")])
        self.assertEqual(first.last_stats.misses, 2)

        restarted = VectorCache(self.cache_dir)
        restarted.load_or_encode(provider, ("a", "b", "c"))
        self.assertEqual(provider.document_calls[-1], ("c",))
        self.assertEqual(restarted.last_stats.hits, 2)
        self.assertEqual(restarted.last_stats.misses, 1)
        self.assertFalse(any(self.cache_dir.glob("*.tmp.npz")))

    def test_corrupt_cache_is_ignored_and_atomically_rebuilt(self) -> None:
        provider = CountingEmbeddingProvider()
        cache = VectorCache(self.cache_dir)
        expected = cache.load_or_encode(provider, ("a", "b"))
        cache_file = next(self.cache_dir.glob("*.npz"))
        cache_file.write_bytes(b"not a valid npz")

        rebuilt = VectorCache(self.cache_dir)
        actual = rebuilt.load_or_encode(provider, ("a", "b"))
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(rebuilt.last_stats.rebuilt)
        self.assertEqual(rebuilt.last_stats.misses, 2)
        with np.load(cache_file, allow_pickle=False) as payload:
            self.assertEqual(payload["vectors"].dtype, np.float32)

    def test_model_fingerprint_isolates_cache_files(self) -> None:
        first_provider = CountingEmbeddingProvider("model:revision-a")
        second_provider = CountingEmbeddingProvider("model:revision-b")
        cache = VectorCache(self.cache_dir)
        cache.load_or_encode(first_provider, ("same text",))
        cache.load_or_encode(second_provider, ("same text",))

        self.assertEqual(len(tuple(self.cache_dir.glob("*.npz"))), 2)
        self.assertEqual(second_provider.document_calls, [("same text",)])

    def test_local_rag_reload_uses_incremental_cache(self) -> None:
        provider = CountingEmbeddingProvider()
        uploaded: list[DocumentChunk] = []
        rag = LocalRAG(
            extra_chunk_loader=lambda: tuple(uploaded),
            embedding_provider=provider,
            vector_cache=VectorCache(self.cache_dir),
        )
        initial_count = len(rag.chunks)
        self.assertEqual(len(provider.document_calls), 1)
        self.assertEqual(len(provider.document_calls[0]), initial_count)

        rag.reload()
        self.assertEqual(len(provider.document_calls), 1)
        self.assertEqual(rag.vector_cache_stats["hits"], initial_count)

        uploaded.append(
            DocumentChunk(
                chunk_id="new-1",
                title="新增资料",
                content="增量缓存测试材料",
                source="new.md#新增资料",
            )
        )
        rag.reload()
        self.assertEqual(len(provider.document_calls), 2)
        self.assertEqual(len(provider.document_calls[-1]), 1)
        self.assertEqual(rag.vector_cache_stats["misses"], 1)

    def test_cached_vectors_remain_normalized_for_matrix_search(self) -> None:
        provider = CountingEmbeddingProvider()
        chunks = (
            DocumentChunk("a", "标题A", "内容A", "a.md#A"),
            DocumentChunk("b", "标题B", "内容B", "b.md#B"),
        )
        rag = LocalRAG(
            extra_chunk_loader=lambda: chunks,
            embedding_provider=provider,
            vector_cache=VectorCache(self.cache_dir),
        )
        hits = rag.retrieve("内容A", strategy="vector")
        self.assertTrue(hits)
        self.assertTrue(all(math.isfinite(hit.score) for hit in hits))


if __name__ == "__main__":
    unittest.main()
