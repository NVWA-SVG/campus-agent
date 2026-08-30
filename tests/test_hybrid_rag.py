from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from campus_agent.rag.embeddings import HashingEmbeddingProvider
from campus_agent.rag.models import DocumentChunk
from campus_agent.rag.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from campus_agent.rag.service import LocalRAG


class FakeEmbeddingProvider:
    name = "fake-semantic"
    dimension = 2
    minimum_similarity = 0.1

    def __init__(self, document_vectors, query_vector) -> None:
        self.document_vectors = tuple(document_vectors)
        self.query_vector = tuple(query_vector)

    def embed_documents(self, texts):
        return self.document_vectors

    def embed_query(self, text):
        return self.query_vector


class ExplodingEmbeddingProvider:
    name = "broken"
    dimension = 2
    minimum_similarity = 0.1

    def embed_documents(self, texts):
        raise RuntimeError("model unavailable")

    def embed_query(self, text):
        raise AssertionError("query vector should not be requested")


class QueryExplodingEmbeddingProvider:
    name = "query-broken"
    dimension = 2
    minimum_similarity = 0.1

    def embed_documents(self, texts):
        return tuple((1.0, 0.0) for _ in texts)

    def embed_query(self, text):
        raise RuntimeError("query device unavailable")


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            DocumentChunk("a", "ZX-900 手册", "精确编号对应第一份资料", "a.md#手册"),
            DocumentChunk("b", "语义资料", "第二份资料没有查询词", "b.md#资料"),
        )

    def test_hash_embedding_is_deterministic_and_normalized(self) -> None:
        provider = HashingEmbeddingProvider(dimension=128)
        first = provider.embed_query("校园卡 API")
        second = provider.embed_query("校园卡 API")

        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)

    def test_vector_channel_can_find_a_match_without_keyword_overlap(self) -> None:
        provider = FakeEmbeddingProvider(((1.0, 0.0), (0.0, 1.0)), (0.0, 1.0))
        retriever = VectorRetriever(self.chunks, provider)

        hits = retriever.search("completely unrelated words")

        self.assertEqual(hits[0].chunk.chunk_id, "b")
        self.assertEqual(hits[0].retrieval_method, "vector")

    def test_hybrid_keeps_an_exact_identifier_at_the_top(self) -> None:
        provider = FakeEmbeddingProvider(((0.0, 1.0), (1.0, 0.0)), (1.0, 0.0))
        hybrid = HybridRetriever(
            BM25Retriever(self.chunks),
            VectorRetriever(self.chunks, provider),
        )

        hits = hybrid.search("ZX-900")

        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["a", "b"])
        self.assertEqual(hits[0].lexical_rank, 1)
        self.assertEqual(hits[1].vector_rank, 1)

    def test_embedding_failure_degrades_to_bm25(self) -> None:
        rag = LocalRAG(embedding_provider=ExplodingEmbeddingProvider())

        hits = rag.retrieve("校园卡挂失")

        self.assertTrue(hits)
        self.assertEqual(rag.vector_status, "degraded")
        self.assertEqual(rag.vector_degraded_reason, "RuntimeError")
        self.assertEqual(hits[0].retrieval_method, "bm25")

    def test_query_embedding_failure_falls_back_to_bm25(self) -> None:
        rag = LocalRAG(embedding_provider=QueryExplodingEmbeddingProvider())
        self.assertEqual(rag.vector_status, "ready")

        hits = rag.retrieve("校园卡挂失")

        self.assertTrue(hits)
        self.assertEqual(hits[0].retrieval_method, "bm25")
        self.assertEqual(rag.vector_status, "degraded")
        self.assertEqual(rag.vector_degraded_reason, "RuntimeError")

    def test_semantic_provider_startup_failure_uses_offline_fallback(self) -> None:
        with patch(
            "campus_agent.rag.service.build_embedding_provider_from_environment",
            side_effect=OSError("model is not cached"),
        ):
            rag = LocalRAG()

        self.assertTrue(rag.retrieve("图书馆借书"))
        self.assertEqual(rag.embedding_provider_name, "local-hashing")
        self.assertEqual(rag.vector_status, "degraded")
        self.assertEqual(rag.vector_degraded_reason, "OSError")

    def test_exact_identifier_beats_dual_channel_partial_match(self) -> None:
        chunks = (
            DocumentChunk(
                "a",
                "ZX-900 ZX-900 手册",
                "精确编号对应第一份资料",
                "a.md#手册",
            ),
            DocumentChunk(
                "b",
                "ZX 900 说明",
                "两个编号片段被分开记录",
                "b.md#说明",
            ),
        )
        provider = FakeEmbeddingProvider(((0.0, 1.0), (1.0, 0.0)), (1.0, 0.0))
        lexical = BM25Retriever(chunks)
        self.assertEqual(lexical.search("ZX-900")[0].chunk.chunk_id, "a")

        hits = HybridRetriever(lexical, VectorRetriever(chunks, provider)).search(
            "ZX-900"
        )

        self.assertEqual(hits[0].chunk.chunk_id, "a")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_exact_time_is_pinned_above_partial_time_tokens(self) -> None:
        chunks = (
            DocumentChunk("a", "闭馆时间 22:30", "工作日安排", "a.md#时间"),
            DocumentChunk("b", "闭馆时间 22 30", "旧资料", "b.md#时间"),
        )
        provider = FakeEmbeddingProvider(((0.0, 1.0), (1.0, 0.0)), (1.0, 0.0))

        hits = HybridRetriever(
            BM25Retriever(chunks),
            VectorRetriever(chunks, provider),
        ).search("闭馆时间是 22:30 吗")

        self.assertEqual(hits[0].chunk.chunk_id, "a")

    def test_invalid_document_vector_degrades_without_breaking_rag(self) -> None:
        provider = FakeEmbeddingProvider(
            tuple((0.0, 0.0) for _ in range(10)),
            (1.0, 0.0),
        )
        rag = LocalRAG(embedding_provider=provider)

        self.assertEqual(rag.vector_status, "degraded")
        self.assertTrue(rag.retrieve("图书馆借书"))


if __name__ == "__main__":
    unittest.main()
