from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from campus_agent.rag import LocalRAG
from campus_agent.rag.knowledge_base import InvalidDocumentError, KnowledgeBaseService
from campus_agent.rag.models import ChunkMetadata, DocumentChunk, RetrievalFilter
from campus_agent.rag.retriever import BM25Retriever
from campus_agent.tools import CampusKnowledgeTool


class MetadataAndCitationTests(unittest.TestCase):
    def test_builtin_chunks_have_stable_metadata(self) -> None:
        rag = LocalRAG()
        campus_card = next(
            chunk for chunk in rag.chunks if chunk.metadata.category == "campus_card"
        )

        self.assertEqual(campus_card.metadata.document_id, "builtin-campus_card")
        self.assertEqual(campus_card.metadata.source_name, "campus_card.md")
        self.assertEqual(campus_card.metadata.domain, "campus_service")
        self.assertEqual(campus_card.metadata.origin, "built_in")

    def test_filter_is_applied_before_ranking(self) -> None:
        rag = LocalRAG()

        academic = rag.retrieve(
            "申请流程材料",
            top_k=10,
            filters=RetrievalFilter(domain="academic"),
        )
        facilities = rag.retrieve(
            "申请流程材料",
            top_k=10,
            filters=RetrievalFilter(domain="facility"),
        )

        self.assertTrue(academic)
        self.assertTrue(
            all(hit.chunk.metadata.domain == "academic" for hit in academic)
        )
        self.assertTrue(
            all(hit.chunk.metadata.domain == "facility" for hit in facilities)
        )

    def test_knowledge_tool_returns_structured_safe_citations(self) -> None:
        output = CampusKnowledgeTool().invoke(query="校园卡丢了怎么补办？")

        self.assertTrue(output.citations)
        citation = output.citations[0]
        self.assertEqual(citation.source, "campus_card.md")
        self.assertEqual(citation.document_id, "builtin-campus_card")
        self.assertNotIn(":\\", citation.source)
        self.assertTrue(output.data["hits"])

    def test_no_hit_has_no_citations(self) -> None:
        answer = LocalRAG().answer("火星基地通行证")

        self.assertEqual(answer.citations, ())

    def test_legacy_uploaded_metadata_gets_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = KnowledgeBaseService(storage_dir=Path(directory))
            result = service.upload(
                filename="legacy.md",
                data="# Legacy\n\n编号 LEGACY-42".encode(),
                content_type="text/markdown",
            )
            document_id = str(result["document"]["document_id"])
            metadata_path = Path(directory) / document_id / "metadata.json"
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("domain", "category", "visibility", "version"):
                raw.pop(key)
            metadata_path.write_text(
                json.dumps(raw, ensure_ascii=False),
                encoding="utf-8",
            )

            restarted = KnowledgeBaseService(storage_dir=Path(directory))
            document = restarted.store.list_documents()[0]
            hit = restarted.rag.retrieve(
                "LEGACY-42",
                filters=RetrievalFilter(origin="uploaded"),
            )[0]

            self.assertEqual(document.domain, "uploaded")
            self.assertEqual(hit.chunk.metadata.document_id, document_id)
            self.assertEqual(hit.chunk.metadata.visibility, "public")

    def test_uploaded_metadata_round_trips_and_version_filter_works(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = KnowledgeBaseService(storage_dir=Path(directory))
            result = service.upload(
                filename="policy.md",
                data="# Policy\n\nPOLICY-2026 applies.".encode(),
                content_type="text/markdown",
                domain="student_affairs",
                category="policy",
                version="2026.1",
            )

            document = result["document"]
            self.assertEqual(document["domain"], "student_affairs")
            self.assertEqual(document["category"], "policy")
            self.assertEqual(document["version"], "2026.1")
            self.assertTrue(
                service.rag.retrieve(
                    "POLICY-2026",
                    filters=RetrievalFilter(
                        domain="student_affairs",
                        category="policy",
                        version="2026.1",
                    ),
                )
            )
            self.assertFalse(
                service.rag.retrieve(
                    "POLICY-2026",
                    filters=RetrievalFilter(version="2025"),
                )
            )

    def test_invalid_uploaded_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = KnowledgeBaseService(storage_dir=Path(directory))

            with self.assertRaisesRegex(InvalidDocumentError, "domain"):
                service.upload(
                    filename="unsafe.md",
                    data=b"# Unsafe\n\nText",
                    content_type="text/markdown",
                    domain="../private",
                )

    def test_excluded_documents_do_not_change_bm25_statistics(self) -> None:
        public = DocumentChunk(
            "public",
            "ZX-77 指南",
            "ZX-77 public answer",
            "public.md",
            ChunkMetadata(visibility="public"),
        )
        private = DocumentChunk(
            "private",
            "ZX-77 私有副本",
            "ZX-77 " * 50,
            "private.md",
            ChunkMetadata(visibility="private"),
        )

        baseline = BM25Retriever((public,)).search("ZX-77")[0].score
        filtered = BM25Retriever((public, private)).search("ZX-77")[0].score

        self.assertAlmostEqual(filtered, baseline)


if __name__ == "__main__":
    unittest.main()
