from __future__ import annotations

import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from campus_agent.rag.knowledge_base import (
    DocumentStore,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
    KnowledgeBaseService,
    UnsupportedDocumentError,
)


def _searchable_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=320, height=240)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 36 180 Td (Night shuttle 22:30 code ZX-900) Tj ET"
    )
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class KnowledgeBaseServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.storage_dir = Path(self.temporary_directory.name)

    def test_uploaded_markdown_is_persistent_and_duplicate_is_rejected(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        initial_version = service.rag.version
        data = (
            "# 创客空间\n\n## 开放时间\n\n"
            "青禾创客空间每周六20:45关闭，门禁口令为MAKER-731。"
        ).encode("utf-8")

        result = service.upload(
            filename="maker.md",
            data=data,
            content_type="text/markdown",
        )
        self.assertGreater(service.rag.version, initial_version)
        self.assertEqual(result["stats"]["uploaded_documents"], 1)
        self.assertIn("20:45", service.rag.answer("青禾创客空间几点关闭").answer)

        with self.assertRaises(DuplicateDocumentError):
            service.upload(
                filename="maker-copy.md",
                data=data,
                content_type="text/markdown",
            )

        restarted = KnowledgeBaseService(storage_dir=self.storage_dir)
        self.assertEqual(restarted.stats()["uploaded_documents"], 1)
        self.assertIn("MAKER-731", restarted.rag.answer("青禾门禁口令").answer)

    def test_upload_reload_failure_rolls_back_document_and_snapshot(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        previous_version = service.rag.version
        previous_chunks = service.rag.chunks

        with patch.object(service.rag, "reload", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                service.upload(
                    filename="rollback.md",
                    data=b"# Rollback\n\ntransaction marker RB-991",
                    content_type="text/markdown",
                )

        self.assertEqual(service.store.list_documents(), ())
        self.assertEqual(service.rag.version, previous_version)
        self.assertEqual(service.rag.chunks, previous_chunks)

    def test_delete_reload_failure_restores_document(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        uploaded = service.upload(
            filename="restore.md",
            data=b"# Restore\n\nrollback delete marker DEL-442",
            content_type="text/markdown",
        )
        document_id = uploaded["document"]["document_id"]
        previous_version = service.rag.version

        with patch.object(service.rag, "reload", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                service.delete(str(document_id))

        self.assertEqual(len(service.store.list_documents()), 1)
        self.assertEqual(service.rag.version, previous_version)
        self.assertIn("DEL-442", service.rag.answer("rollback delete marker").answer)

    def test_startup_recovers_interrupted_staged_delete(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        uploaded = service.upload(
            filename="recover.md",
            data=b"# Recover\n\ncrash recovery marker CRASH-882",
            content_type="text/markdown",
        )
        document_id = str(uploaded["document"]["document_id"])
        staged = service.store.stage_delete(document_id)
        self.assertTrue(staged.exists())

        restarted = KnowledgeBaseService(storage_dir=self.storage_dir)
        self.assertEqual(len(restarted.store.list_documents()), 1)
        self.assertIn("CRASH-882", restarted.rag.answer("crash recovery marker").answer)

    def test_searchable_pdf_is_extracted_locally(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        result = service.upload(
            filename="night-shuttle.pdf",
            data=_searchable_pdf(),
            content_type="application/pdf",
        )

        self.assertEqual(result["document"]["extension"], ".pdf")
        answer = service.rag.answer("Night shuttle code").answer
        self.assertIn("ZX-900", answer)
        self.assertIn("night-shuttle.pdf", answer)

    def test_retrieval_remains_available_during_concurrent_rebuilds(self) -> None:
        service = KnowledgeBaseService(storage_dir=self.storage_dir)
        service.upload(
            filename="concurrency.md",
            data=b"# Atomic Snapshot\n\nconcurrent marker SNAP-551",
            content_type="text/markdown",
        )

        def retrieve_repeatedly() -> None:
            for _ in range(100):
                answer = service.rag.answer("concurrent marker").answer
                self.assertIn("SNAP-551", answer)

        def rebuild_repeatedly() -> None:
            for _ in range(12):
                service.rebuild()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(retrieve_repeatedly) for _ in range(4)]
            futures.append(executor.submit(rebuild_repeatedly))
            for future in futures:
                future.result()

    def test_store_validates_name_type_encoding_and_size(self) -> None:
        store = DocumentStore(self.storage_dir, max_upload_bytes=16)
        with self.assertRaises(UnsupportedDocumentError):
            store.add_document(
                filename="manual.exe",
                data=b"hello",
                content_type="application/octet-stream",
            )
        with self.assertRaises(InvalidDocumentError):
            store.add_document(
                filename="../manual.md",
                data=b"# title\ntext",
                content_type="text/markdown",
            )
        with self.assertRaises(InvalidDocumentError):
            store.add_document(
                filename="manual.txt",
                data=b"\xff\xfe",
                content_type="text/plain",
            )
        with self.assertRaises(DocumentTooLargeError):
            store.add_document(
                filename="large.md",
                data=b"x" * 17,
                content_type="text/markdown",
            )


if __name__ == "__main__":
    unittest.main()
