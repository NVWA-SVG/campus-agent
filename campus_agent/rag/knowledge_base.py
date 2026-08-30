"""动态知识库：安全摄取文件、持久化元数据并原子更新RAG快照。"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pypdf import PdfReader

from campus_agent.rag.chunking import load_markdown_file, load_markdown_text
from campus_agent.rag.models import ChunkMetadata, DocumentChunk
from campus_agent.rag.service import DEFAULT_KNOWLEDGE_DIR, LocalRAG


MAX_UPLOAD_BYTES: Final = 10 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 100 * 1024 * 1024
MAX_DOCUMENTS: Final = 100
MAX_PDF_PAGES: Final = 100
MAX_EXTRACTED_CHARS: Final = 1_000_000
DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
METADATA_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STAGED_DELETE_PATTERN = re.compile(r"^\.trash-([0-9a-f]{32})-[0-9a-f]{32}$")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
ALLOWED_CONTENT_TYPES = {
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
}


class KnowledgeBaseError(RuntimeError):
    """动态知识库错误基类。"""


class InvalidDocumentError(KnowledgeBaseError):
    """文件名、内容或解析结果无效。"""


class UnsupportedDocumentError(KnowledgeBaseError):
    """文件类型或声明的内容类型不受支持。"""


class DocumentTooLargeError(KnowledgeBaseError):
    """单文件、总容量或解析后文本超过限制。"""


class DuplicateDocumentError(KnowledgeBaseError):
    """相同内容已经存在。"""


class DocumentNotFoundError(KnowledgeBaseError):
    """文档ID不存在。"""


def _validate_metadata_value(field_name: str, value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.strip())
    if not METADATA_VALUE_PATTERN.fullmatch(normalized):
        raise InvalidDocumentError(
            f"{field_name} 只能包含1–64位字母、数字、点、下划线或连字符"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    display_name: str
    extension: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: str
    chunk_count: int
    domain: str = "uploaded"
    category: str = "general"
    visibility: str = "public"
    version: str = "1"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "KnowledgeDocument":
        if not isinstance(data, dict):
            raise InvalidDocumentError("文档元数据不是JSON对象")
        try:
            document = cls(
                document_id=str(data["document_id"]),
                display_name=str(data["display_name"]),
                extension=str(data["extension"]),
                content_type=str(data["content_type"]),
                size_bytes=int(data["size_bytes"]),
                sha256=str(data["sha256"]),
                created_at=str(data["created_at"]),
                chunk_count=int(data["chunk_count"]),
                domain=str(data.get("domain", "uploaded")),
                category=str(data.get("category", "general")),
                visibility=str(data.get("visibility", "public")),
                version=str(data.get("version", "1")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidDocumentError("文档元数据字段不完整") from error
        if not DOCUMENT_ID_PATTERN.fullmatch(document.document_id):
            raise InvalidDocumentError("文档元数据包含非法ID")
        normalized_name, extension = _validate_display_name(document.display_name)
        if normalized_name != document.display_name or extension != document.extension:
            raise InvalidDocumentError("文档元数据中的文件名或扩展名不一致")
        if _validate_content_type(extension, document.content_type) != document.content_type:
            raise InvalidDocumentError("文档元数据中的Content-Type不规范")
        if not 0 < document.size_bytes <= MAX_UPLOAD_BYTES:
            raise InvalidDocumentError("文档元数据中的文件大小无效")
        if not SHA256_PATTERN.fullmatch(document.sha256):
            raise InvalidDocumentError("文档元数据中的SHA256无效")
        if document.chunk_count < 1:
            raise InvalidDocumentError("文档元数据中的Chunk数量无效")
        try:
            created_at = datetime.fromisoformat(document.created_at)
        except ValueError as error:
            raise InvalidDocumentError("文档元数据中的创建时间无效") from error
        if created_at.tzinfo is None:
            raise InvalidDocumentError("文档元数据中的创建时间必须带时区")
        for field_name in ("domain", "category", "version"):
            _validate_metadata_value(field_name, getattr(document, field_name))
        if document.visibility != "public":
            raise InvalidDocumentError("上传文档的visibility必须为public")
        return document


def default_storage_dir() -> Path:
    configured = os.getenv("CAMPUS_KNOWLEDGE_DIR")
    path = (
        Path(configured)
        if configured
        else Path.cwd() / ".campus_agent_data" / "knowledge"
    )
    return path.expanduser().absolute()


def _validate_display_name(filename: str) -> tuple[str, str]:
    normalized = unicodedata.normalize("NFC", filename.strip())
    if not normalized or len(normalized) > 120:
        raise InvalidDocumentError("文件名不能为空且最多120个字符")
    if normalized.endswith((".", " ")):
        raise InvalidDocumentError("文件名不能以点或空格结尾")
    if any(
        character in normalized
        for character in ("/", "\\", ":", "<", ">", '"', "|", "?", "*")
    ):
        raise InvalidDocumentError("文件名包含路径或非法字符")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise InvalidDocumentError("文件名包含控制字符")

    path = Path(normalized)
    extension = path.suffix.lower()
    if extension not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedDocumentError("只支持 .md、.txt 和 .pdf 文件")
    inner_suffixes = [suffix.lower() for suffix in path.suffixes[:-1]]
    if any(suffix in ALLOWED_CONTENT_TYPES for suffix in inner_suffixes):
        raise InvalidDocumentError("不接受伪装成双扩展名的知识文件")
    if path.stem.upper() in WINDOWS_RESERVED_NAMES:
        raise InvalidDocumentError("文件名是Windows保留名称")
    return normalized, extension


def _validate_content_type(extension: str, content_type: str) -> str:
    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
    normalized = normalized or "application/octet-stream"
    if normalized not in ALLOWED_CONTENT_TYPES[extension]:
        raise UnsupportedDocumentError("文件扩展名与Content-Type不一致")
    return normalized


def _validate_text(text: str) -> str:
    if not text.strip():
        raise InvalidDocumentError("文档中没有可用文本")
    if len(text) > MAX_EXTRACTED_CHARS:
        raise DocumentTooLargeError("解析后的文本超过100万字符")
    invalid_controls = [
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\r\t"
    ]
    if invalid_controls:
        raise InvalidDocumentError("文本包含二进制控制字符")
    return text.strip()


def _extract_markdown(
    display_name: str,
    extension: str,
    data: bytes,
) -> str:
    title = Path(display_name).stem
    if extension in {".md", ".txt"}:
        if data.startswith(b"%PDF-") or b"\x00" in data:
            raise InvalidDocumentError("文本文件包含PDF或二进制内容")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise InvalidDocumentError("文本文件必须使用UTF-8编码") from error
        text = _validate_text(text)
        if extension == ".txt" or not re.search(r"^#\s+", text, flags=re.MULTILINE):
            return f"# {title}\n\n{text}"
        return text

    if not data.startswith(b"%PDF-"):
        raise InvalidDocumentError("PDF文件签名无效")
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise InvalidDocumentError("暂不支持加密PDF")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentTooLargeError("PDF最多允许100页")

        sections = [f"# {title}"]
        extracted_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            extracted_chars += len(page_text)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise DocumentTooLargeError("PDF提取文本超过100万字符")
            sections.append(f"## 第{page_number}页\n\n{page_text}")
    except (InvalidDocumentError, DocumentTooLargeError):
        raise
    except Exception as error:
        raise InvalidDocumentError("PDF损坏或无法安全解析") from error

    if len(sections) == 1:
        raise InvalidDocumentError("PDF没有可提取文本，当前版本不进行联网OCR")
    return _validate_text("\n\n".join(sections))


class DocumentStore:
    def __init__(
        self,
        storage_dir: Path | None = None,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_documents: int = MAX_DOCUMENTS,
    ) -> None:
        configured_path = storage_dir or default_storage_dir()
        expanded_path = configured_path.expanduser().absolute()
        if expanded_path.exists() and expanded_path.is_symlink():
            raise InvalidDocumentError("知识库存储目录不能是符号链接")
        self.storage_dir = expanded_path.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_total_bytes = max_total_bytes
        self.max_documents = max_documents
        self._lock = threading.RLock()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._recover_staged_deletions()

    def add_document(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str,
        domain: str = "uploaded",
        category: str = "general",
        version: str = "1",
    ) -> KnowledgeDocument:
        display_name, extension = _validate_display_name(filename)
        normalized_content_type = _validate_content_type(extension, content_type)
        normalized_domain = _validate_metadata_value("domain", domain)
        normalized_category = _validate_metadata_value("category", category)
        normalized_version = _validate_metadata_value("version", version)
        if not data:
            raise InvalidDocumentError("不能上传空文件")
        if len(data) > self.max_upload_bytes:
            limit_mib = self.max_upload_bytes / (1024 * 1024)
            raise DocumentTooLargeError(f"单个文件不能超过{limit_mib:g} MiB")

        digest = hashlib.sha256(data).hexdigest()
        markdown = _extract_markdown(display_name, extension, data)
        document_id = uuid.uuid4().hex
        chunks = load_markdown_text(
            markdown,
            source_name=display_name,
            chunk_id_prefix=document_id,
            default_title=Path(display_name).stem,
        )
        if not chunks:
            raise InvalidDocumentError("文档没有产生可检索片段")

        document = KnowledgeDocument(
            document_id=document_id,
            display_name=display_name,
            extension=extension,
            content_type=normalized_content_type,
            size_bytes=len(data),
            sha256=digest,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            chunk_count=len(chunks),
            domain=normalized_domain,
            category=normalized_category,
            version=normalized_version,
        )

        with self._lock:
            existing = self.list_documents()
            if len(existing) >= self.max_documents:
                raise DocumentTooLargeError("知识库最多允许100个上传文档")
            if sum(item.size_bytes for item in existing) + len(data) > self.max_total_bytes:
                raise DocumentTooLargeError("上传知识库总容量不能超过100 MiB")
            if any(item.sha256 == digest for item in existing):
                raise DuplicateDocumentError("相同内容的文档已经上传")
            self._write_document(document, markdown)
        return document

    def list_documents(self) -> tuple[KnowledgeDocument, ...]:
        with self._lock:
            documents: list[KnowledgeDocument] = []
            for path in sorted(self.storage_dir.iterdir()):
                if not path.is_dir() or path.is_symlink():
                    continue
                if not DOCUMENT_ID_PATTERN.fullmatch(path.name):
                    continue
                try:
                    document = self._read_document(path.name)
                except (OSError, json.JSONDecodeError, KnowledgeBaseError):
                    continue
                documents.append(document)
            return tuple(sorted(documents, key=lambda item: item.created_at, reverse=True))

    def load_chunks(self) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        for document in self.list_documents():
            content_path = self._document_dir(document.document_id) / "content.md"
            chunks.extend(
                load_markdown_file(
                    content_path,
                    source_name=document.display_name,
                    chunk_id_prefix=document.document_id,
                    metadata=ChunkMetadata(
                        document_id=document.document_id,
                        source_name=document.display_name,
                        domain=document.domain,
                        category=document.category,
                        visibility=document.visibility,
                        version=document.version,
                        origin="uploaded",
                    ),
                )
            )
        return tuple(chunks)

    def stage_delete(self, document_id: str) -> Path:
        with self._lock:
            self._read_document(document_id)
            source = self._document_dir(document_id)
            staged = self.storage_dir / f".trash-{document_id}-{uuid.uuid4().hex}"
            source.replace(staged)
            return staged

    def restore_staged(self, document_id: str, staged: Path) -> None:
        with self._lock:
            destination = self._document_dir(document_id)
            if (
                staged.parent.resolve() != self.storage_dir
                or not STAGED_DELETE_PATTERN.fullmatch(staged.name)
                or not staged.is_dir()
                or staged.is_symlink()
            ):
                raise InvalidDocumentError("暂存删除路径无效")
            staged.replace(destination)

    def finalize_staged(self, staged: Path) -> None:
        with self._lock:
            if (
                staged.parent.resolve() != self.storage_dir
                or not STAGED_DELETE_PATTERN.fullmatch(staged.name)
                or not staged.is_dir()
                or staged.is_symlink()
            ):
                raise InvalidDocumentError("暂存删除路径无效")
            shutil.rmtree(staged)

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            path = self._document_dir(document_id)
            if path.exists():
                shutil.rmtree(path)

    def _write_document(self, document: KnowledgeDocument, markdown: str) -> None:
        temporary = self.storage_dir / f".upload-{document.document_id}-{uuid.uuid4().hex}"
        destination = self._document_dir(document.document_id)
        temporary.mkdir(exist_ok=False)
        try:
            (temporary / "content.md").write_text(markdown, encoding="utf-8")
            (temporary / "metadata.json").write_text(
                json.dumps(document.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _read_document(self, document_id: str) -> KnowledgeDocument:
        path = self._document_dir(document_id)
        if not path.is_dir() or path.is_symlink():
            raise DocumentNotFoundError("知识文档不存在")
        metadata_path = path / "metadata.json"
        content_path = path / "content.md"
        if (
            metadata_path.is_symlink()
            or content_path.is_symlink()
            or not metadata_path.is_file()
            or not content_path.is_file()
        ):
            raise InvalidDocumentError("知识文档存储结构无效")
        document = KnowledgeDocument.from_dict(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        if document.document_id != document_id:
            raise InvalidDocumentError("文档ID与目录不一致")
        return document

    def _document_dir(self, document_id: str) -> Path:
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise DocumentNotFoundError("知识文档不存在")
        path = (self.storage_dir / document_id).resolve()
        if path.parent != self.storage_dir:
            raise DocumentNotFoundError("知识文档不存在")
        return path

    def _recover_staged_deletions(self) -> None:
        """启动时回滚未提交的删除事务，优先避免资料意外丢失。"""

        with self._lock:
            for staged in self.storage_dir.iterdir():
                match = STAGED_DELETE_PATTERN.fullmatch(staged.name)
                if match is None or not staged.is_dir() or staged.is_symlink():
                    continue
                destination = self._document_dir(match.group(1))
                if destination.exists():
                    shutil.rmtree(staged)
                else:
                    staged.replace(destination)


class KnowledgeBaseService:
    """把磁盘文档事务与LocalRAG快照更新组合成一个服务。"""

    def __init__(
        self,
        storage_dir: Path | None = None,
        *,
        built_in_dir: Path = DEFAULT_KNOWLEDGE_DIR,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self.store = DocumentStore(
            storage_dir,
            max_upload_bytes=max_upload_bytes,
        )
        self.rag = LocalRAG(
            built_in_dir,
            extra_chunk_loader=self.store.load_chunks,
        )
        self._mutation_lock = threading.RLock()
        self._built_in_dir = built_in_dir

    def upload(
        self,
        *,
        filename: str,
        data: bytes,
        content_type: str,
        domain: str = "uploaded",
        category: str = "general",
        version: str = "1",
    ) -> dict[str, object]:
        with self._mutation_lock:
            document = self.store.add_document(
                filename=filename,
                data=data,
                content_type=content_type,
                domain=domain,
                category=category,
                version=version,
            )
            try:
                self.rag.reload()
            except Exception:
                self.store.remove_document(document.document_id)
                raise
            return {"document": document.as_dict(), "stats": self.stats()}

    def delete(self, document_id: str) -> dict[str, object]:
        with self._mutation_lock:
            staged = self.store.stage_delete(document_id)
            try:
                self.rag.reload()
            except Exception:
                self.store.restore_staged(document_id, staged)
                raise
            self.store.finalize_staged(staged)
            return {"deleted": True, "document_id": document_id, "stats": self.stats()}

    def rebuild(self) -> dict[str, object]:
        with self._mutation_lock:
            self.rag.reload()
            return self.stats()

    def documents(self) -> dict[str, object]:
        with self._mutation_lock:
            return {
                "documents": [item.as_dict() for item in self.store.list_documents()],
                "stats": self.stats(),
            }

    def stats(self) -> dict[str, object]:
        with self._mutation_lock:
            uploaded = self.store.list_documents()
            return {
                "built_in_documents": len(tuple(self._built_in_dir.glob("*.md"))),
                "uploaded_documents": len(uploaded),
                "chunk_count": len(self.rag.chunks),
                "index_version": self.rag.version,
                "built_at": self.rag.built_at,
                "retrieval_strategy": "hybrid",
                "embedding_provider": self.rag.embedding_provider_name,
                "embedding_model": self.rag.embedding_model_name,
                "embedding_revision": self.rag.embedding_revision,
                "embedding_fingerprint": self.rag.embedding_fingerprint,
                "vector_status": self.rag.vector_status,
                "vector_degraded_reason": self.rag.vector_degraded_reason,
                "vector_cache": self.rag.vector_cache_stats,
                "uploaded_bytes": sum(item.size_bytes for item in uploaded),
            }
