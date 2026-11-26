"""Import a local official SCAU handbook PDF through the existing Markdown path.

Long handbooks are converted locally; the web PDF upload limit stays unchanged.
Original PDF, page numbers, outline titles and provenance are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from campus_agent.rag.chunking import _split_long_content
from campus_agent.rag.knowledge_base import KnowledgeBaseService, default_storage_dir
from campus_agent.rag.models import RetrievalFilter


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PAGE = "https://jwc.scau.edu.cn/2026/0831/c19374a442576/page.htm"
SOURCE_PDF = "https://jwc.scau.edu.cn/_upload/article/files/9f/4f/ef3f14004af6a9d5e977bc6de0e2/1ecb0b20-60a8-418d-a407-0c102c63d3d2.pdf"
MAX_CHARS = 320
OVERLAP_CHARS = 48
PAGE_NUMBER = re.compile(r"^-?\s*(\d+)\s*-?$")


def outline_pages(reader: PdfReader) -> dict[int, list[str]]:
    titles: dict[int, list[str]] = {}

    def walk(items: list) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            title = str(item.get("/Title", "")).strip()
            page_index = reader.get_destination_page_number(item)
            if title and page_index is not None and page_index >= 0:
                page_titles = titles.setdefault(page_index + 1, [])
                if title not in page_titles:
                    page_titles.append(title)

    walk(reader.outline)
    return titles


def clean_page(text: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in text.replace("\x00", "").splitlines() if line.strip()]
    printed_page = None
    if lines and (match := PAGE_NUMBER.fullmatch(lines[0])):
        printed_page = match.group(1)
        lines.pop(0)
    # Join wrapped Chinese lines; retain punctuation and clause boundaries.
    paragraphs: list[str] = []
    for line in lines:
        if not paragraphs or re.match(r"第[一二三四五六七八九十百零〇\d]+[条章]|[（(][一二三四五六七八九十\d]+[）)]", line):
            paragraphs.append(line)
        else:
            paragraphs[-1] += line
    return "\n".join(paragraphs), printed_page


def convert(reader: PdfReader, year: str) -> tuple[str, dict[str, object]]:
    titles = outline_pages(reader)
    if not titles:
        raise ValueError("PDF 没有规章目录书签，请先核对原文件")
    first_body_page = min(titles)
    current_title = "正文"
    sections = [f"# 华南农业大学{year}学生手册"]
    extracted_chars = 0
    part_count = 0
    empty_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_number < first_body_page:
            continue  # Cover and table of contents stay in the source PDF.
        page_titles = titles.get(page_number, [current_title])
        # This edition wraps eight long policy names over same-page bookmarks.
        page_title = "".join(page_titles)
        current_title = page_title
        content, printed_page = clean_page(page.extract_text() or "")
        if not content:
            empty_pages.append(page_number)
            continue
        extracted_chars += len(content)
        if extracted_chars > 1_000_000:
            raise ValueError("提取正文超过100万字符")
        page_label = f"PDF第{page_number}页"
        if printed_page:
            page_label += f"·印刷第{printed_page}页"
        for index, part in enumerate(
            _split_long_content(content, max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS),
            start=1,
        ):
            sections.append(f"## {page_title}｜{page_label}｜片段{index}\n\n{part}")
            part_count += 1
    if empty_pages:
        raise ValueError(f"正文存在无法提取文本的页，请先检查/OCR：{empty_pages}")
    if not part_count:
        raise ValueError("没有可导入正文")
    return "\n\n".join(sections) + "\n", {
        "pdf_pages": len(reader.pages),
        "first_body_pdf_page": first_body_page,
        "excluded_front_matter_pages": first_body_page - 1,
        "extracted_chars_without_overlap": extracted_chars,
        "markdown_sections": part_count,
        "max_chars": MAX_CHARS,
        "overlap_chars": OVERLAP_CHARS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--year", default="2026", choices=("2026",))
    parser.add_argument("--storage-dir", type=Path, default=default_storage_dir())
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".campus_agent_data" / "source_documents")
    parser.add_argument("--check-query", action="append", default=[])
    parser.add_argument("--replace-document-id", help="成功导入后替换同名、同版本的旧导入")
    args = parser.parse_args()
    if not args.pdf.is_file() or args.pdf.stat().st_size > 25 * 1024 * 1024:
        raise SystemExit("需要不超过25 MiB的本地PDF")
    reader = PdfReader(args.pdf, strict=True)
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 600:
        raise SystemExit("PDF加密或页数不在1–600之间")
    identity = "".join((page.extract_text() or "") for page in reader.pages[:8])
    if "华南农业大学" not in re.sub(r"\s+", "", identity) or "学生手册" not in re.sub(r"\s+", "", identity):
        raise SystemExit("文件内容与华南农业大学学生手册不匹配")
    markdown, conversion = convert(reader, args.year)
    data = markdown.encode("utf-8")
    service = KnowledgeBaseService(storage_dir=args.storage_dir)
    if args.replace_document_id:
        old_document = next((doc for doc in service.store.list_documents() if doc.document_id == args.replace_document_id), None)
        if old_document is None or (
            old_document.display_name != f"华南农业大学{args.year}学生手册.md"
            or old_document.domain != "scau"
            or old_document.category != "student_handbook"
            or old_document.version != args.year
        ):
            raise SystemExit("替换目标必须是已存在的同名、同分类、同版本手册")
    digest = hashlib.sha256(data).hexdigest()
    existing = next((doc for doc in service.store.list_documents() if doc.sha256 == digest), None)
    if existing is None:
        uploaded = service.upload(
            filename=f"华南农业大学{args.year}学生手册.md",
            data=data,
            content_type="text/markdown",
            domain="scau",
            category="student_handbook",
            version=args.year,
        )
        document = uploaded["document"]
    else:
        document = existing.as_dict()
    if args.replace_document_id and args.replace_document_id != document["document_id"]:
        service.delete(args.replace_document_id)
    checks = []
    for query in args.check_query:
        for scope, filters in (
            ("whole_corpus", None),
            ("handbook_only", RetrievalFilter(document_id=document["document_id"])),
        ):
            hits = service.rag.retrieve(query, filters=filters)
            checks.append({
                "query": query,
                "scope": scope,
                "hits": [{"chunk_id": hit.chunk.chunk_id, "title": hit.chunk.title, "content": hit.chunk.content} for hit in hits],
            })
    with args.pdf.open("rb") as source_stream:
        pdf_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()
    manifest = {
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "publisher": "华南农业大学本科生院（招生办公室）",
        "published_date": "2026-08-31",
        "edition": args.year,
        "source_page": SOURCE_PAGE,
        "source_pdf_url": SOURCE_PDF,
        "local_pdf": str(args.pdf.resolve()),
        "pdf_sha256": pdf_sha256,
        "markdown_sha256": digest,
        "markdown_newline": "LF",
        "conversion": conversion,
        "document": document,
        "already_imported": existing is not None,
        "stats": service.stats(),
        "retrieval_checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.output_dir / f"scau-student-handbook-{args.year}.md"
    manifest_path = args.output_dir / f"scau-student-handbook-{args.year}.json"
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
