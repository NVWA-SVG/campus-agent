"""将 Markdown 按标题和长度切分为可检索片段。"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from campus_agent.rag.models import ChunkMetadata, DocumentChunk


HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 120
BUILT_IN_CLASSIFICATION = {
    "campus_card": ("campus_service", "campus_card"),
    "laboratory": ("facility", "laboratory"),
    "library": ("campus_service", "library"),
    "scholarship": ("academic", "scholarship"),
    "transcript": ("academic", "transcript"),
}


def _split_long_content(
    content: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[str, ...]:
    if len(content) <= max_chars:
        return (content,)

    parts: list[str] = []
    start = 0
    while start < len(content):
        hard_end = min(start + max_chars, len(content))
        end = hard_end
        if hard_end < len(content):
            search_start = start + max_chars // 2
            candidates = [
                content.rfind(separator, search_start, hard_end)
                for separator in ("\n", "。", "！", "？", ";", "；")
            ]
            natural_end = max(candidates)
            if natural_end >= search_start:
                end = natural_end + 1
        part = content[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(content):
            break
        start = max(start + 1, end - overlap_chars)
    return tuple(parts)


def _flush_chunk(
    chunks: list[DocumentChunk],
    document_title: str,
    section_title: str,
    lines: list[str],
    section_index: int,
    *,
    source_name: str,
    chunk_id_prefix: str,
    base_metadata: ChunkMetadata,
) -> None:
    content = "\n".join(line.strip() for line in lines if line.strip()).strip()
    if not content:
        return
    title = section_title or document_title
    for part_index, part in enumerate(_split_long_content(content)):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{chunk_id_prefix}-{section_index}-{part_index}",
                title=title,
                content=part,
                source=f"{source_name}#{title}",
                metadata=replace(
                    base_metadata,
                    chunk_index=section_index * 1000 + part_index,
                ),
            )
        )


def load_markdown_text(
    text: str,
    *,
    source_name: str,
    chunk_id_prefix: str,
    default_title: str,
    metadata: ChunkMetadata | None = None,
) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    document_title = default_title
    section_title = default_title
    section_lines: list[str] = []
    section_index = 0
    base_metadata = metadata or ChunkMetadata(
        document_id=chunk_id_prefix,
        source_name=source_name,
    )

    for raw_line in text.splitlines():
        match = HEADING_PATTERN.match(raw_line)
        if not match:
            section_lines.append(raw_line)
            continue

        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1:
            had_content = any(line.strip() for line in section_lines)
            _flush_chunk(
                chunks,
                document_title,
                section_title,
                section_lines,
                section_index,
                source_name=source_name,
                chunk_id_prefix=chunk_id_prefix,
                base_metadata=base_metadata,
            )
            if had_content:
                section_index += 1
            document_title = heading
            section_title = heading
            section_lines = []
            continue

        _flush_chunk(
            chunks,
            document_title,
            section_title,
            section_lines,
            section_index,
            source_name=source_name,
            chunk_id_prefix=chunk_id_prefix,
            base_metadata=base_metadata,
        )
        section_index += 1
        section_title = f"{document_title} - {heading}"
        section_lines = []

    _flush_chunk(
        chunks,
        document_title,
        section_title,
        section_lines,
        section_index,
        source_name=source_name,
        chunk_id_prefix=chunk_id_prefix,
        base_metadata=base_metadata,
    )
    return tuple(chunks)


def load_markdown_file(
    path: Path,
    *,
    source_name: str | None = None,
    chunk_id_prefix: str | None = None,
    metadata: ChunkMetadata | None = None,
) -> tuple[DocumentChunk, ...]:
    return load_markdown_text(
        path.read_text(encoding="utf-8"),
        source_name=source_name or path.name,
        chunk_id_prefix=chunk_id_prefix or path.stem,
        default_title=path.stem,
        metadata=metadata,
    )


def load_markdown_chunks(directory: Path) -> tuple[DocumentChunk, ...]:
    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在：{directory}")

    chunks: list[DocumentChunk] = []
    for path in sorted(directory.glob("*.md")):
        domain, category = BUILT_IN_CLASSIFICATION.get(
            path.stem,
            ("campus_service", "general"),
        )
        chunks.extend(
            load_markdown_file(
                path,
                metadata=ChunkMetadata(
                    document_id=f"builtin-{path.stem}",
                    source_name=path.name,
                    domain=domain,
                    category=category,
                    origin="built_in",
                ),
            )
        )

    if not chunks:
        raise ValueError(f"知识库中没有可用的 Markdown 内容：{directory}")
    return tuple(chunks)
