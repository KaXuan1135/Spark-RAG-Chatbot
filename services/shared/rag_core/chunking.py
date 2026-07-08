from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    document_id: str
    source_file: str
    markdown_path: str
    chunk_id: str
    chunk_text: str
    section_title: str
    heading_path: str
    token_count: int
    content_hash: str


def strip_front_matter(markdown: str) -> str:
    return FRONT_MATTER_RE.sub("", markdown, count=1).strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "document"


def split_markdown_sections(markdown: str) -> list[tuple[str, str, str]]:
    sections: list[tuple[str, str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path = "Document"
    current_title = "Document"

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_title, current_path, text))

    for line in strip_front_matter(markdown).splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_lines = [line]
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = [(lvl, text) for lvl, text in heading_stack if lvl < level]
            heading_stack.append((level, title))
            current_title = title
            current_path = " > ".join(text for _, text in heading_stack)
        else:
            current_lines.append(line)

    flush()
    return sections


def window_words(words: list[str], max_words: int, overlap_words: int) -> Iterable[list[str]]:
    if not words:
        return

    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        yield words[start : start + max_words]
        if start + max_words >= len(words):
            break
        start += step


def chunk_markdown_file(
    markdown_path: Path,
    markdown_text: str,
    *,
    max_words: int,
    overlap_words: int,
) -> list[Chunk]:
    document_id = slugify(markdown_path.stem)
    source_file = f"{markdown_path.stem}.pdf"
    chunks: list[Chunk] = []

    for section_title, heading_path, section_text in split_markdown_sections(markdown_text):
        words = section_text.split()
        if len(words) <= max_words:
            chunk_texts = [(section_text.strip(), len(words))]
        else:
            chunk_texts = [(" ".join(window).strip(), len(window)) for window in window_words(words, max_words=max_words, overlap_words=overlap_words)]

        for chunk_text, token_count in chunk_texts:
            if not chunk_text:
                continue
            digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = f"{document_id}:{len(chunks):05d}"
            chunks.append(
                Chunk(
                    document_id=document_id,
                    source_file=source_file,
                    markdown_path=str(markdown_path),
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    section_title=section_title,
                    heading_path=heading_path,
                    token_count=token_count,
                    content_hash=digest,
                )
            )

    return chunks
