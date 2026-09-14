from __future__ import annotations

from collections.abc import Iterable

from .corpus import stable_id
from .models import ChunkRecord, RawDocument

_BOUNDARIES = ("\n\n", "\n", ". ", "; ", ", ", " ")


def _aligned_overlap_start(text: str, previous_start: int, end: int, overlap: int) -> int:
    target = max(previous_start + 1, end - overlap)
    candidates: list[int] = []
    for separator in ("\n", ". ", "; ", ", ", " "):
        location = text.find(separator, target, end)
        if location >= 0:
            candidates.append(location + len(separator))
    return min(candidates) if candidates else target


def split_spans(text: str, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split text deterministically while retaining stable character offsets."""
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid max_chars/overlap")
    if not text:
        return []

    spans: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    minimum_window = max(1, int(max_chars * 0.55))

    while start < length:
        hard_end = min(length, start + max_chars)
        end = hard_end
        if hard_end < length:
            lower = start + minimum_window
            window = text[lower:hard_end]
            for separator in _BOUNDARIES:
                location = window.rfind(separator)
                if location >= 0:
                    end = lower + location + len(separator)
                    break

        piece = text[start:end].strip()
        if piece:
            spans.append((start, end, piece))
        if end >= length:
            break
        next_start = _aligned_overlap_start(text, start, end, overlap)
        while next_start < length and text[next_start].isspace():
            next_start += 1
        start = next_start

    return spans


def chunk_documents(
    documents: Iterable[RawDocument],
    *,
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    seen_chunk_ids: set[str] = set()

    for document in documents:
        header = (
            f"[Topic: {document.metadata.topic} | Section: {document.metadata.section}]\n"
        )
        for parent_start, parent_end, parent in split_spans(
            document.text, parent_size, parent_overlap
        ):
            parent_id = stable_id(
                "parent", document.document_id, parent_start, parent_end, parent
            )
            parent_display = header + parent
            for child_start, child_end, child in split_spans(
                parent, child_size, child_overlap
            ):
                chunk_id = stable_id(
                    "chunk", parent_id, child_start, child_end, child
                )
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        document_id=document.document_id,
                        text=header + child,
                        parent_text=parent_display,
                        metadata=document.metadata,
                    )
                )
    return chunks
