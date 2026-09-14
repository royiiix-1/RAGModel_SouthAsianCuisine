from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .errors import CorpusValidationError
from .models import RawDocument, SourceMetadata


def stable_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True, slots=True)
class CorpusReport:
    source_files: int
    input_records: int
    valid_documents: int
    duplicate_documents: int
    filtered_documents: int
    invalid_records: int


def _metadata_from_record(raw: object, filename: str, position: int) -> SourceMetadata:
    if not isinstance(raw, dict):
        raise CorpusValidationError(f"{filename}:{position} metadata must be an object")
    required = ("source", "topic", "section", "url")
    missing = [key for key in required if not str(raw.get(key, "")).strip()]
    if missing:
        raise CorpusValidationError(
            f"{filename}:{position} metadata missing fields: {', '.join(missing)}"
        )
    url = str(raw["url"]).strip()
    if urlparse(url).scheme not in {"http", "https"}:
        raise CorpusValidationError(f"{filename}:{position} has a non-HTTP source URL")
    captured_at = raw.get("captured_at") or raw.get("scraped_at") or raw.get("timestamp")
    return SourceMetadata(
        source=str(raw["source"]).strip(),
        topic=str(raw["topic"]).strip(),
        section=str(raw["section"]).strip(),
        url=url,
        captured_at=str(captured_at) if captured_at else None,
    )


def load_corpus(
    raw_data_dir: Path, *, strict: bool = True
) -> tuple[list[RawDocument], CorpusReport]:
    files = sorted(raw_data_dir.glob("*.json"))
    if not files:
        raise CorpusValidationError(f"No JSON corpus files found in {raw_data_dir}")

    documents: list[RawDocument] = []
    seen: set[str] = set()
    input_records = 0
    duplicates = 0
    filtered = 0
    invalid = 0

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusValidationError(f"Cannot read corpus file {path.name}: {exc}") from exc
        if not isinstance(payload, list):
            raise CorpusValidationError(f"Corpus file {path.name} must contain a JSON list")

        for position, item in enumerate(payload, start=1):
            input_records += 1
            try:
                if not isinstance(item, dict):
                    raise CorpusValidationError(f"{path.name}:{position} must be an object")
                text = normalize_text(str(item.get("text", "")))
                if len(text) < 12 or text.lower().strip("- ") in {"read more"}:
                    filtered += 1
                    continue
                metadata = _metadata_from_record(item.get("metadata"), path.name, position)
            except CorpusValidationError:
                invalid += 1
                if strict:
                    raise
                continue

            document_id = stable_id(
                "doc", metadata.source, metadata.topic, metadata.section, text
            )
            if document_id in seen:
                duplicates += 1
                continue
            seen.add(document_id)
            documents.append(
                RawDocument(document_id=document_id, text=text, metadata=metadata)
            )

    report = CorpusReport(
        source_files=len(files),
        input_records=input_records,
        valid_documents=len(documents),
        duplicate_documents=duplicates,
        filtered_documents=filtered,
        invalid_records=invalid,
    )
    return documents, report
