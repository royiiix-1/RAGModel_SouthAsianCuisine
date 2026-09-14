from __future__ import annotations

from .artifacts import ArtifactManifest, write_artifacts
from .chunking import chunk_documents
from .config import Settings
from .corpus import load_corpus
from .embeddings import Embedder, SentenceTransformerEmbedder
from .errors import ArtifactError


def build_index(
    settings: Settings, *, embedder: Embedder | None = None
) -> tuple[ArtifactManifest, dict[str, int]]:
    settings.validate()
    documents, report = load_corpus(settings.raw_data_dir, strict=True)
    chunks = chunk_documents(
        documents,
        parent_size=settings.parent_chunk_size,
        parent_overlap=settings.parent_chunk_overlap,
        child_size=settings.child_chunk_size,
        child_overlap=settings.child_chunk_overlap,
    )
    if not chunks:
        raise ArtifactError("Corpus produced no chunks")

    # This explicitly prevents the collision bug present in the legacy notebook.
    parent_text_by_id: dict[str, str] = {}
    for chunk in chunks:
        existing = parent_text_by_id.setdefault(chunk.parent_id, chunk.parent_text)
        if existing != chunk.parent_text:
            raise ArtifactError(f"Parent ID collision detected: {chunk.parent_id}")

    active_embedder = embedder or SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_revision,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
        local_files_only=settings.offline_mode,
    )
    vectors = active_embedder.encode_documents([chunk.text for chunk in chunks])
    manifest = write_artifacts(
        chunks=chunks,
        vectors=vectors,
        settings=settings,
        source_document_count=len(documents),
    )
    stats = {
        "source_files": report.source_files,
        "input_records": report.input_records,
        "valid_documents": report.valid_documents,
        "duplicate_documents": report.duplicate_documents,
        "filtered_documents": report.filtered_documents,
        "invalid_records": report.invalid_records,
        "parent_chunks": len(parent_text_by_id),
        "child_chunks": len(chunks),
    }
    return manifest, stats
