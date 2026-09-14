from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import ArtifactError
from .models import ChunkRecord

MANIFEST_SCHEMA_VERSION = 2
INDEX_FILENAME = "index.faiss"
RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_file_hashes(raw_data_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(raw_data_dir.glob("*.json"))
    }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: int
    created_at: str
    index_fingerprint: str
    embedding_model: str
    embedding_revision: str
    index_engine: str
    vector_dimension: int
    source_document_count: int
    chunk_count: int
    source_files: dict[str, str]
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                created_at=str(data["created_at"]),
                index_fingerprint=str(data["index_fingerprint"]),
                embedding_model=str(data["embedding_model"]),
                embedding_revision=str(data["embedding_revision"]),
                index_engine=str(data["index_engine"]),
                vector_dimension=int(data["vector_dimension"]),
                source_document_count=int(data["source_document_count"]),
                chunk_count=int(data["chunk_count"]),
                source_files={str(k): str(v) for k, v in data["source_files"].items()},
                files={str(k): str(v) for k, v in data["files"].items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("Artifact manifest has an invalid schema") from exc


class ArtifactStore:
    def __init__(self, index: Any, records: list[ChunkRecord], manifest: ArtifactManifest):
        self.index = index
        self.records = records
        self.manifest = manifest

    @classmethod
    def load(cls, settings: Settings, *, verify_sources: bool = True) -> ArtifactStore:
        artifact_dir = settings.artifact_dir
        manifest_path = artifact_dir / MANIFEST_FILENAME
        index_path = artifact_dir / INDEX_FILENAME
        records_path = artifact_dir / RECORDS_FILENAME
        missing = [
            str(path)
            for path in (manifest_path, index_path, records_path)
            if not path.is_file()
        ]
        if missing:
            raise ArtifactError(
                "Index artifacts are missing. Run `rag-cuisine build-index`. Missing: "
                + ", ".join(missing)
            )

        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("Cannot read artifact manifest") from exc
        manifest = ArtifactManifest.from_dict(manifest_payload)

        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ArtifactError("Artifact schema is not supported; rebuild the index")
        if manifest.index_fingerprint != settings.index_fingerprint():
            raise ArtifactError("Artifact configuration differs from runtime configuration")
        if (
            manifest.embedding_model != settings.embedding_model
            or manifest.embedding_revision != settings.embedding_revision
        ):
            raise ArtifactError("Artifact embedding model/revision does not match runtime")
        for filename, path in (
            (INDEX_FILENAME, index_path),
            (RECORDS_FILENAME, records_path),
        ):
            if manifest.files.get(filename) != sha256_file(path):
                raise ArtifactError(f"Artifact checksum failed for {filename}")

        if verify_sources and corpus_file_hashes(settings.raw_data_dir) != manifest.source_files:
            raise ArtifactError("Source corpus changed after indexing; rebuild the index")

        records: list[ChunkRecord] = []
        try:
            with records_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip():
                        try:
                            records.append(ChunkRecord.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                            raise ArtifactError(
                                f"Invalid record at records.jsonl:{line_number}"
                            ) from exc
        except OSError as exc:
            raise ArtifactError("Cannot read artifact records") from exc

        try:
            import faiss
        except ImportError as exc:
            raise ArtifactError("faiss-cpu is not installed") from exc
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(records) or len(records) != manifest.chunk_count:
            raise ArtifactError("FAISS index and record count do not match")
        if index.d != manifest.vector_dimension:
            raise ArtifactError("FAISS vector dimension does not match the manifest")
        return cls(index=index, records=records, manifest=manifest)

    def search(self, query_vector: Any, k: int) -> list[tuple[ChunkRecord, float]]:
        try:
            import numpy as np
        except ImportError as exc:
            raise ArtifactError("numpy is not installed") from exc
        vector = np.asarray(query_vector, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self.manifest.vector_dimension):
            raise ArtifactError(
                f"Query vector shape {vector.shape} does not match index dimension "
                f"{self.manifest.vector_dimension}"
            )
        distances, indices = self.index.search(vector, min(k, len(self.records)))
        return [
            (self.records[int(index)], float(score))
            for index, score in zip(indices[0], distances[0], strict=True)
            if index >= 0
        ]


def write_artifacts(
    *,
    chunks: list[ChunkRecord],
    vectors: Any,
    settings: Settings,
    source_document_count: int,
) -> ArtifactManifest:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise ArtifactError("faiss-cpu and numpy are required to build the index") from exc

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or not len(chunks):
        raise ArtifactError("Embedding matrix does not match the generated chunks")
    if not np.isfinite(matrix).all():
        raise ArtifactError("Embedding matrix contains non-finite values")
    faiss.normalize_L2(matrix)

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    temp_index = settings.artifact_dir / f".{INDEX_FILENAME}.{suffix}.tmp"
    temp_records = settings.artifact_dir / f".{RECORDS_FILENAME}.{suffix}.tmp"
    temp_manifest = settings.artifact_dir / f".{MANIFEST_FILENAME}.{suffix}.tmp"

    try:
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        faiss.write_index(index, str(temp_index))

        with temp_records.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(
                    json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

        manifest = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            index_fingerprint=settings.index_fingerprint(),
            embedding_model=settings.embedding_model,
            embedding_revision=settings.embedding_revision,
            index_engine="faiss.IndexFlatIP",
            vector_dimension=int(matrix.shape[1]),
            source_document_count=source_document_count,
            chunk_count=len(chunks),
            source_files=corpus_file_hashes(settings.raw_data_dir),
            files={
                INDEX_FILENAME: sha256_file(temp_index),
                RECORDS_FILENAME: sha256_file(temp_records),
            },
        )
        temp_manifest.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Manifest is replaced last so readers never accept a partially updated build.
        os.replace(temp_index, settings.artifact_dir / INDEX_FILENAME)
        os.replace(temp_records, settings.artifact_dir / RECORDS_FILENAME)
        os.replace(temp_manifest, settings.artifact_dir / MANIFEST_FILENAME)
        return manifest
    finally:
        for path in (temp_index, temp_records, temp_manifest):
            path.unlink(missing_ok=True)
