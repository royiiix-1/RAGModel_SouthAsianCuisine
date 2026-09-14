from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The user requirement is enforced here as an invariant, not just a default.
GENERATION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
INDEX_PIPELINE_VERSION = 2


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


def _path(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    environment: str = "development"
    log_level: str = "INFO"

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    artifact_dir: Path = PROJECT_ROOT / "artifacts" / "current"
    evaluation_dir: Path = PROJECT_ROOT / "evaluation-results"

    generation_model: str = GENERATION_MODEL_ID
    generation_revision: str = GENERATION_MODEL_REVISION
    embedding_model: str = EMBEDDING_MODEL_ID
    embedding_revision: str = EMBEDDING_MODEL_REVISION
    reranker_model: str = RERANKER_MODEL_ID
    reranker_revision: str = RERANKER_MODEL_REVISION

    device: str = "auto"
    offline_mode: bool = False
    use_reranker: bool = True
    embedding_batch_size: int = 64
    top_k: int = 5
    candidate_k: int = 24
    minimum_similarity: float = 0.65

    parent_chunk_size: int = 1400
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 320
    child_chunk_overlap: int = 50

    max_query_chars: int = 2000
    max_context_chars: int = 12000
    max_generation_sources: int = 1
    max_input_tokens: int = 8192
    max_new_tokens: int = 256
    max_answer_sentences: int = 2
    max_answer_chars: int = 600
    minimum_grounding_precision: float = 0.30
    request_concurrency: int = 1

    api_key: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        root = Path(env.get("RAG_PROJECT_ROOT", str(PROJECT_ROOT))).resolve()
        settings = cls(
            project_root=root,
            environment=env.get("RAG_ENVIRONMENT", "development"),
            log_level=env.get("RAG_LOG_LEVEL", "INFO").upper(),
            raw_data_dir=_path(env.get("RAG_RAW_DATA_DIR", "data/raw"), root),
            artifact_dir=_path(env.get("RAG_ARTIFACT_DIR", "artifacts/current"), root),
            evaluation_dir=_path(
                env.get("RAG_EVALUATION_DIR", "evaluation-results"), root
            ),
            generation_model=env.get("RAG_GENERATION_MODEL", GENERATION_MODEL_ID),
            generation_revision=env.get(
                "RAG_GENERATION_REVISION", GENERATION_MODEL_REVISION
            ),
            embedding_model=env.get("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL_ID),
            embedding_revision=env.get(
                "RAG_EMBEDDING_REVISION", EMBEDDING_MODEL_REVISION
            ),
            reranker_model=env.get("RAG_RERANKER_MODEL", RERANKER_MODEL_ID),
            reranker_revision=env.get(
                "RAG_RERANKER_REVISION", RERANKER_MODEL_REVISION
            ),
            device=env.get("RAG_DEVICE", "auto"),
            offline_mode=_as_bool(env.get("RAG_OFFLINE_MODE", "false")),
            use_reranker=_as_bool(env.get("RAG_USE_RERANKER", "true")),
            embedding_batch_size=int(env.get("RAG_EMBEDDING_BATCH_SIZE", "64")),
            top_k=int(env.get("RAG_TOP_K", "5")),
            candidate_k=int(env.get("RAG_CANDIDATE_K", "24")),
            minimum_similarity=float(env.get("RAG_MINIMUM_SIMILARITY", "0.65")),
            parent_chunk_size=int(env.get("RAG_PARENT_CHUNK_SIZE", "1400")),
            parent_chunk_overlap=int(env.get("RAG_PARENT_CHUNK_OVERLAP", "150")),
            child_chunk_size=int(env.get("RAG_CHILD_CHUNK_SIZE", "320")),
            child_chunk_overlap=int(env.get("RAG_CHILD_CHUNK_OVERLAP", "50")),
            max_query_chars=int(env.get("RAG_MAX_QUERY_CHARS", "2000")),
            max_context_chars=int(env.get("RAG_MAX_CONTEXT_CHARS", "12000")),
            max_generation_sources=int(env.get("RAG_MAX_GENERATION_SOURCES", "1")),
            max_input_tokens=int(env.get("RAG_MAX_INPUT_TOKENS", "8192")),
            max_new_tokens=int(env.get("RAG_MAX_NEW_TOKENS", "256")),
            max_answer_sentences=int(env.get("RAG_MAX_ANSWER_SENTENCES", "2")),
            max_answer_chars=int(env.get("RAG_MAX_ANSWER_CHARS", "600")),
            minimum_grounding_precision=float(
                env.get("RAG_MINIMUM_GROUNDING_PRECISION", "0.30")
            ),
            request_concurrency=int(env.get("RAG_REQUEST_CONCURRENCY", "1")),
            api_key=env.get("RAG_API_KEY") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.generation_model != GENERATION_MODEL_ID:
            raise ConfigurationError(
                f"Generation model is locked to {GENERATION_MODEL_ID}; "
                f"received {self.generation_model!r}."
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("RAG_DEVICE must be one of: auto, cpu, cuda")
        if not 1 <= self.top_k <= 10:
            raise ConfigurationError("RAG_TOP_K must be between 1 and 10")
        if not self.top_k <= self.candidate_k <= 200:
            raise ConfigurationError("RAG_CANDIDATE_K must be >= top_k and <= 200")
        for name, size, overlap in (
            ("parent", self.parent_chunk_size, self.parent_chunk_overlap),
            ("child", self.child_chunk_size, self.child_chunk_overlap),
        ):
            if size < 100 or overlap < 0 or overlap >= size:
                raise ConfigurationError(f"Invalid {name} chunk size/overlap")
        if self.child_chunk_size > self.parent_chunk_size:
            raise ConfigurationError("Child chunks cannot be larger than parent chunks")
        if not 0.0 <= self.minimum_similarity <= 1.0:
            raise ConfigurationError("RAG_MINIMUM_SIMILARITY must be in [0, 1]")
        if not 0.0 <= self.minimum_grounding_precision <= 1.0:
            raise ConfigurationError("RAG_MINIMUM_GROUNDING_PRECISION must be in [0, 1]")
        if self.max_query_chars < 32 or self.max_context_chars < 1000:
            raise ConfigurationError("Query/context limits are unreasonably small")
        if not 1 <= self.max_generation_sources <= self.top_k:
            raise ConfigurationError("Generation source count must be between 1 and top_k")
        if self.max_answer_sentences < 1 or self.max_answer_chars < 100:
            raise ConfigurationError("Answer limits are unreasonably small")
        if self.request_concurrency < 1:
            raise ConfigurationError("RAG_REQUEST_CONCURRENCY must be positive")

    def index_configuration(self) -> dict[str, object]:
        return {
            "index_pipeline_version": INDEX_PIPELINE_VERSION,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "child_chunk_size": self.child_chunk_size,
            "child_chunk_overlap": self.child_chunk_overlap,
        }

    def index_fingerprint(self) -> str:
        payload = json.dumps(
            self.index_configuration(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def public_summary(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("api_key", None)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
