class RagError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(RagError):
    """Raised when runtime configuration is invalid."""


class CorpusValidationError(RagError):
    """Raised when source corpus records do not match the expected schema."""


class ArtifactError(RagError):
    """Raised when an index artifact is missing, stale, or corrupt."""


class QueryValidationError(RagError):
    """Raised when a user query is unsafe to process as supplied."""
