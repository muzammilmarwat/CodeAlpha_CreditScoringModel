"""Custom exceptions for deployment inference services."""


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required model, preprocessing, or documentation artifact is missing."""


class ModelLoadingError(RuntimeError):
    """Raised when an artifact exists but cannot be loaded for inference."""


class PredictionInputError(ValueError):
    """Raised when applicant input does not match the expected prediction schema."""


class PredictionServiceError(RuntimeError):
    """Raised when prediction orchestration fails after validation."""
