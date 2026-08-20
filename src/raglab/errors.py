class RagLabError(Exception):
    """Base error for actionable pipeline failures."""


class ConversionError(RagLabError):
    pass


class EmptySourceError(ConversionError):
    pass


class UnsafeRemoteURLError(ConversionError):
    pass


class ParsingError(RagLabError):
    pass


class EmbeddingError(RagLabError):
    pass


class StorageError(RagLabError):
    pass


class GenerationError(RagLabError):
    """Generation failed or crossed the strict grounding boundary."""


class GenerationLengthError(GenerationError):
    """The model exhausted its output allowance before completing valid JSON."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
