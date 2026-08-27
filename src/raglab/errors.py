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


class ChunkingError(RagLabError):
    pass


class EmbeddingError(RagLabError):
    pass


class StorageError(RagLabError):
    pass


class GenerationError(RagLabError):
    """Generation failed or crossed the strict grounding boundary."""


class GenerationContractError(GenerationError):
    """A completed model response violated the strict generation contract."""

    def __init__(
        self,
        message: str,
        *,
        raw_output: str,
        answer: str | None = None,
        abstained: bool | None = None,
        cited_source_ids: tuple[str, ...] | None = None,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
        model_calls: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.answer = answer
        self.abstained = abstained
        self.cited_source_ids = cited_source_ids
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.model_calls = model_calls


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
