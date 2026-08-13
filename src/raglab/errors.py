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
