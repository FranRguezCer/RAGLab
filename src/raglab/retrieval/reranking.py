"""Optional local reranking adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_MAX_LENGTH = 512
_BATCH_SIZE = 4


class BGEReranker:
    """Lazy local adapter for BAAI/bge-reranker-v2-m3."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = model
        self._tokenizer: Any = None
        self._model: Any = None

    def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install `raglab[retrieval]` to use the BGE reranker") from exc
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install `raglab[retrieval]` to use the BGE reranker") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)  # type: ignore[no-untyped-call]
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            self._model.eval()
        pairs = [[query, document] for document in documents]
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), _BATCH_SIZE):
                inputs = self._tokenizer(
                    pairs[start : start + _BATCH_SIZE],
                    padding=True,
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    return_tensors="pt",
                )
                outputs = self._model(**inputs)
                scores.extend(
                    float(value) for value in outputs.logits.view(-1).detach().cpu().tolist()
                )
        return scores
