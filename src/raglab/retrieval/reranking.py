"""Optional local reranking adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class BGEReranker:
    """Lazy local adapter for BAAI/bge-reranker-v2-m3."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = model
        self._tokenizer: Any = None
        self._model: Any = None

    def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install `raglab[retrieval]` to use the BGE reranker") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)  # type: ignore[no-untyped-call]
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            self._model.eval()
        pairs = [[query, document] for document in documents]
        inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        outputs = self._model(**inputs)
        return [float(value) for value in outputs.logits.view(-1).detach().cpu().tolist()]
