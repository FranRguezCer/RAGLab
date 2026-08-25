"""Optional local reranking adapters."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

_MAX_LENGTH = 512
_BATCH_SIZE = 4
_DEVICES = {"auto", "cuda", "cpu"}


class BGEReranker:
    """Lazy FP32 adapter with bounded CUDA batches and deterministic CPU fallback."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
    ) -> None:
        requested = (device or os.getenv("RAGLAB_RERANK_DEVICE") or "auto").casefold()
        if requested not in _DEVICES:
            raise ValueError("RAGLAB_RERANK_DEVICE must be auto, cuda, or cpu")
        self.model = model
        self.requested_device = requested
        self.device = "uninitialized"
        self._tokenizer: Any = None
        self._model: Any = None

    def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install `raglab[retrieval]` to use the BGE reranker") from exc
        self._ensure_loaded(torch)
        try:
            return self._infer(torch, query, documents)
        except Exception as exc:
            if self.device != "cuda" or not self._is_oom(torch, exc):
                raise
            self._fallback_to_cpu(torch)
            return self._infer(torch, query, documents)

    def _ensure_loaded(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install `raglab[retrieval]` to use the BGE reranker") from exc
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self.model
            )
        target = self._target_device(torch)
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            self._prepare_model(target)
        except Exception:
            if target != "cuda":
                raise
            self._model = None
            self._empty_cuda_cache(torch)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            self._prepare_model("cpu")

    def _prepare_model(self, device: str) -> None:
        if hasattr(self._model, "float"):
            self._model.float()
        if hasattr(self._model, "to"):
            self._model.to(device)
        self._model.eval()
        self.device = device

    def _infer(self, torch: Any, query: str, documents: Sequence[str]) -> list[float]:
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
                if self.device == "cuda":
                    inputs = {
                        key: value.to("cuda") if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                outputs = self._model(**inputs)
                scores.extend(
                    float(value) for value in outputs.logits.view(-1).detach().cpu().tolist()
                )
        return scores

    def _fallback_to_cpu(self, torch: Any) -> None:
        try:
            self._prepare_model("cpu")
        except Exception:
            from transformers import AutoModelForSequenceClassification

            self._model = None
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            self._prepare_model("cpu")
        self._empty_cuda_cache(torch)

    def _target_device(self, torch: Any) -> str:
        if self.requested_device == "cpu":
            return "cpu"
        cuda = getattr(torch, "cuda", None)
        available = bool(cuda is not None and cuda.is_available())
        return "cuda" if available else "cpu"

    @staticmethod
    def _empty_cuda_cache(torch: Any) -> None:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()

    @staticmethod
    def _is_oom(torch: Any, error: Exception) -> bool:
        cuda = getattr(torch, "cuda", None)
        oom_type = getattr(cuda, "OutOfMemoryError", ()) if cuda is not None else ()
        return (bool(oom_type) and isinstance(error, oom_type)) or "out of memory" in str(
            error
        ).casefold()
