from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from raglab.retrieval.reranking import BGEReranker


class _Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def view(self, *_shape: int) -> _Tensor:
        return self

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> list[float]:
        return self.values


def test_reranker_uses_bounded_inference_batches_and_reuses_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "depth": 0,
        "inference_calls": 0,
        "model_loads": [],
        "tokenizer_loads": [],
        "tokenizer_calls": [],
        "model_batches": [],
        "eval_calls": 0,
    }

    class InferenceMode:
        def __enter__(self) -> None:
            state["depth"] += 1

        def __exit__(self, *_args: object) -> None:
            state["depth"] -= 1

    def inference_mode() -> InferenceMode:
        state["inference_calls"] += 1
        return InferenceMode()

    class Tokenizer:
        def __call__(self, pairs: list[list[str]], **kwargs: object) -> dict[str, object]:
            state["tokenizer_calls"].append((pairs, kwargs))
            return {"documents": [pair[1] for pair in pairs]}

    class Model:
        def eval(self) -> None:
            state["eval_calls"] += 1

        def __call__(self, *, documents: list[str]) -> SimpleNamespace:
            assert state["depth"] == 1
            state["model_batches"].append(documents)
            return SimpleNamespace(logits=_Tensor([float(document[3:]) for document in documents]))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str) -> Tokenizer:
            state["tokenizer_loads"].append(model)
            return Tokenizer()

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model: str) -> Model:
            state["model_loads"].append(model)
            return Model()

    torch = ModuleType("torch")
    torch.inference_mode = inference_mode  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForSequenceClassification = (  # type: ignore[attr-defined]
        AutoModelForSequenceClassification
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    reranker = BGEReranker("test-model")

    assert reranker.rerank("query", [f"doc{index}" for index in range(9)]) == list(
        map(float, range(9))
    )
    assert reranker.rerank("query", ["doc9"]) == [9.0]
    assert [len(pairs) for pairs, _ in state["tokenizer_calls"]] == [4, 4, 1, 1]
    assert all(
        kwargs
        == {
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt",
        }
        for _, kwargs in state["tokenizer_calls"]
    )
    assert state["model_batches"] == [
        ["doc0", "doc1", "doc2", "doc3"],
        ["doc4", "doc5", "doc6", "doc7"],
        ["doc8"],
        ["doc9"],
    ]
    assert state["inference_calls"] == 2
    assert state["tokenizer_loads"] == ["test-model"]
    assert state["model_loads"] == ["test-model"]
    assert state["eval_calls"] == 1


def test_reranker_does_not_load_model_for_empty_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    assert BGEReranker().rerank("query", []) == []
