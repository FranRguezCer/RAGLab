from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from raglab import Citation, ProvenanceStatus
from raglab.errors import GenerationContractError, GenerationError, GenerationLengthError
from raglab.generation import (
    GenerationConfig,
    GenerationPipeline,
    GenerationRequest,
    GenerationStrategy,
    ModelInvocation,
)
from raglab.retrieval import (
    CollectionMetadata,
    RankingTrace,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)


def _result(index: int, content: str = "grounded evidence") -> RetrievalResult:
    return RetrievalResult(
        id=f"result-{index}",
        document_id=f"document-{index}",
        content=content,
        citation=Citation(
            source_uri=f"file:///source-{index}.md",
            source_name=f"source-{index}.md",
            title=f"Source {index}",
            heading_path=("Evidence",),
            start_page=None,
            end_page=None,
            start_line=1,
            end_line=2,
            provenance_status=ProvenanceStatus.COMPLETE,
        ),
        matched_chunk_ids=(f"chunk-{index}",),
        first_chunk_index=0,
        last_chunk_index=0,
        trace=RankingTrace(1, 1, 0.1, 1.0, 0.5, 0.8, None),
    )


class Retrieval:
    def __init__(
        self,
        results: Sequence[RetrievalResult],
        *,
        model: str = "embed",
        rewritten_query: str | None = None,
    ) -> None:
        self.response = RetrievalResponse(
            "question", rewritten_query, ("question",), (), tuple(results)
        )
        self.model = model
        self.requests: list[RetrievalRequest] = []

    def collection_metadata(self, collection: str) -> CollectionMetadata:
        return CollectionMetadata(collection, self.model, 1024)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        return self.response


class ScenarioModel:
    def __init__(self, facts_by_content: dict[str, list[str]], answer: str) -> None:
        self.facts_by_content = facts_by_content
        self.answer = answer
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.schemas: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, Any],
        config: GenerationConfig,
    ) -> ModelInvocation:
        del config
        self.prompts.append(prompt)
        self.systems.append(system)
        self.schemas.append(schema)
        if "analyze exactly one" in system:
            matches = [
                facts for content, facts in self.facts_by_content.items() if content in prompt
            ]
            assert len(matches) == 1
            return ModelInvocation({"facts": matches[0]}, 10, 5)
        assert "using only the supplied grounded facts" in system
        return ModelInvocation({"answer": self.answer}, 10, 5)


def _run_answered_case(
    contents: list[str], facts: dict[str, list[str]], answer: str
) -> tuple[Retrieval, ScenarioModel, Any]:
    retrieval = Retrieval([_result(index, content) for index, content in enumerate(contents, 1)])
    model = ScenarioModel(facts, answer)
    response = GenerationPipeline(retrieval, model, embedding_model="embed").generate(
        GenerationRequest(RetrievalRequest("What should we do?"))
    )
    return retrieval, model, response


def test_wet_bed_answer_includes_skipped_wet_bed_fact() -> None:
    contents = ["rank one", "wet bed", "rank three", "rank four", "rank five"]
    facts = {
        "rank one": [],
        "wet bed": ["A wet bed maps to SKIPPED_WET_BED."],
        "rank three": [],
        "rank four": [],
        "rank five": [],
    }

    _, model, response = _run_answered_case(
        contents, facts, "Use SKIPPED_WET_BED when the bed is wet."
    )

    assert "SKIPPED_WET_BED" in response.answer
    assert response.source_ids == ("S2",)
    assert response.metrics.model_calls == 6
    assert len(model.systems) == 6


def test_rollback_answer_covers_both_actions_for_harbor_and_meridian() -> None:
    contents = ["harbor stop", "meridian stop", "harbor revert", "meridian revert", "noise"]
    facts = {
        "harbor stop": ["Pause the Harbor rollout."],
        "meridian stop": ["Pause the Meridian rollout."],
        "harbor revert": ["Roll back Harbor."],
        "meridian revert": ["Roll back Meridian."],
        "noise": [],
    }
    answer = "Pause and roll back both Harbor and Meridian."

    _, model, response = _run_answered_case(contents, facts, answer)

    assert all(term in response.answer for term in ("Pause", "roll back", "Harbor", "Meridian"))
    assert response.source_ids == ("S1", "S2", "S3", "S4")
    assert response.metrics.model_calls == 6
    assert len(model.prompts) == 6


def test_canary_immediate_rollback_names_harbor_not_aster() -> None:
    contents = ["baseline", "aster context", "harbor alert", "monitoring", "appendix"]
    facts = {
        "baseline": [],
        "aster context": [],
        "harbor alert": ["Immediately roll back Harbor when the canary alert fires."],
        "monitoring": [],
        "appendix": [],
    }

    _, model, response = _run_answered_case(contents, facts, "Immediately roll back Harbor.")

    assert "Immediately" in response.answer and "Harbor" in response.answer
    assert "Aster" not in response.answer
    assert response.source_ids == ("S3",)
    assert response.metrics.model_calls == 6
    assert len(model.prompts) == 6


def test_metrics_answer_removes_user_id_and_raw_request_path_without_invention() -> None:
    contents = ["user field", "request field", "noise one", "noise two", "noise three"]
    facts = {
        "user field": ["Remove the user ID from metrics."],
        "request field": ["Remove the raw request path from metrics."],
        "noise one": [],
        "noise two": [],
        "noise three": [],
    }

    _, model, response = _run_answered_case(
        contents, facts, "Remove the user ID and raw request path from metrics."
    )

    assert response.answer == "Remove the user ID and raw request path from metrics."
    assert response.source_ids == ("S1", "S2")
    assert response.metrics.model_calls == 6
    assert len(model.prompts) == 6


def test_unsupported_password_question_abstains_without_synthesis() -> None:
    contents = [f"unrelated {index}" for index in range(1, 6)]
    facts = {content: [] for content in contents}
    retrieval = Retrieval([_result(index, content) for index, content in enumerate(contents, 1)])
    model = ScenarioModel(facts, "must not be called")

    response = GenerationPipeline(retrieval, model, embedding_model="embed").generate(
        GenerationRequest(RetrievalRequest("What is the password?"))
    )

    assert response.answer == (
        "I cannot answer because the retrieved evidence contains no relevant facts."
    )
    assert response.abstained is True
    assert response.source_ids == ()
    assert response.sources == ()
    assert response.metrics.model_calls == 5
    assert len(model.prompts) == 5


def test_original_history_and_question_are_authoritative_and_retrieval_is_unchanged() -> None:
    contents = [f"source {index}" for index in range(1, 6)]
    facts = {
        content: ["Relevant fact."] if index == 1 else []
        for index, content in enumerate(contents, 1)
    }
    retrieval = Retrieval(
        [_result(index, content) for index, content in enumerate(contents, 1)],
        rewritten_query="WRONG rewritten retrieval-only query",
    )
    request = RetrievalRequest(
        "What action applies now?", history=("Earlier we discussed Harbor.",)
    )
    original_results = retrieval.response.results
    model = ScenarioModel(facts, "Apply the relevant action.")

    response = GenerationPipeline(retrieval, model, embedding_model="embed").generate(
        GenerationRequest(request)
    )

    assert retrieval.requests == [request]
    assert response.retrieval is retrieval.response
    assert response.retrieval.results is original_results
    assert all("What action applies now?" in prompt for prompt in model.prompts)
    assert all("Earlier we discussed Harbor." in prompt for prompt in model.prompts)
    assert all("WRONG rewritten retrieval-only query" not in prompt for prompt in model.prompts)


def test_duplicate_facts_preserve_ordered_source_lineage() -> None:
    contents = ["first", "second", "third", "fourth", "fifth"]
    facts = {
        "first": ["Shared fact."],
        "second": ["Unique fact."],
        "third": ["Shared fact."],
        "fourth": [],
        "fifth": [],
    }

    _, model, response = _run_answered_case(contents, facts, "Shared and unique facts apply.")

    synthesis_prompt = model.prompts[-1]
    assert synthesis_prompt.count("Shared fact.") == 1
    assert response.source_ids == ("S1", "S3", "S2")
    assert [source.id for source in response.sources] == ["S1", "S3", "S2"]


def test_analysis_schema_has_no_model_controlled_source_ids() -> None:
    contents = [f"source {index}" for index in range(1, 6)]
    facts = {content: [f"Fact {index}."] for index, content in enumerate(contents, 1)}
    _, model, response = _run_answered_case(contents, facts, "All facts apply.")

    assert response.source_ids == ("S1", "S2", "S3", "S4", "S5")
    for schema in model.schemas[:5]:
        assert schema["required"] == ["facts"]
        assert "source_ids" not in json.dumps(schema)
    assert model.schemas[-1]["required"] == ["answer"]
    assert "source_ids" not in json.dumps(model.schemas[-1])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"facts": "not a list"},
        {"facts": [""]},
        {"facts": ["valid"], "source_ids": ["S1"]},
    ],
)
def test_invalid_source_analysis_fails_without_contract_retry(payload: dict[str, object]) -> None:
    class InvalidModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *args: object, **kwargs: object) -> ModelInvocation:
            self.calls += 1
            return ModelInvocation(payload)

    model = InvalidModel()
    pipeline = GenerationPipeline(Retrieval([_result(1)]), model, embedding_model="embed")

    with pytest.raises(GenerationContractError, match="Source analysis") as raised:
        pipeline.generate(GenerationRequest(RetrievalRequest("question")))

    assert model.calls == 1
    assert raised.value.model_calls == 1


def test_invalid_synthesis_fails_without_contract_retry() -> None:
    class InvalidSynthesis:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self, prompt: str, *, system: str, schema: dict[str, Any], config: GenerationConfig
        ) -> ModelInvocation:
            del prompt, schema, config
            self.calls += 1
            if "analyze exactly one" in system:
                return ModelInvocation({"facts": ["Grounded."]})
            return ModelInvocation({"answer": "Grounded.", "source_ids": ["S99"]})

    model = InvalidSynthesis()
    pipeline = GenerationPipeline(
        Retrieval([_result(index) for index in range(1, 6)]), model, embedding_model="embed"
    )

    with pytest.raises(GenerationContractError, match="Synthesis") as raised:
        pipeline.generate(GenerationRequest(RetrievalRequest("question")))

    assert model.calls == 6
    assert raised.value.model_calls == 6


def test_oversized_source_fails_explicitly_without_model_call_or_truncation() -> None:
    class NeverCalled:
        def generate(self, *args: object, **kwargs: object) -> ModelInvocation:
            raise AssertionError("model must not be called")

    content = "oversized evidence " * 500
    pipeline = GenerationPipeline(
        Retrieval([_result(1, content)]), NeverCalled(), embedding_model="embed"
    )

    with pytest.raises(GenerationError, match="without truncation"):
        pipeline.generate(
            GenerationRequest(
                RetrievalRequest("question"),
                GenerationConfig(num_ctx=1800, num_predict=128),
            )
        )


def test_synthesis_length_reduction_uses_smaller_inputs_and_preserves_lineage() -> None:
    class LengthThenReduction:
        def __init__(self) -> None:
            self.synthesis_calls = 0
            self.reduction_input_sizes: list[int] = []

        def generate(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, Any],
            config: GenerationConfig,
        ) -> ModelInvocation:
            del schema
            if "analyze exactly one" in system:
                alias = prompt.split("[", 1)[1].split("]", 1)[0]
                return ModelInvocation({"facts": [f"{alias}: " + "detail " * 40]})
            if "compress the supplied grounded facts" in system:
                facts = json.loads(prompt.rsplit("\n", 1)[1])
                self.reduction_input_sizes.append(len(facts))
                return ModelInvocation({"summary": f"Summary of {len(facts)} facts."})
            self.synthesis_calls += 1
            if self.synthesis_calls == 1:
                raise GenerationLengthError(
                    "length", prompt_tokens=100, generated_tokens=config.num_predict
                )
            return ModelInvocation({"answer": "The five grounded facts apply."})

    model = LengthThenReduction()
    response = GenerationPipeline(
        Retrieval([_result(index) for index in range(1, 6)]),
        model,
        embedding_model="embed",
    ).generate(GenerationRequest(RetrievalRequest("question")))

    assert model.reduction_input_sizes
    assert all(size < 5 for size in model.reduction_input_sizes)
    assert response.source_ids == ("S1", "S2", "S3", "S4", "S5")
    assert [source.id for source in response.sources] == ["S1", "S2", "S3", "S4", "S5"]


def test_length_terminated_source_analysis_is_not_retried() -> None:
    class LengthModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *args: object, **kwargs: object) -> ModelInvocation:
            self.calls += 1
            raise GenerationLengthError("length", prompt_tokens=100, generated_tokens=128)

    model = LengthModel()
    pipeline = GenerationPipeline(Retrieval([_result(1)]), model, embedding_model="embed")

    with pytest.raises(GenerationError, match="not truncated or retried"):
        pipeline.generate(GenerationRequest(RetrievalRequest("question")))

    assert model.calls == 1


def test_empty_retrieval_abstains_without_calling_model() -> None:
    model = ScenarioModel({}, "must not be called")
    retrieval = Retrieval([])

    response = GenerationPipeline(retrieval, model, embedding_model="embed").generate(
        GenerationRequest(RetrievalRequest("question"))
    )

    assert response.abstained is True
    assert response.strategy is GenerationStrategy.SINGLE_PASS
    assert response.metrics.model_calls == 0
    assert model.prompts == []


def test_source_shortfall_still_analyzes_every_available_source() -> None:
    contents = ["alpha evidence", "beta evidence"]
    facts = {"alpha evidence": ["First."], "beta evidence": ["Second."]}
    _, model, response = _run_answered_case(contents, facts, "Both apply.")

    assert response.source_shortfall is True
    assert response.source_count == 2
    assert response.metrics.model_calls == 3
    assert len(model.prompts) == 3


def test_embedding_model_must_match_collection_before_retrieval() -> None:
    retrieval = Retrieval([_result(1)], model="indexed-model")
    pipeline = GenerationPipeline(retrieval, ScenarioModel({}, ""), embedding_model="other")

    with pytest.raises(GenerationError, match="Reindex"):
        pipeline.generate(GenerationRequest(RetrievalRequest("question")))

    assert retrieval.requests == []
