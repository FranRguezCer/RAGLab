from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from raglab import Citation, ProvenanceStatus
from raglab.errors import GenerationContractError, GenerationError
from raglab.generation import (
    GenerationConfig,
    GenerationPipeline,
    GenerationRequest,
    ModelInvocation,
)
from raglab.generation.pipeline import _Fact
from raglab.retrieval import (
    CollectionMetadata,
    RankingTrace,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)


def _result(index: int, content: str) -> RetrievalResult:
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
    def __init__(self, contents: Sequence[str], *, model: str = "embed") -> None:
        self.response = RetrievalResponse(
            "question",
            None,
            ("question",),
            (),
            tuple(_result(index, content) for index, content in enumerate(contents, 1)),
        )
        self.model = model
        self.requests: list[RetrievalRequest] = []

    def collection_metadata(self, collection: str) -> CollectionMetadata:
        return CollectionMetadata(collection, self.model, 1024)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        return self.response


class Verifier:
    def __init__(self, rejected_claims: set[str] | None = None) -> None:
        self.rejected_claims = rejected_claims or set()
        self.pairs: list[tuple[str, str]] = []

    def verify(self, pairs: Sequence[tuple[str, str]]) -> tuple[bool, ...]:
        self.pairs.extend(pairs)
        return tuple(claim not in self.rejected_claims for _, claim in pairs)


class ScenarioModel:
    def __init__(
        self,
        facts_by_content: dict[str, list[dict[str, str]]],
        *,
        units: list[dict[str, str]] | None = None,
        unused: list[str] | None = None,
    ) -> None:
        self.facts_by_content = facts_by_content
        self.units = units
        self.unused = unused
        self.calls = 0
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
        self.calls += 1
        self.schemas.append(schema)
        if "analyze exactly one" in system:
            matches = [
                facts
                for content, facts in self.facts_by_content.items()
                if content in prompt
            ]
            assert len(matches) == 1
            return ModelInvocation({"facts": matches[0]}, 10, 5)
        encoded = prompt.split("Verified relevant facts:\n", 1)[1].split(
            "\n\nAnswer only this request:", 1
        )[0]
        rows = json.loads(encoded)
        ids = [row["fact_id"] for row in rows]
        units = self.units or [{"fact_id": ids[0], "text": rows[0]["claim"]}]
        unused = (
            self.unused
            if self.unused is not None
            else [item for item in ids if item != units[0]["fact_id"]]
        )
        return ModelInvocation({"units": units, "unused_fact_ids": unused}, 10, 5)


def _fact(claim: str, quote: str) -> dict[str, str]:
    return {"claim": claim, "evidence_quote": quote}


def _pipeline(
    contents: list[str],
    facts: dict[str, list[dict[str, str]]],
    *,
    verifier: Verifier | None = None,
    units: list[dict[str, str]] | None = None,
    unused: list[str] | None = None,
) -> tuple[GenerationPipeline, ScenarioModel, Verifier]:
    grounding = verifier or Verifier()
    model = ScenarioModel(facts, units=units, unused=unused)
    pipeline = GenerationPipeline(
        Retrieval(contents),
        model,
        grounding_verifier=grounding,
        embedding_model="embed",
    )
    return pipeline, model, grounding


def test_invented_password_policy_with_irrelevant_exact_quote_is_rejected() -> None:
    quote = "Passwords are hashed with Argon2 before storage."
    claim = "Passwords must be rotated every 90 days."
    pipeline, model, _ = _pipeline(
        [quote], {quote: [_fact(claim, quote)]}, verifier=Verifier({claim})
    )

    response = pipeline.generate(GenerationRequest(RetrievalRequest("password policy")))

    assert response.abstained is True
    assert response.source_ids == ()
    assert response.metrics.facts_extracted == 1
    assert response.metrics.facts_rejected == 1
    assert model.calls == 1


def test_unanswerable_question_abstains_without_citations() -> None:
    contents = ["Unrelated deployment note.", "Unrelated metrics note."]
    pipeline, _, _ = _pipeline(contents, {content: [] for content in contents})

    response = pipeline.generate(GenerationRequest(RetrievalRequest("What is the password?")))

    assert response.abstained is True
    assert response.sources == ()
    assert response.metrics.facts_accepted == 0


def test_truthful_but_question_irrelevant_claims_are_not_accepted() -> None:
    source = "Every cache key contains the tenant ID."
    pipeline, _, _ = _pipeline(
        [source], {source: [_fact(source, source)]}
    )

    response = pipeline.generate(
        GenerationRequest(RetrievalRequest("How often must employees rotate passwords?"))
    )

    assert response.abstained is True
    assert response.source_ids == ()
    assert response.metrics.facts_accepted == 0


def test_noisy_source_is_discarded_while_valid_source_answers() -> None:
    noisy = "Passwords are hashed with Argon2 before storage."
    valid = "Use SKIPPED_WET_BED when the bed is wet."
    false_claim = "Rotate passwords every 90 days."
    true_claim = "Use SKIPPED_WET_BED when the bed is wet."
    pipeline, _, _ = _pipeline(
        [noisy, valid],
        {noisy: [_fact(false_claim, noisy)], valid: [_fact(true_claim, valid)]},
        verifier=Verifier({false_claim}),
    )

    response = pipeline.generate(
        GenerationRequest(RetrievalRequest("What action applies when the bed is wet?"))
    )

    assert response.answer == true_claim
    assert response.source_ids == ("S2",)
    assert response.metrics.facts_rejected == 1
    assert response.metrics.facts_used == 1


def test_e41_action_beats_diagnosis_distractor() -> None:
    distractor = "The diagnostic code is D14."
    action = "Raise E41 when pressure exceeds the shutdown threshold."
    pipeline, _, _ = _pipeline(
        [distractor, action],
        {distractor: [], action: [_fact(action, action)]},
    )

    response = pipeline.generate(
        GenerationRequest(RetrievalRequest("What E41 action is required for high pressure?"))
    )

    assert response.answer == action
    assert response.source_ids == ("S2",)


def test_multipart_answer_uses_only_user_id_and_raw_request_path_facts() -> None:
    user = "Remove user ID from metric labels."
    path = "Remove raw request path from metric labels."
    region = "Region is an approved metric label."
    units = [
        {"fact_id": "F1", "text": user},
        {"fact_id": "F2", "text": path},
    ]
    pipeline, _, _ = _pipeline(
        [user, path, region],
        {user: [_fact(user, user)], path: [_fact(path, path)], region: [_fact(region, region)]},
        units=units,
        unused=["F3"],
    )

    response = pipeline.generate(GenerationRequest(RetrievalRequest("Which labels are removed?")))

    assert "user ID" in response.answer and "raw request path" in response.answer
    assert "Region" not in response.answer
    assert response.source_ids == ("S1", "S2")
    assert response.metrics.facts_used == 2


@pytest.mark.parametrize(
    ("units", "unused"),
    [
        ([{"fact_id": "F1", "text": "Alpha."}], []),
        ([{"fact_id": "F1", "text": "Alpha."}], ["F1", "F2"]),
        ([{"fact_id": "F9", "text": "Alpha."}], ["F1", "F2"]),
    ],
)
def test_synthesis_requires_exact_duplicate_free_fact_partition(
    units: list[dict[str, str]], unused: list[str]
) -> None:
    alpha, beta = "Alpha.", "Beta."
    pipeline, _, _ = _pipeline(
        [alpha, beta],
        {alpha: [_fact(alpha, alpha)], beta: [_fact(beta, beta)]},
        units=units,
        unused=unused,
    )

    with pytest.raises(GenerationContractError, match="partition"):
            pipeline.generate(GenerationRequest(RetrievalRequest("Alpha Beta")))


def test_unsupported_final_unit_fails_without_retry() -> None:
    quote = "The supported action is E41."
    claim = "Use E41."
    unsupported = "Use E17."
    verifier = Verifier({unsupported})
    pipeline, model, _ = _pipeline(
        [quote],
        {quote: [_fact(claim, quote)]},
        verifier=verifier,
        units=[{"fact_id": "F1", "text": unsupported}],
        unused=[],
    )

    with pytest.raises(GenerationContractError, match="unsupported"):
        pipeline.generate(GenerationRequest(RetrievalRequest("Which E41 action applies?")))

    assert model.calls == 2


def test_noncontiguous_quote_is_rejected_without_failing_other_sources() -> None:
    source = "Alpha appears before Beta."
    pipeline, _, _ = _pipeline(
        [source], {source: [_fact("Alpha and Beta appear.", "Alpha Beta")]}
    )

    response = pipeline.generate(GenerationRequest(RetrievalRequest("question")))

    assert response.abstained is True
    assert response.metrics.facts_extracted == 1
    assert response.metrics.facts_rejected == 1


def test_missing_nli_checkpoint_is_actionable() -> None:
    class Missing:
        def verify(self, pairs: Sequence[tuple[str, str]]) -> tuple[bool, ...]:
            del pairs
            raise GenerationError(
                "Pinned NLI model is unavailable locally; install raglab[generation]"
            )

    quote = "Grounded fact."
    pipeline = GenerationPipeline(
        Retrieval([quote]),
        ScenarioModel({quote: [_fact(quote, quote)]}),
        grounding_verifier=Missing(),
        embedding_model="embed",
    )

    with pytest.raises(GenerationError, match=r"raglab\[generation\]"):
        pipeline.generate(GenerationRequest(RetrievalRequest("question")))


def test_incomplete_reduction_fails_without_retry() -> None:
    class ReductionModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *args: object, **kwargs: object) -> ModelInvocation:
            self.calls += 1
            return ModelInvocation({"summary": "Alpha."})

    model = ReductionModel()
    pipeline = GenerationPipeline(
        Retrieval([]),
        model,
        grounding_verifier=Verifier({"Beta."}),
        embedding_model="embed",
    )
    facts = [
        _Fact("F1", "Alpha.", ("Alpha.",), ("S1",), ("Alpha.",)),
        _Fact("F2", "Beta.", ("Beta.",), ("S2",), ("Beta.",)),
    ]

    with pytest.raises(GenerationContractError, match="preserve every"):
        pipeline._reduce_group(
            "question", facts, GenerationRequest(RetrievalRequest("question")),
        )

    assert model.calls == 1


def test_empty_retrieval_abstains_without_model_or_nli_call() -> None:
    model = ScenarioModel({})
    verifier = Verifier()
    response = GenerationPipeline(
        Retrieval([]), model, grounding_verifier=verifier, embedding_model="embed"
    ).generate(GenerationRequest(RetrievalRequest("question")))

    assert response.abstained is True
    assert model.calls == 0
    assert verifier.pairs == []
