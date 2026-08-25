from __future__ import annotations

import pytest

from raglab.retrieval.rewriting import OllamaQueryRewriter


def test_rewriter_parses_fenced_json_and_limits_expansions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewriter = OllamaQueryRewriter()
    bodies: list[dict[str, object]] = []
    monkeypatch.setattr(
        rewriter,
        "_request",
        lambda body: bodies.append(body) or {
            "response": "```json\n"
            '{"standalone_query":"Fault E17 cause",'
            '"expansions":["E17 root cause","error E17","ignored"]}\n```'
        },
    )

    result = rewriter.rewrite("What causes it?", ["Fault E17"], max_expansions=2)

    assert result.standalone_query == "Fault E17 cause"
    assert result.expansions == ("E17 root cause", "error E17")
    assert bodies[0]["think"] is False
    assert bodies[0]["options"] == {"temperature": 0}
    assert "Resolve every pronoun" in str(bodies[0]["prompt"])


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"expansions": []}',
        '{"standalone_query": "ok", "expansions": [1]}',
    ],
)
def test_rewriter_rejects_invalid_model_output(
    response: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rewriter = OllamaQueryRewriter()
    monkeypatch.setattr(rewriter, "_request", lambda _body: {"response": response})

    with pytest.raises(RuntimeError):
        rewriter.rewrite("query", (), max_expansions=2)
