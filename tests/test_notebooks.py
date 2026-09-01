from __future__ import annotations

import os
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = {
    "01_ingestion_and_indexing.ipynb": (
        "Converter",
        "MarkdownParser",
        "SemanticChunker",
    ),
    "02_retrieval.ipynb": (
        "reciprocal_rank_fusion",
        "RetrievalResult",
    ),
    "03_generation.ipynb": (
        "GenerationPipeline",
        "GenerationError",
    ),
    "04_rag_evaluation.ipynb": (
        "load_cases",
        "evaluate_retrieval",
        "evaluate_generation",
        "create_live_evaluation_application",
    ),
}

HUMAN_OUTPUT_MARKERS = {
    "01_ingestion_and_indexing.ipynb": (
        "INPUT DOCUMENT",
        "CONVERTED CONTENT",
        "PARSED CONTENT",
        "CHUNKS",
        "INDEXING RESULT",
        "Evidence:",
    ),
    "02_retrieval.ipynb": (
        "QUERY",
        "SEMANTIC RESULTS",
        "BM25 RESULTS",
        "FUSED RETRIEVAL RESULTS",
        "FINAL RETRIEVAL EVIDENCE",
        "RANKING EXPLANATION",
    ),
    "03_generation.ipynb": (
        "QUERY",
        "RETRIEVAL EVIDENCE",
        "SELECTED EVIDENCE",
        "REJECTED FACTS",
        "GENERATED ANSWER",
        "CITATIONS",
        "STAGE DIAGNOSIS",
    ),
    "04_rag_evaluation.ipynb": (
        "QUERY",
        "KNOWN WRONG CLAIMS",
        "CONTROLLED RETRIEVAL RESULTS",
        "GENERATED ANSWER",
        "STAGE DIAGNOSIS",
        "LIVE RETRIEVAL RESULTS",
    ),
}


@pytest.mark.parametrize(("filename", "api_markers"), NOTEBOOKS.items())
def test_short_notebook_lab_is_valid_output_free_and_hermetic(
    filename: str,
    api_markers: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "notebooks" / filename
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    code = "\n".join(str(cell.source) for cell in code_cells)
    assert all(marker in code for marker in api_markers)
    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in code_cells
    )
    main_cells = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and "## Optional appendix" in str(cell.source):
            break
        if cell.cell_type == "code":
            main_cells.append(cell)
    main_code = "\n".join(str(cell.source) for cell in main_cells)
    assert not any(
        boundary in main_code
        for boundary in (
            "OllamaEmbeddingProvider",
            "OllamaGenerationModel",
            "PostgresRepository",
            "PostgresRetrievalRepository",
            "psycopg",
            "subprocess",
        )
    )

    markdown = "\n".join(
        str(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    main_markdown = markdown.split("## Optional appendix", maxsplit=1)[0]
    checkpoint_numbers = re.findall(
        r"^## Checkpoint ([1-3]) — Objective:", main_markdown, flags=re.MULTILINE
    )
    assert checkpoint_numbers == ["1", "2", "3"]
    checkpoint_sections = re.split(
        r"(?=^## Checkpoint [1-3] — Objective:)", main_markdown, flags=re.MULTILINE
    )[1:]
    assert len(checkpoint_sections) == 3
    for section in checkpoint_sections:
        assert section.count("**Run:**") == 1
        assert section.count("### What to observe") == 1
        assert section.count("### Conclusion") == 1

    assert len(notebook.cells) <= 13
    code_limit = {
        "03_generation.ipynb": 6_000,
        "04_rag_evaluation.ipynb": 9_500,
    }.get(filename, 5_000)
    assert sum(len(str(cell.source)) for cell in code_cells) <= code_limit
    assert len(main_markdown) <= 3_500

    if filename == "04_rag_evaluation.ipynb":
        activation = 'os.environ["RAGLAB_RUN_EVALUATION_NOTEBOOK"] = "1"'
        live_cell_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if "create_live_evaluation_application" in str(cell.source)
        )
        assert activation in code
        assert activation in str(notebook.cells[live_cell_index - 1].source)
        notebook.cells[live_cell_index - 1].source = str(
            notebook.cells[live_cell_index - 1].source
        ).replace(activation, 'os.environ["RAGLAB_RUN_EVALUATION_NOTEBOOK"] = "0"')

    for variable in (
        "RAGLAB_RUN_RETRIEVAL_NOTEBOOK",
        "RAGLAB_RUN_GENERATION_NOTEBOOK",
        "RAGLAB_RUN_EVALUATION_NOTEBOOK",
        "RAGLAB_RUN_TENANT_DEMO",
        "RAGLAB_PDF",
    ):
        monkeypatch.setenv(variable, "0")

    execution_cwd = source.parent if filename == "04_rag_evaluation.ipynb" else repository
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(
        cwd=execution_cwd,
        env=dict(os.environ),
    )
    output = "\n".join(
        str(item.get("text", ""))
        for cell in executed.cells
        for item in cell.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    for marker in HUMAN_OUTPUT_MARKERS[filename]:
        assert marker in code
        if not (filename == "04_rag_evaluation.ipynb" and marker.startswith("LIVE ")):
            assert marker in output

    if filename == "03_generation.ipynb":
        assert "Generation: ANSWERED" in output
        assert "Generation: ABSTAINED" in output
        assert "Retrieval: EVIDENCE FOUND, BUT IRRELEVANT TO THE QUERY" in output
        assert "None — rejected facts cannot produce citations." in output
    elif filename == "04_rag_evaluation.ipynb":
        assert "OllamaEmbeddingProvider" not in code
        assert "OllamaGenerationModel" not in code
        assert "PostgresRetrievalRepository" not in code
        assert "SELECTED EVIDENCE" in code
        assert "facts_invalid_quotes" in code
        assert "facts_nli_rejected" in code
        assert "QUERY" in output
        assert "KNOWN WRONG CLAIMS" in output
        assert "CONTROLLED RETRIEVAL RESULTS" in output
        assert "STAGE DIAGNOSIS" in output
        assert "Retrieval: UNCHANGED" in output
        assert "Generation: REGRESSED" in output
