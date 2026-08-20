from __future__ import annotations

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
        "EvaluationApplication",
        "HermeticEvaluationExecutor",
        ".promote(",
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
    assert not any(
        boundary in code
        for boundary in (
            "OllamaEmbeddingProvider",
            "OllamaGenerationModel",
            "PostgresRepository",
            "PostgresRetrievalRepository",
            "LiveEvaluationExecutor",
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
    assert sum(len(str(cell.source)) for cell in code_cells) <= 5_000
    assert len(main_markdown) <= 3_500

    for variable in (
        "RAGLAB_RUN_RETRIEVAL_NOTEBOOK",
        "RAGLAB_RUN_GENERATION_NOTEBOOK",
        "RAGLAB_RUN_EVALUATION_NOTEBOOK",
        "RAGLAB_RUN_TENANT_DEMO",
        "RAGLAB_PDF",
    ):
        monkeypatch.delenv(variable, raising=False)

    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=repository)
