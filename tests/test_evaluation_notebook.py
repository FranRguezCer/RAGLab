from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_evaluation_notebook_is_output_free_and_executes_hermetically(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "notebooks/04_rag_evaluation.ipynb"
    notebook = nbformat.read(source, as_version=4)

    code = "\n".join(
        str(cell.source) for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "EvaluationApplication" in code
    assert "subprocess" not in code
    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )

    previous = os.environ.pop("RAGLAB_RUN_EVALUATION_NOTEBOOK", None)
    try:
        executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(
            cwd=repository
        )
    finally:
        if previous is not None:
            os.environ["RAGLAB_RUN_EVALUATION_NOTEBOOK"] = previous
    output = tmp_path / "04_rag_evaluation.executed.ipynb"
    nbformat.write(executed, output)
    assert output.exists()
