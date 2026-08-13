# RAGLab

RAGLab is a local-first, inspectable ingestion pipeline for learning the foundations of
retrieval-augmented generation. It converts sources to canonical Markdown, parses their
structure, refines oversized sections with semantic boundaries, creates Ollama embeddings,
and stores them in PostgreSQL with pgvector. It deliberately does not use LangChain or
LlamaIndex.

## Architecture

```text
SourceInput
  -> conversion (direct / local Docling / local URL / opt-in Jina)
  -> Markdown AST (markdown-it-py)
  -> structural units -> semantic boundary refinement
  -> Ollama qwen3-embedding:0.6b (1024 dimensions)
  -> one atomic PostgreSQL document replacement
```

`content` remains faithful to the converted source. Title and heading breadcrumbs are added
only to `embedding_text`. All embeddings are produced before the database transaction begins.
The `(source URI, content hash, pipeline fingerprint)` tuple makes repeated ingestion a no-op;
content or configuration changes replace the document's chunks atomically.

## Setup

Requirements: Python 3.12, Docker, and Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,conversion,tokenizers]'
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
docker compose up -d
ollama pull qwen3-embedding:0.6b
```

Docling and the Qwen tokenizer are optional, heavy dependencies. Markdown/text tests and all
module imports work without them. Docling is loaded only for binary document conversion and is
configured for local table extraction and EasyOCR in Spanish and English. The tokenizer adapter
requires the model files to already exist locally; no hidden download occurs at ingestion time.

Run migrations once:

```python
from raglab.storage import PostgresRepository

repository = PostgresRepository("postgresql://raglab:raglab@127.0.0.1:5432/raglab")
repository.migrate()
```

## Ingest a source

```python
from raglab import CollectionConfig, SourceInput, ingest

report = ingest(
    SourceInput.path("document.pdf"),
    CollectionConfig(name="documents"),
    dsn="postgresql://raglab:raglab@127.0.0.1:5432/raglab",
)
print(report)
```

Markdown and UTF-8 text use a lossless direct adapter. PDF, Office, HTML, CSV, and image inputs
use local Docling. HTTP(S) URLs are downloaded and converted locally by default.

### Remote Jina Reader is explicit

Jina Reader is never a fallback. It can receive document contents, so it is restricted to a URL
that resolves entirely to public IP addresses and requires both controls:

```python
source = SourceInput.url("https://example.org/article", allow_remote_service=True)
report = pipeline.ingest(source, collection, use_jina=True)
```

Localhost, credentials in URLs, non-HTTP schemes, private/link-local/reserved networks, and local
files are rejected. DNS is validated before calling the service.

## Inspect and search

The migration creates `collections`, `documents`, `chunks`, an HNSW cosine index, and the
`collection_stats` view. `PostgresRepository.search(..., exact=True)` forces a sequential exact
diagnostic query; the default path uses HNSW and permits an `ef_search` override. Comparing both
is useful for verifying recall during development.

For pgAdmin 4 on Windows, register a server with host `localhost`, port `5432`, database/user/
password `raglab`. Docker publishes PostgreSQL only on the WSL loopback interface. Current WSL
normally forwards Windows localhost; if it does not, use `hostname -I` inside WSL as a temporary
fallback and review your firewall before exposing any additional interface.

## Notebook and verification

`notebooks/01_ingestion_pipeline.ipynb` contains an output-free, package-only walkthrough. Set
`RAGLAB_PDF` and `RAGLAB_DSN` to override its source and database. Service failures include the
command needed to start or configure the missing dependency.

```bash
pytest
ruff check .
mypy src
jupyter execute notebooks/01_ingestion_pipeline.ipynb --output /tmp/raglab.ipynb
```

Integration tests are opt-in because they require heavyweight services:

```bash
pytest -m integration
pytest -m e2e
```

Retrieval product APIs, generation, reranking, and citations are intentionally outside this
phase.
