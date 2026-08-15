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
  -> canonical Markdown
  -> Markdown AST blocks (markdown-it-py)
  -> structural units -> semantic boundary refinement
  -> Ollama qwen3-embedding:0.6b (1024 dimensions)
  -> one atomic PostgreSQL document replacement
  -> faithful retrieval content + structured citation
```

`content` remains faithful to the converted source. Title and heading breadcrumbs are added
only to `embedding_text`. All embeddings are produced before the database transaction begins.
The `(source URI, content hash, pipeline fingerprint)` tuple makes repeated ingestion a no-op.
The fingerprint includes citable source identity and provenance, so content, provenance, or
configuration changes replace the document's chunks atomically.

## Setup

Requirements: Python 3.12, Docker, and Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tokenizers]'
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
docker compose up -d
ollama pull qwen3-embedding:0.6b
```

The first notebook uses a versioned Markdown fixture and does not need Docling. Install
`raglab[conversion]` when experimenting with PDF, Office, HTML, CSV, or image sources. Docling is
loaded only for those formats and is configured for local table extraction and EasyOCR in Spanish
and English. The tokenizer adapter requires the model files to already exist locally; no hidden
download occurs at ingestion time.

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

## Format and provenance support

Every format becomes canonical Markdown. Location metadata describes that Markdown without
inventing coordinates the source does not provide.

| Source | Converter | Original lines | Pages | Expected status |
|---|---|---:|---:|---|
| Markdown / UTF-8 text | Direct | Yes | No | `complete` |
| Local or public HTML | Docling | No | No | `unavailable` with a warning |
| PDF | Docling | No | Yes, when page markers map successfully | `complete`, or `partial` with a warning |
| DOCX | Docling | No | Only when Docling exposes meaningful pages | `complete`, `partial`, or `unavailable` |
| ODT / ODS / ODP | Docling | No | Only when Docling exposes meaningful pages | `complete`, `partial`, or `unavailable` |

`source_name` is always a displayable source identity. A local source uses its filename. A URL
uses, in order, its Content-Disposition filename, decoded final path segment, hostname, or full
URI. `title` is different: it stores a parsed H1 or converter-recognised title and remains `NULL`
when no real title exists. Citation UIs may display `source_name` as a fallback, but that fallback
is not persisted as a fabricated title.

Line and page fields are 1-based and nullable:

- `markdown_line` identifies a line in canonical Markdown and is always present in each line
  provenance record;
- `source_line` identifies an original source line when that relationship is exact;
- `start_line` / `end_line` are canonical chunk line ranges;
- `page_number` and chunk `start_page` / `end_page` exist only for meaningful pagination.

HTML and Markdown never receive an invented page 1. If optional location enrichment fails while
conversion succeeds, ingestion continues with canonical lines, a `partial` or `unavailable`
status, and warnings in both `IngestionReport` and PostgreSQL. Conversion, parsing, embedding, and
database failures remain hard failures.

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

Each `SearchResult` keeps the existing IDs, URI, distance, and faithful chunk `content`, and adds
a structured `citation` with source name, nullable title, heading path, canonical line range,
nullable page range, and provenance status. `embedding_text` may contain title and heading context
to improve vector search, but it is vector-only and is never returned as evidence. Future prompt
assembly should pass `result.content` together with `result.citation`, not `embedding_text`.

For pgAdmin 4 on Windows, register a server with host `localhost`, port `5432`, database/user/
password `raglab`. Docker publishes PostgreSQL only on the WSL loopback interface. Current WSL
normally forwards Windows localhost; if it does not, use `hostname -I` inside WSL as a temporary
fallback and review your firewall before exposing any additional interface.

## Sequential notebook curriculum

Notebook names follow `NN_<rag_stage>.ipynb`. Only the first stage exists today; the remaining
rows document the intended learning order rather than placeholder files.

| Stage | Notebook | Outcome |
|---|---|---|
| 01 | `01_ingestion_and_indexing.ipynb` | Inspect conversion, AST parsing, chunking, embeddings, and pgvector indexing |
| 02 | `02_retrieval.ipynb` | Compare query design, filters, recall, and ranking |
| 03 | `03_generation.ipynb` | Generate answers from retrieved evidence |
| 04 | `04_rag_evaluation.ipynb` | Evaluate the complete RAG behaviour |

The first notebook uses the fictional Aster greenhouse controller manual under `data/samples/`.
Its separate manifest under `data/evaluation/` defines boundaries that should stay together or
split, plus retrieval queries with known answer anchors. Both files are tracked inputs;
`data/generated/` remains ignored for disposable conversion artifacts.

The notebook is committed without outputs. Run it from the project environment after starting
PostgreSQL and Ollama. Set `RAGLAB_DSN` only when the database is not using the Compose default.

```bash
pytest
ruff check .
mypy src
jupyter execute notebooks/01_ingestion_and_indexing.ipynb --output /tmp/raglab.ipynb
```

Integration tests are opt-in because they require heavyweight services:

```bash
pytest -m integration
pytest -m e2e
```

The first notebook performs diagnostic retrieval and inspects citable results. Product retrieval
orchestration, generation, and reranking remain outside this stage.
