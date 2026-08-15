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
Within one collection, the `(source URI, content hash, pipeline fingerprint)` tuple makes repeated
ingestion a no-op. The fingerprint includes citable source identity and provenance, so content or
provenance changes replace the document's chunks atomically. A collection name is permanently bound
to its embedding and chunking configuration; a different configuration belongs in another
collection.

## Quick start without notebooks

Requirements: Python 3.12, Docker, and Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tokenizers,conversion]'
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
docker compose up -d --wait
ollama pull qwen3-embedding:0.6b
raglab-ingest --help
```

If Ollama is not already managed as a system service, run `ollama serve` in another terminal. The
Compose service exposes PostgreSQL only at `127.0.0.1:5432`.

Now ingest the tracked Markdown sample from the terminal:

```bash
raglab-ingest data/samples/aster_greenhouse_controller_manual.md \
  --collection greenhouse-manuals
```

The command runs the complete pipeline and prints a machine-readable report:

```json
{
  "chunk_count": 9,
  "collection": "greenhouse-manuals",
  "document_id": "...",
  "provenance_status": "complete",
  "status": "indexed"
}
```

Exact chunk counts depend on the document and chunking profile. Run the same command again and
`status` becomes `skipped` with `chunk_count: 0`; PostgreSQL does not receive duplicate vectors.

### What `raglab-ingest` does

The CLI is a thin `argparse` adapter over the same public Python pipeline used by the tests and
notebook. It does not introduce a framework or a second ingestion implementation:

1. resolves a local path to one stable absolute `source_uri`, or accepts an HTTP(S) URL;
2. creates or verifies the PostgreSQL/pgvector schema;
3. converts the source to canonical Markdown and records observable provenance warnings;
4. parses Markdown into structural blocks;
5. chunks using headings, token budgets, and semantic boundary evidence;
6. creates final 1024-dimensional embeddings with local Ollama;
7. atomically inserts or replaces the document and its chunks;
8. prints the `IngestionReport` as JSON.

Use `raglab-ingest --help` for the complete argument list. `RAGLAB_DSN` overrides the default
Compose connection without placing credentials in shell history:

```bash
export RAGLAB_DSN='postgresql://raglab:raglab@127.0.0.1:5432/raglab'
```

### Ingest local files

Native Markdown and UTF-8 text use the lossless direct converter and do not require Docling:

```bash
raglab-ingest ./knowledge/guide.md --collection product-knowledge
raglab-ingest ./knowledge/notes.txt --collection product-knowledge
```

PDF and document formats are converted locally with Docling:

```bash
raglab-ingest /absolute/path/operator-manual.pdf --collection operator-manuals
raglab-ingest ./documents/handbook.docx --collection company-handbook
raglab-ingest ./documents/policy.odt --collection company-handbook
raglab-ingest ./documents/reference.html --collection web-archive
```

Install the local conversion extras with `python -m pip install -e '.[conversion,tokenizers]'` for
these sources. Docling is imported only when a non-text source needs it and is configured for
local table extraction and EasyOCR in Spanish and English.

### Ingest a public URL

HTTP(S) sources are downloaded and converted locally by default. The original URL remains the
stable `source_uri` even when redirects are followed:

```bash
raglab-ingest 'https://example.org/public-guide.html' --collection public-guides
raglab-ingest 'https://example.org/manual.pdf' --collection public-guides
```

Jina Reader is a separate, explicit privacy decision because document contents leave the local
machine. It is never an automatic fallback:

```bash
raglab-ingest 'https://example.org/article' \
  --collection public-articles-jina \
  --use-jina
```

`--use-jina` accepts only public HTTP(S) URLs. Local files, credentials in URLs, localhost, and
private, reserved, or link-local network targets are rejected.

### Choose a chunking profile

The default profile is target 512, minimum 120, maximum 768, semantic percentile 90, and zero
overlap. Override it explicitly when running an experiment:

```bash
raglab-ingest ./documents/manual.pdf \
  --collection manuals-p80 \
  --target-tokens 512 \
  --min-tokens 120 \
  --max-tokens 768 \
  --semantic-percentile 80
```

A collection is one comparable vector space with one stable configuration. Reusing
`manuals-p80` later with percentile 90 fails instead of silently mixing incompatible artifacts;
use another collection name such as `manuals-p90`.

### Reingestion and document versions

Identity is scoped by `(collection, source_uri)`:

| Situation | Result |
|---|---|
| Same source, content, provenance, and configuration | `skipped`; existing UUID and vectors remain |
| Same source URI, changed content or provenance | Same document UUID is updated; old chunks are replaced atomically |
| Same filename at a different path | Separate document because the absolute `source_uri` differs |
| Same source in another collection | Separate indexed artifact |

The current schema stores the latest version of one source per collection; it is not a version
history. Model `document_versions` explicitly if old editions must remain searchable or auditable.

## Use the pipeline from a Python script

The CLI is convenient automation, while the public API is better when application code needs to
choose sources or compose further stages. A standalone script does not need Jupyter:

```python
import os
from pathlib import Path

from raglab import CollectionConfig, SourceInput, ingest
from raglab.chunking import ChunkingConfig
from raglab.storage import PostgresRepository

dsn = os.environ.get(
    "RAGLAB_DSN",
    "postgresql://raglab:raglab@127.0.0.1:5432/raglab",
)
chunk_config = ChunkingConfig(
    target_tokens=512,
    min_tokens=120,
    max_tokens=768,
    semantic_percentile=90,
)
collection = CollectionConfig(
    name="product-knowledge",
    chunk_config={
        "strategy": "structure_plus_semantics",
        "target_tokens": chunk_config.target_tokens,
        "min_tokens": chunk_config.min_tokens,
        "max_tokens": chunk_config.max_tokens,
        "semantic_percentile": chunk_config.semantic_percentile,
        "overlap_tokens": chunk_config.overlap_tokens,
    },
)
source = SourceInput.path(Path("documents/guide.pdf").resolve())

repository = PostgresRepository(dsn)
repository.migrate()

report = ingest(
    source,
    collection,
    dsn=dsn,
    chunk_config=chunk_config,
)
print(report)
```

Use `SourceInput.url("https://example.org/guide")` for local URL conversion, or
`SourceInput.text(markdown, uri="memory://stable-name")` for Markdown already held in memory. A
stable URI is important: it is how later executions find the same logical document.

## Inspect indexed data

The CLI output is the first verification boundary. PostgreSQL can then show collection totals:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT * FROM collection_stats ORDER BY name;"
```

Inspect documents and their citable provenance without reading vector payloads:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT source_uri, source_name, title, provenance_status, updated_at FROM documents ORDER BY updated_at DESC;"
```

Common failures are deliberately explicit:

| Error | Meaning and action |
|---|---|
| PostgreSQL connection refused | Run `docker compose up -d --wait` and check `RAGLAB_DSN` |
| Ollama cannot be reached | Start `ollama serve` or the local Ollama service |
| Embedding model unavailable | Run `ollama pull qwen3-embedding:0.6b` |
| Docling conversion unavailable | Install `python -m pip install -e '.[conversion,tokenizers]'` |
| Tokenizer files unavailable locally | Run the explicit `AutoTokenizer.from_pretrained(...)` setup command once |
| Collection exists with another configuration | Reuse the original flags or choose a new collection name |

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

The direct path is intentionally limited to `.md`, `.markdown`, and `.txt`. Other local files are
delegated to the installed Docling version. PDF, HTML, DOCX, ODT, ODS, and ODP routing is covered
by this project; acceptance of additional Docling formats such as presentations, spreadsheets,
CSV, or images depends on that installed Docling release and may provide weaker provenance.

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
