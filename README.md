# RAGLab

RAGLab is a local-first, inspectable ingestion pipeline for learning the foundations of
retrieval-augmented generation. It converts sources to canonical Markdown, parses their
structure, refines oversized sections with semantic boundaries, creates Ollama embeddings,
and stores them in PostgreSQL with pgvector. It deliberately does not use LangChain or
LlamaIndex.

## Architecture

```mermaid
flowchart LR
    FILE["Local file"] --> CONVERT{"Choose converter"}
    URL["HTTP(S) URL"] --> CONVERT
    MEMORY["In-memory Markdown"] --> CONVERT

    CONVERT -->|Markdown or text| DIRECT["Direct, lossless conversion"]
    CONVERT -->|PDF, Office, HTML, image| DOCLING["Local Docling conversion"]
    CONVERT -->|URL by default| LOCAL_URL["Local download + direct or Docling"]
    CONVERT -->|URL + --use-jina| JINA["Jina Reader<br/>external opt-in"]

    DIRECT --> MARKDOWN["Canonical Markdown<br/>+ provenance"]
    DOCLING --> MARKDOWN
    LOCAL_URL --> MARKDOWN
    JINA --> MARKDOWN

    MARKDOWN --> AST["Markdown AST blocks"]
    AST --> UNITS["Structural units"]
    UNITS --> BOUNDARY["Temporary Ollama embeddings<br/>for boundary evidence"]
    UNITS --> CHUNKING["Hybrid chunking"]
    BOUNDARY --> CHUNKING
    CHUNKING --> CHUNKS["Faithful content<br/>+ contextual embedding_text"]
    CHUNKS --> OLLAMA["Final local Ollama embeddings<br/>1024 dimensions"]
    CHUNKS --> TX["Atomic storage transaction"]
    OLLAMA --> TX
    TX --> POSTGRES["PostgreSQL + pgvector<br/>document, chunks, citations, vectors"]
```

The temporary embeddings compare neighbouring structural units and are discarded after chunking.
The final stored vectors are generated separately from each chunk's `embedding_text`. `content`
remains faithful to the converted source, while title and heading breadcrumbs are added only to
`embedding_text`. All embeddings are produced before the database transaction begins.
Within one collection, the `(source URI, content hash, pipeline fingerprint)` tuple makes repeated
ingestion a no-op. The fingerprint includes citable source identity and provenance, so content or
provenance changes replace the document's chunks atomically. RAGLab treats a collection name as
permanently bound to its embedding and chunking configuration; a different configuration belongs
in another collection.

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

## Collections are retrieval boundaries

A collection is **not just a folder or label**. It is a logical search boundary plus an immutable
retrieval contract. Think of it as one library room whose books share the same cataloguing rules:
documents may have different formats and topics, but their chunks are prepared and compared under
one declared model, vector dimension, distance metric, and chunking profile.

PostgreSQL stores this contract once in `collections`. Every document points to one collection,
and every chunk points to one document. Retrieval always names a collection, so chunks from other
collections are excluded from that search.

The boundary is logical, not a separate PostgreSQL database or table. All chunk vectors currently
share the same `chunks` table and HNSW index; the SQL query joins through `documents` and filters by
the selected collection name.

### What a collection fixes

| Field | Why it belongs to the collection |
|---|---|
| `name` | Stable identifier supplied to ingestion and retrieval |
| `model` | Embedding model that gives every vector its coordinate system |
| `dimension` | Vector length; currently fixed by the schema at 1024 |
| `metric` | Distance interpretation; currently cosine |
| `chunk_config` | Structural and semantic rules used to create retrieval units |

The first ingestion creates the collection row. Later ingestion with the same name must provide
exactly the same model, dimension, metric, and `chunk_config`. A mismatch raises an error **before
documents are mixed**. RAGLab therefore refuses to combine p80 chunks with p90 chunks in one named
collection.

### Reuse a collection or declare a new one?

Start with the retrieval question: **should these chunks compete in the same search result?** Then
check whether they use the same technical contract.

| Situation | Decision | Why |
|---|---|---|
| Add another Markdown, PDF, DOCX, ODT, or URL to the same knowledge base | Reuse | File format does not define vector compatibility |
| Ingest a newer edition at the same `source_uri` | Reuse | It is the same logical resource and should replace its old chunks |
| Add a different language that should be searched together with the corpus | Reuse | The current embedding model is multilingual and the profile is unchanged |
| Compare p80 against p90 chunking | New collection | Each profile creates different retrieval units and must be evaluated independently |
| Change embedding model, dimension, or distance metric | New collection | Old and new vectors do not share the same retrieval contract |
| Keep product manuals and HR policies out of each other's results | Usually new collection | They answer different search intents and should not compete |
| Create a frozen historical snapshot while keeping the current corpus | New collection or explicit version model | Reusing the collection intentionally replaces the current source version |
| Add one more ordinary document with the same profile | Reuse | One collection per document creates needless operational fragmentation |

A collection is **not an authorization boundary**. A separate collection is useful for retrieval
isolation, but confidential tenants or environments still require PostgreSQL permissions, schemas,
databases, or application-level access control.

```mermaid
flowchart TD
    START["New ingestion request"] --> TOGETHER{"Should its chunks compete<br/>in the same searches?"}
    TOGETHER -->|No| NEW["Declare a new collection"]
    TOGETHER -->|Yes| MODEL{"Same model, dimension,<br/>and metric?"}
    MODEL -->|No| NEW
    MODEL -->|Yes| PROFILE{"Same chunking profile?"}
    PROFILE -->|No| NEW
    PROFILE -->|Yes| URI{"Same source_uri already<br/>exists in the collection?"}
    URI -->|No| ADD["Add a new document"]
    URI -->|Yes| IDENTITY{"Same content hash<br/>and fingerprint?"}
    IDENTITY -->|Yes| SKIP["Skip: keep existing vectors"]
    IDENTITY -->|No| REPLACE["Update document<br/>and replace old chunks"]
```

### Name collections by corpus and contract

Prefer a stable, readable name such as:

```text
product-manuals-qwen-p90
hr-policies-qwen-p90
product-manuals-qwen-p80-experiment
```

The name should describe **what is searched together** and, when multiple profiles coexist, the
profile. Do not include a content hash unless the collection intentionally represents an immutable
snapshot. Different formats do not need format-specific collections when they belong to the same
corpus.

### Reingestion inside one collection

Identity is scoped by `(collection, source_uri)`:

| Situation | Result |
|---|---|
| Same source, content, provenance, and configuration | `skipped`; existing UUID and vectors remain |
| Same source URI, changed content or provenance | Same document UUID is updated; old chunks are replaced atomically |
| Same filename at a different path | Separate document because the absolute `source_uri` differs |
| Same source in another collection | Separate indexed artifact |

The current schema stores the latest version of one source per collection; it is not a version
history. Model `document_versions` explicitly if old editions must remain searchable or auditable.
If a local file is moved, its absolute `source_uri` changes and PostgreSQL sees a new document; the
old row must be removed explicitly if it should no longer be searchable.

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

## PostgreSQL + pgvector storage model

The database keeps configuration, source-level truth, and retrieval units at different levels.
This avoids copying the full document metadata into every chunk while still allowing retrieval to
return faithful evidence and a complete citation.

```mermaid
erDiagram
    COLLECTIONS ||--o{ DOCUMENTS : contains
    DOCUMENTS ||--o{ CHUNKS : produces

    COLLECTIONS {
        uuid id PK
        text name UK
        text model
        integer dimension
        text metric
        jsonb chunk_config
        timestamptz created_at
    }

    DOCUMENTS {
        uuid id PK
        uuid collection_id FK
        text source_uri
        text source_name
        text media_type
        text title
        text markdown
        text content_hash
        text converter
        text converter_version
        jsonb metadata
        jsonb line_provenance
        text provenance_status
        jsonb provenance_warnings
        text fingerprint
        timestamptz created_at
        timestamptz updated_at
    }

    CHUNKS {
        uuid id PK
        uuid document_id FK
        integer chunk_index
        text content
        text embedding_text
        integer token_count
        jsonb heading_path
        integer start_line
        integer end_line
        integer start_page
        integer end_page
        jsonb metadata
        vector embedding
    }
```

The important relational constraints are:

- `collections.name` is globally unique;
- `(documents.collection_id, documents.source_uri)` is unique, so one collection stores one current
  row for each logical source;
- `(chunks.document_id, chunks.chunk_index)` is unique;
- deleting a collection cascades to its documents and chunks;
- deleting a document cascades to its chunks.

### What belongs at each level

| Level | Stored information | Reason |
|---|---|---|
| Collection | Model, dimension, cosine metric, complete chunk profile | Defines how all children are produced and compared |
| Document | Full canonical Markdown, source identity, title, media type, converter, hash, fingerprint, metadata, line/page provenance and warnings | Preserves source-level truth once |
| Chunk | Faithful `content`, contextual `embedding_text`, token count, heading path, line/page ranges, chunk metadata and `vector(1024)` | Supplies the unit ranked and cited by retrieval |

`content` is the evidence eventually shown to an LLM or user. `embedding_text` adds title and
heading context only to produce a better vector; storing it makes the indexed artifact inspectable,
but retrieval returns faithful `content` instead. Document metadata is joined at search time to
build the citation, so title and source identity do not need to be duplicated into every chunk.

The HNSW index exists on `chunks.embedding`. It is a shared physical index over the table, while
the collection name remains the logical filter. The `collection_stats` view joins all three levels
to expose document, chunk, and token totals per collection.

### Atomic writes and replacements

An unchanged source is detected by collection contract, `source_uri`, `content_hash`, and
fingerprint and returns `skipped`. For a changed source, RAGLab creates every replacement embedding
before opening the database transaction. Inside that transaction it:

1. updates the existing document row while preserving its UUID;
2. deletes all old chunks for that document;
3. inserts the complete new chunk set and vectors;
4. commits everything together.

If storage fails, PostgreSQL rolls back the transaction instead of leaving a half-updated document.
The previous edition is replaced, not retained as history.

### How retrieval reconstructs citable evidence

```mermaid
flowchart LR
    QUESTION["User query"] --> QUERY_VECTOR["Ollama query embedding"]
    COLLECTION["Selected collection name"] --> SEARCH["PostgresRepository.search"]
    QUERY_VECTOR --> SEARCH
    SEARCH --> FILTER["Scope chunks to the selected collection"]
    FILTER --> RANK["Cosine distance<br/>HNSW or exact diagnostic"]
    RANK --> JOIN["Join document metadata<br/>and chunk location fields"]
    JOIN --> RESULT["SearchResult<br/>faithful content + distance + Citation"]
```

Search does not return a raw vector as evidence. It uses the vector only for ranking, then returns
the stored faithful chunk plus the document title/source and the chunk heading, page, and line
ranges needed to cite it.

### Inspect indexed data

The CLI output is the first verification boundary. PostgreSQL can then show collection totals:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT * FROM collection_stats ORDER BY name;"
```

Inspect the immutable contract attached to each collection:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT name, model, dimension, metric, jsonb_pretty(chunk_config) AS chunk_config FROM collections ORDER BY name;"
```

Inspect documents and their citable provenance without reading vector payloads:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT source_uri, source_name, title, provenance_status, updated_at FROM documents ORDER BY updated_at DESC;"
```

Check whether the same source exists in more than one collection:

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT c.name, d.source_uri, d.id, count(ch.id) AS chunks FROM collections c JOIN documents d ON d.collection_id = c.id LEFT JOIN chunks ch ON ch.document_id = d.id GROUP BY c.name, d.source_uri, d.id ORDER BY d.source_uri, c.name;"
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
