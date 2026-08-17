# RAGLab

RAGLab is a local-first, inspectable laboratory for learning **Retrieval-Augmented Generation
(RAG)** from first principles. RAG retrieves source evidence before a language model answers,
reducing reliance on the model's internal memory and making citations possible. This repository
teaches the two foundations that determine whether that evidence is useful:

1. **ingestion and indexing** — convert sources into faithful, searchable units; and
2. **hybrid retrieval** — combine semantic and lexical search, then refine the evidence.

The components remain explicit. There is no LangChain or LlamaIndex layer hiding conversion,
chunking, ranking, SQL, or failure modes. Ollama runs embeddings and optional query rewriting
locally; Docling converts complex files locally; PostgreSQL stores the inspectable artifacts.
Jina Reader is available only as an explicit opt-in for public URLs.

## Learning path

| Step | Start here | What you learn |
| ---- | ---------- | -------------- |
| 1 | This introduction and [architecture](#architecture) | How a source becomes citable evidence |
| 2 | [Chapter 1](#chapter-1--ingestion-and-indexing) | Conversion, chunking, vectors, and atomic storage |
| 3 | `notebooks/01_ingestion_and_indexing.ipynb` | Inspect every ingestion artifact |
| 4 | [Chapter 2](#chapter-2--hybrid-retrieval) | ANN, BM25, RRF, reranking, expansion, and MMR |
| 5 | `notebooks/02_retrieval.ipynb` | Change a query and observe every retrieval stage |
| 6 | [Appendix A](#appendix-a--test-strategy-and-suite) | Prove each system boundary |

The tracked fictional **Aster Greenhouse Controller Manual** provides a controlled corpus with
known boundaries and answer anchors. For a real-world example, search arXiv for the well-known
paper *Attention Is All You Need*. RAGLab used it to test the complete ingestion path with a real
PDF.

## Architecture

```mermaid
flowchart LR
    SOURCE["File, URL, or in-memory Markdown"] --> CONVERT["Local conversion"]
    CONVERT --> MARKDOWN["Canonical Markdown + provenance"]
    MARKDOWN --> AST["Markdown AST"]
    AST --> CHUNK["Structural + semantic chunking"]
    CHUNK --> EMBED["Local Qwen embeddings"]
    EMBED --> DB["ParadeDB / PostgreSQL 17"]
    DB --> ANN["Semantic ANN / HNSW"]
    DB --> BM25["Lexical BM25"]
    ANN --> RANK["Fusion + refinement"]
    BM25 --> RANK
    RANK --> EVIDENCE["Faithful content + citation + ranking trace"]
```

An **AST (Abstract Syntax Tree)** represents Markdown as typed blocks such as headings,
paragraphs, lists, and code. An **embedding** is a numeric vector that places semantically related
text near each other. RAGLab compares embeddings with cosine distance.

### Why ParadeDB and PostgreSQL 17

Compose pins `paradedb/paradedb:0.25.0-pg17`. It combines:

- PostgreSQL transactions and relational constraints;
- JSONB metadata and citable document provenance;
- pgvector vectors and **HNSW (Hierarchical Navigable Small World)** semantic search;
- ParadeDB's `pg_search` extension and **BM25 (Best Matching 25)** lexical search.

Keeping both retrieval channels in one database avoids synchronizing document IDs, filters, and
updates across a vector database and a search engine. It also makes replacement atomic: document
truth and every derived chunk commit or roll back together.

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
    }
    DOCUMENTS {
        uuid id PK
        uuid collection_id FK
        text source_uri
        text source_name
        text title
        text markdown
        text content_hash
        jsonb metadata
        jsonb line_provenance
        text provenance_status
        text fingerprint
    }
    CHUNKS {
        uuid id PK
        uuid document_id FK
        integer chunk_index
        text content
        text embedding_text
        integer token_count
        jsonb heading_path
        jsonb metadata
        vector embedding
    }
```

`collections` define a retrieval contract. `documents` preserve canonical source truth and
provenance. `chunks` are searchable and citable units. The schema fixes vectors at 1024 dimensions
with cosine distance.

## Quick start

Requirements: Python 3.12, Docker, and Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tokenizers,conversion,retrieval]'
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
docker compose up -d --wait
ollama pull qwen3-embedding:0.6b
```

Run `ollama serve` in another terminal if Ollama is not already a system service. Compose exposes
PostgreSQL only at `127.0.0.1:5432`.

```bash
raglab-ingest data/samples/aster_greenhouse_controller_manual.md \
  --collection greenhouse-manuals

raglab-retrieve "What causes fault E17?" \
  --collection greenhouse-manuals
```

Running ingestion again returns `status: "skipped"` and `chunk_count: 0`; it does not duplicate
vectors.

# Chapter 1 — Ingestion and indexing

Ingestion turns heterogeneous sources into one explicit, reproducible retrieval contract. Think
of each chunk as a library card: it must carry enough context to be found, while its quotation
remains faithful to the source.

## Ingestion pipeline

```mermaid
flowchart TD
    INPUT["Local file, public URL, or memory text"] --> ROUTE{"Source type?"}
    ROUTE -->|Markdown / text| DIRECT["Lossless direct conversion"]
    ROUTE -->|PDF / Office / HTML / image| DOCLING["Local Docling conversion"]
    ROUTE -->|URL + --use-jina| JINA["External Jina Reader opt-in"]
    DIRECT --> CANONICAL["Canonical Markdown + provenance"]
    DOCLING --> CANONICAL
    JINA --> CANONICAL
    CANONICAL --> AST["Markdown AST blocks"]
    AST --> UNITS["Structural units"]
    UNITS --> TEMP["Temporary embeddings for boundary evidence"]
    UNITS --> BOUNDARIES["Heading + token + semantic boundaries"]
    TEMP --> BOUNDARIES
    BOUNDARIES --> CHUNKS["Faithful content + contextual embedding_text"]
    CHUNKS --> FINAL["Final Qwen embeddings, 1024 dimensions"]
    FINAL --> TX["Atomic PostgreSQL transaction"]
    TX --> STORED["Document, chunks, vectors, citations"]
```

### Canonical Markdown and provenance

Native `.md`, `.markdown`, and `.txt` files use a lossless direct converter. PDF, HTML, DOCX, ODT,
ODS, ODP, and other supported complex formats use local Docling. Public URLs are downloaded and
converted locally by default.

Every successful path emits canonical Markdown plus source identity, converter version, content
hash, and observable line/page provenance. RAGLab never invents page 1 for a source without
meaningful pagination. Optional location enrichment may be `partial` or `unavailable`; conversion,
parsing, embedding, and storage failures remain hard failures.

### AST, structure, and semantic boundaries

The Markdown parser builds an AST so headings, paragraphs, lists, tables, and code blocks remain
recognizable. These structural units are the first chunk candidates. Headings and token budgets
decide hard boundaries. When a unit is too large, temporary embeddings compare neighbouring
material; a large semantic change becomes evidence for a split. These temporary vectors are
discarded.

The chunker preserves faithful `content`. It builds separate `embedding_text` from the title,
heading breadcrumbs, and faithful chunk. That contextual text creates the final vector, while
retrieval returns only `content`.

### Final embeddings and atomic storage

Ollama's `qwen3-embedding:0.6b` produces final 1024-dimensional vectors. Cosine distance compares
their direction rather than magnitude. Every embedding completes before the transaction opens.

PostgreSQL inserts a new document or updates the current UUID, deletes old chunks, inserts the
complete replacement, and commits. A failure rolls back the whole replacement.

## Collections, fingerprints, and idempotency

A collection is not a folder. It is a search boundary whose documents share one embedding model,
dimension, distance metric, and chunking profile. Ask: **should these chunks compete in the same
results?**

| Situation | Decision |
| --------- | -------- |
| More documents with the same intent and contract | Reuse the collection |
| New edition at the same `source_uri` | Reuse; replace atomically |
| Different model, dimension, metric, or chunking | Create a collection |
| Corpus that must not compete in the same results | Usually create a collection |

The fingerprint includes pipeline version, converter, source identity and provenance, model,
dimension, metric, and chunk configuration.

| Identity check | Result |
| -------------- | ------ |
| Same `source_uri`, content hash, and fingerprint | Skip; keep current UUID and vectors |
| Same `source_uri`, changed content or provenance | Update and atomically replace chunks |
| Different `source_uri` | Store a separate document |

The schema stores the latest version, not history. A collection is not an authorization boundary;
enforce tenant access separately.

## `raglab-ingest` reference

Only `source` is required. The CLI is a thin adapter over the public pipeline used by tests and
notebooks.

| Parameter | Default | Meaning, range, and tradeoff |
| --------- | ------- | ---------------------------- |
| `source` | Required | Local path or public HTTP(S) URL. Local paths resolve to stable absolute URIs. |
| `--collection` | `documents` | Search boundary. Reuse only with a compatible contract. |
| `--dsn` | `RAGLAB_DSN`, else `postgresql://raglab:raglab@127.0.0.1:5432/raglab` | PostgreSQL connection string. |
| `--target-tokens` | `512` | Preferred size. Larger chunks add context but reduce precision. |
| `--min-tokens` | `120` | Minimum preferred size. Raising it reduces fragments but can merge ideas. |
| `--max-tokens` | `768` | Upper chunk target. Raising it increases context and embedding cost. |
| `--semantic-percentile` | `90` | Lower values split more often; higher values require stronger evidence. |
| internal overlap | `0` | Fixed default; avoids duplicated evidence. |
| `--use-jina` | Disabled | Sends a public URL to Jina Reader; never automatic and rejects local/private targets. |
| `-h`, `--help` | Disabled | Print parser help and exit. |

Example profile experiment:

```bash
raglab-ingest ./documents/manual.pdf \
  --collection manuals-p80 \
  --target-tokens 512 \
  --min-tokens 120 \
  --max-tokens 768 \
  --semantic-percentile 80
```

Use a new collection because the profile is part of its immutable contract.

## Ingestion notebooks

### `01_ingestion_and_indexing.ipynb`

The guided notebook exposes conversion, AST blocks, structural units, chunk diagnostics,
`content` versus `embedding_text`, final vectors, PostgreSQL rows, exact/vector diagnostics, and
citations. The tracked Aster manual keeps observations repeatable.

### `benchmark_ingestion_hyperparameters.ipynb`

This unnumbered workbench compares chunking profiles and HNSW behavior, records PostgreSQL's
chosen plan, and measures recall@K, latency, build time, and index size. Imagine HNSW as a
multilayer network of shortcuts: upper layers approach a promising region and lower layers refine
the neighbourhood.

The migration fixes `m=16` and `ef_construction=64`. These are physical index-build settings, not
per-collection ingestion flags.

## Optional real PDF: *Attention Is All You Need*

Download the paper from arXiv and opt into the notebook appendix:

```bash
curl -L https://arxiv.org/pdf/1706.03762 -o attention.pdf
export RAGLAB_PDF="$PWD/attention.pdf"
jupyter execute notebooks/01_ingestion_and_indexing.ipynb \
  --output /tmp/raglab-ingestion-attention.ipynb
```

The Aster corpus remains the controlled fixture; the paper provides a practical example for
exploring real PDF conversion.

# Chapter 2 — Hybrid retrieval

Retrieval asks two questions in parallel: “which chunks mean something similar?” and “which chunks
contain the discriminating words?” Semantic search answers the first; BM25 answers the second.

## Retrieval pipeline, in order

```mermaid
flowchart LR
    Q["1. Original query"] --> RW["2. Optional history rewrite + expansions"]
    RW --> ANN["3. Semantic ANN / HNSW"]
    RW --> BM25["4. Lexical BM25"]
    FILTER["5. Shared SQL filters"] --> ANN
    FILTER --> BM25
    ANN --> RRF["6. Reciprocal Rank Fusion"]
    BM25 --> RRF
    RRF --> BGE["7. BGE cross-encoder reranking"]
    BGE --> EXPAND["8. Small-to-big expansion"]
    EXPAND --> DEDUPE["9. Parent deduplication"]
    DEDUPE --> MMR["10. Maximal Marginal Relevance"]
    MMR --> OUT["11. Faithful content + citation + traces"]
```

1. The original query is always retained.
2. Rewriting optionally creates a standalone query and up to two expansions; failure falls back.
3. **ANN (Approximate Nearest Neighbors)** searches Qwen vectors through HNSW.
4. BM25 searches lexical evidence through ParadeDB.
5. Identical SQL filters constrain both channels and parent expansion.
6. **RRF (Reciprocal Rank Fusion)** combines ranks without equating incompatible raw scores.
7. `BAAI/bge-reranker-v2-m3`, a **BGE** cross-encoder, jointly scores query/document pairs.
8. Small-to-big expands a matched child across contiguous same-heading chunks within a budget.
9. Equal `document_id:first-last` parents are deduplicated.
10. **MMR (Maximal Marginal Relevance)** balances relevance and redundancy.
11. Results expose faithful content, citations, matched child IDs, and ranking traces.

Expansion stops at a heading change, gap, document boundary, token limit, or neighbour that fails
the filters. It does not persist new chunks.

## Exact search, ANN, HNSW, and PostgreSQL

Exact search computes cosine distance against every eligible vector and is the diagnostic
reference. ANN visits promising HNSW regions, reducing latency at the cost of possibly missing an
exact neighbour. `recall@K` measures agreement with exact top K.

`ef_search` controls HNSW query breadth. Higher values generally improve recall and cost latency.
RAGLab uses `100`; pgvector defaults to `40`.

An index-enabled query does **not** guarantee an HNSW scan. PostgreSQL may prefer exact scan and
sort for a small or selective collection; `EXPLAIN` is authoritative. All collections share one
physical HNSW index, so post-index filtering can hurt recall for a small collection in a large
table. Raising `ef_search` may help but does not remove that tradeoff.

## Filters and history

Repeat `--filter field:operator=value`. Identical predicates constrain ANN, BM25, and expansion,
so a rejected tenant cannot reappear through a neighbouring chunk.

| Field | Operators | Example |
| ----- | --------- | ------- |
| `document.metadata.<path>` | `eq`, `ne`, `in`, `contains` | `document.metadata.tenant_id:eq=acme` |
| `chunk.metadata.<path>` | `eq`, `ne`, `in`, `contains` | `chunk.metadata.language:in=["en","es"]` |
| `document.source_uri` | `eq`, `ne`, `in`, `contains` | `document.source_uri:contains=/manuals/` |
| `document.source_name`, `document.title`, `document.media_type` | `eq`, `ne`, `in`, `contains` | `document.media_type:eq=application/pdf` |

Values accept JSON scalars. `in` accepts a JSON list or comma-separated scalars.

`--history-file` reads strings or objects with a string `content`:

```json
[
  {"content": "We were discussing greenhouse controller faults."},
  {"content": "Which one requires replacing the sensor?"}
]
```

Providing history enables rewriting even without `--rewrite`.

## `raglab-retrieve` reference

Only `query` is required.

| Parameter | Default | Meaning, range, and tradeoff |
| --------- | ------- | ---------------------------- |
| `query` | Required | Non-blank question or search query. |
| `--collection` | `documents` | Boundary searched by ANN and BM25. |
| `--dsn` | `RAGLAB_DSN`, else Compose DSN | ParadeDB connection string. |
| `--filter` | None | Repeatable SQL prefilter. |
| `--candidate-k` | `50` | Positive, per channel and variant. More costs memory/latency but gives refinement more evidence. |
| `--top-k` | `5` | Positive final result limit after all refinement. |
| `--ef-search` | `100` | Positive HNSW breadth; trades latency for recall. |
| `--exact` | Disabled | Force exact semantic diagnostics; slower at scale. |
| `--rewrite` | Disabled | Ask local `qwen3:4b` for a standalone query. |
| `--history-file` | None | JSON conversation history; also enables rewriting. |
| `--expansions` | `0` | Additional variants, `0..2`; each adds ANN and BM25 work. |
| `--no-rerank` | Disabled | BGE stays enabled. This fallback reduces memory and latency. |
| `--no-mmr` | Disabled | MMR stays enabled; disabling may return redundant parents. |
| `--mmr-lambda` | `0.7` | `0..1`: `1` favors relevance, `0` diversity. |
| `--no-small-to-big` | Disabled | Expansion stays enabled; disabling returns matched chunks. |
| `--parent-max-tokens` | `1500` | Positive parent budget; more adds context and prompt cost. |
| `-h`, `--help` | Disabled | Print parser help and exit. |

Fixed non-CLI defaults are RRF `k=60`, semantic weight `1.0`, and BM25 weight `1.0`.

| Responsibility | Model |
| -------------- | ----- |
| Query and document embeddings | `qwen3-embedding:0.6b` via Ollama |
| Optional rewriting | `qwen3:4b` via Ollama |
| Reranking | `BAAI/bge-reranker-v2-m3` via Transformers/PyTorch |

```bash
python -m pip install -e '.[retrieval]'
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b  # only for rewriting
```

The JSON response contains queries, variants, filters, faithful content, citations, parent ranges,
matched children, and ANN, BM25, RRF, reranker, and MMR traces.

## `02_retrieval.ipynb`

This progressive playground first demonstrates pure ranking mechanics, then exposes editable
query, collection, filters, history, and every `RetrievalConfig` value. Use it to observe BM25,
ANN, RRF, BGE, small-to-big, and MMR.

Live PostgreSQL, Ollama, and BGE execution is disabled by default:

```bash
export RAGLAB_RUN_RETRIEVAL_NOTEBOOK=1
export RAGLAB_RETRIEVAL_COLLECTION=greenhouse-manuals
jupyter execute notebooks/02_retrieval.ipynb \
  --output /tmp/raglab-retrieval-live.ipynb
```

Without that variable, the full notebook still executes its hermetic examples.

# Appendix A — Test strategy and suite

The pyramid keeps algorithmic feedback fast and reserves real converters, databases, models, and
PDFs for explicit boundaries. The suite collects **76 tests**. The validated ordinary run produces
**72 passed and 4 skipped**; skips are environment-dependent integration or E2E cases.

## Levels

| Level | What it proves | Requirements |
| ----- | -------------- | ------------ |
| Unit | Conversion, parsing, chunking, embeddings, CLIs, ranking, reranking, SQL, orchestration | None |
| Docling integration | Real non-text conversion and provenance | Conversion extras |
| PostgreSQL integration | Migrations, atomic replacement, vectors, filters, BM25, retrieval SQL | Disposable ParadeDB |
| PDF E2E | PDF → Docling → Ollama → PostgreSQL | PDF, Docling, Ollama, PostgreSQL |
| Retrieval E2E | Real ANN + BM25 + BGE | ParadeDB, Ollama, BGE model |
| Notebooks | Teaching paths remain executable and output-free in Git | None by default; services for live cells |

```bash
# Fast hermetic suite
pytest -m "not integration and not e2e"

# Complete collection, including environment-gated skips
pytest

# Static quality and strict typing
ruff check .
mypy src/raglab

# Coverage
pytest --cov=raglab --cov-report=term-missing -m "not integration and not e2e"
```

Docling integration:

```bash
python -m pip install -e '.[conversion,tokenizers]'
pytest -m integration tests/integration/test_docling.py
```

PostgreSQL integration:

```bash
docker compose up -d --wait
export RAGLAB_TEST_DSN='postgresql://raglab:raglab@127.0.0.1:5432/raglab'
pytest -m integration tests/integration/test_postgres.py tests/integration/test_retrieval.py
```

PDF E2E:

```bash
export RAGLAB_E2E=1
export RAGLAB_PDF="$PWD/attention.pdf"
pytest -m e2e tests/e2e/test_pdf_ingestion.py
```

Hybrid retrieval E2E:

```bash
export RAGLAB_RETRIEVAL_E2E=1
export RAGLAB_TEST_DSN='postgresql://raglab:raglab@127.0.0.1:5432/raglab'
pytest -m e2e tests/e2e/test_hybrid_retrieval.py
```

Always write executed notebooks outside the repository:

```bash
jupyter execute notebooks/01_ingestion_and_indexing.ipynb \
  --output /tmp/raglab-ingestion.ipynb
jupyter execute notebooks/02_retrieval.ipynb \
  --output /tmp/raglab-retrieval.ipynb
jupyter execute notebooks/benchmark_ingestion_hyperparameters.ipynb \
  --output /tmp/raglab-benchmark.ipynb
```

Use a **disposable** `RAGLAB_TEST_DSN`: tests migrate the schema and create, replace, or delete test
collections. Never aim destructive fixtures at production.

## Subsystem test matrix

| Subsystem | Primary tests |
| --------- | ------------- |
| Conversion and provenance | `tests/test_conversion.py`, `tests/integration/test_docling.py`, `tests/e2e/test_pdf_ingestion.py` |
| Parsing and chunking | `tests/test_parsing.py`, `tests/test_chunking.py` |
| Embeddings and ingestion | `tests/test_embeddings.py`, `tests/test_pipeline.py` |
| Ingestion CLI and storage | `tests/test_cli.py`, `tests/test_storage.py`, `tests/integration/test_postgres.py` |
| Filters and retrieval SQL | `tests/test_retrieval_repository.py`, `tests/integration/test_retrieval.py` |
| RRF and MMR | `tests/test_retrieval_ranking.py` |
| Rewriting and history | `tests/test_retrieval_rewriting.py`, `tests/test_retrieval_cli.py` |
| BGE reranking | `tests/test_retrieval_reranking.py` |
| Retrieval orchestration | `tests/test_retrieval_pipeline.py`, `tests/e2e/test_hybrid_retrieval.py` |

# Appendix B — BGE reranker OOM incident

## Symptom and root cause

During a retrieval run, the operating system terminated `raglab-retrieve` when memory usage
exceeded the available RAM. The kernel OOM killer, not an ordinary Python exception, ended the
process.

The original reranker processed 32 query/document pairs as one padded tensor. Sequences reached
864 tokens, the float32 BGE model occupied about 2.2 GiB, and autograd remained active. Peak RSS
reached approximately 14.6 GiB.

Calling `.eval()` was insufficient. It changes training-sensitive layers such as dropout, but it
does **not** disable gradient recording or the autograd graph. Inference correctness and inference
memory are separate controls.

## Implemented correction

`BGEReranker` now:

1. loads tokenizer and model lazily and reuses them;
2. wraps scoring in `torch.inference_mode()`;
3. truncates every pair with `max_length=512`;
4. scores internal microbatches of four;
5. appends scores in stable candidate order.

Reranking remains enabled by default. The fix bounds memory at inference instead of hiding the
defect by reducing `candidate_k`.

## Validation

| Scenario | Result |
| -------- | ------ |
| Corrected BGE | 32 candidates, 32 scores |
| Peak RSS | 2,512,688 KiB, approximately 2.4 GiB |
| Elapsed time | 24.21 seconds |
| Swap / OOM | Zero swap; no new OOM event |
| `--no-rerank` | Approximately 48 MiB and 1.03 seconds |

Figures depend on hardware, kernel, model, and dependency versions. The stable rule is to limit
sequence length, autograd state, and batch size at inference.

`tests/test_retrieval_reranking.py` locks down microbatching, the 512-token limit,
`inference_mode`, order, empty input, and lazy reuse.

## Diagnosis and fallback

```bash
/usr/bin/time -v raglab-retrieve "What causes fault E17?" \
  --collection greenhouse-manuals \
  --candidate-k 32

journalctl -k --since "10 minutes ago" | grep -Ei 'oom|out of memory|killed process'
dmesg --ctime | grep -Ei 'oom|out of memory|killed process'
```

On a machine that cannot accommodate BGE:

```bash
raglab-retrieve "What causes fault E17?" \
  --collection greenhouse-manuals \
  --no-rerank
```

This keeps ANN, BM25, filters, RRF, expansion, and MMR; only BGE is skipped.

## Operational notes

Compose uses the `raglab-paradedb` volume. The former `raglab-postgres` volume is retained but not
migrated or deleted. Reingest documents after moving to the ParadeDB volume; never copy PostgreSQL
data files between images.

```bash
docker compose exec postgres psql -U raglab -d raglab -c \
  "SELECT * FROM collection_stats ORDER BY name;"
```

| Error | Action |
| ----- | ------ |
| PostgreSQL refused | Run `docker compose up -d --wait`; verify `RAGLAB_DSN`. |
| Ollama unavailable | Start `ollama serve`. |
| Embedding model missing | Run `ollama pull qwen3-embedding:0.6b`. |
| Docling missing | Install `python -m pip install -e '.[conversion,tokenizers]'`. |
| Collection mismatch | Reuse its original flags or create a new collection. |
| BGE still too large | Verify the bounded code, then use `--no-rerank`. |

## Next curriculum stages

| Stage | Notebook | Status |
| ----- | -------- | ------ |
| 01 | `01_ingestion_and_indexing.ipynb` | Implemented |
| 02 | `02_retrieval.ipynb` | Implemented |
| 03 | `03_generation.ipynb` | Planned |
| 04 | `04_rag_evaluation.ipynb` | Planned |

Generation should consume `result.content` with `result.citation`, never `embedding_text`.
