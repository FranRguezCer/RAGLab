# RAGLab

RAGLab is a local-first, inspectable laboratory for learning **Retrieval-Augmented Generation
(RAG)** from first principles. RAG retrieves source evidence before a language model answers,
reducing reliance on the model's internal memory and making citations possible. This repository
teaches the four stages that determine whether an answer is trustworthy:

1. **ingestion and indexing** — convert sources into faithful, searchable units; and
2. **hybrid retrieval** — combine semantic and lexical search, then refine the evidence; and
3. **strict generation** — answer only from retrieved evidence and validate every citation; and
4. **evaluation** — measure ingestion, retrieval, and generation before accepting a change.

The components remain explicit. There is no LangChain or LlamaIndex layer hiding conversion,
chunking, ranking, generation, SQL, or failure modes. Ollama runs embeddings, optional query
rewriting, and generation locally; Docling converts complex files locally; PostgreSQL stores the
inspectable artifacts.
Jina Reader is available only as an explicit opt-in for public URLs.

## Learning path

| Step | Start here | What you learn |
| ---- | ---------- | -------------- |
| 1 | This introduction and [architecture](#architecture) | How a source becomes citable evidence |
| 2 | [Chapter 1](#chapter-1--ingestion-and-indexing) | Conversion, chunking, vectors, and atomic storage |
| 3 | `notebooks/01_ingestion_and_indexing.ipynb` | Inspect every ingestion artifact |
| 4 | [Chapter 2](#chapter-2--hybrid-retrieval) | ANN, BM25, RRF, reranking, expansion, and MMR |
| 5 | `notebooks/02_retrieval.ipynb` | Change a query and observe every retrieval stage |
| 6 | [Chapter 3](#chapter-3--strict-rag-generation) | Generate cited answers without dropping evidence |
| 7 | `notebooks/03_generation.ipynb` | Compare single-pass and hierarchical generation |
| 8 | [Chapter 4](#chapter-4--reproducible-rag-evaluation) | Run, inspect, compare, and promote evaluation results |
| 9 | `notebooks/04_rag_evaluation.ipynb` | Interpret deterministic metrics and conservative verdicts |
| 10 | [Appendix A](#appendix-a--test-strategy-and-suite) | Prove each system boundary |

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
    EVIDENCE --> PLAN{"Complete sources fit?"}
    PLAN -->|Yes| SINGLE["Single-pass synthesis"]
    PLAN -->|No| HIER["Hierarchical extraction + synthesis"]
    SINGLE --> VALIDATE["Strict citation validation"]
    HIER --> VALIDATE
    VALIDATE --> ANSWER["JSON answer + sources + original retrieval"]
    ANSWER --> EVAL["Evaluate ingestion + retrieval + generation"]
    EVAL --> DECIDE{"Promote candidate?"}
    DECIDE -. "change and rerun" .-> CHUNK
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
cp env.template .env
docker compose up -d --wait
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
```

Run `ollama serve` in another terminal if Ollama is not already a system service. Compose exposes
PostgreSQL only at `127.0.0.1:5432`.

The four CLI commands automatically load the nearest `.env` from the current directory upward.
Explicit CLI flags take priority over exported environment variables, which take priority over
`.env`; internal defaults remain available when no `.env` exists. Docker Compose reads the same
file. If you change `POSTGRES_DB`, `POSTGRES_USER`, or `POSTGRES_PASSWORD`, update the credentials
and database name inside `RAGLAB_DSN` as well. The Ollama variables configure generation, and
`RAGLAB_EMBEDDING_MODEL` must match the model stored in the selected collection. Vector dimension
is fixed at 1024 by the current collection contract and database schema.

```bash
raglab-ingest data/samples/aster_greenhouse_controller_manual.md \
  --collection greenhouse-manuals

raglab-retrieve "What causes fault E17?" \
  --collection greenhouse-manuals

raglab-generate "What causes fault E17, and how should it be resolved?" \
  --collection greenhouse-manuals

raglab-evaluate run --profile core
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

Only `source` is positional. In a terminal, omitting `--collection` lists the existing collections
and prompts for an existing or new name. Scripts and redirected input must pass `--collection`
explicitly, preventing accidental creation of a `documents` collection. Selecting an existing
collection reuses its stored embedding and chunking contract; incompatible explicit chunk flags
fail before ingestion.

| Parameter | Default | Meaning, range, and tradeoff |
| --------- | ------- | ---------------------------- |
| `source` | Required | Local path or public HTTP(S) URL. Local paths resolve to stable absolute URIs. |
| `--collection` | Interactive prompt; required without a TTY | Existing search boundary or the name of a new collection. |
| `--dsn` | `RAGLAB_DSN`, else `postgresql://raglab:raglab@127.0.0.1:5432/raglab` | PostgreSQL connection string. |
| `--target-tokens` | Stored value; new collection: `512` | Preferred size. Larger chunks add context but reduce precision. |
| `--min-tokens` | Stored value; new collection: `120` | Minimum preferred size. Raising it reduces fragments but can merge ideas. |
| `--max-tokens` | Stored value; new collection: `768` | Upper chunk target. Raising it increases context and embedding cost. |
| `--semantic-percentile` | Stored value; new collection: `90` | Lower values split more often; higher values require stronger evidence. |
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

# Chapter 3 — Strict RAG generation

Generation completes the path from source to answer. `raglab-generate` performs one typed
`RetrievalPipeline` call, gives the resulting evidence to a local LLM, validates the model's JSON,
and returns the **original retrieval response** alongside the answer. It never shells out to
`raglab-retrieve` and never asks the model to reconstruct retrieval metadata.

## The grounding contract

RAGLab treats grounding as a checked boundary, not just a prompt instruction.

The Ollama request sends trusted policy through the dedicated `system` field. The question,
retrieval metadata, and source content remain untrusted user data in `prompt`; source text cannot
replace the system policy merely by containing instruction-like prose.

| Contract | Enforced behavior |
| -------- | ----------------- |
| Evidence | The model may use only complete `RetrievalResult.content` values. |
| Inline citations | Supported claims use stable IDs such as `[S1]` and `[S2]`. |
| Structured sources | `sources` are derived from validated inline citation IDs in first-appearance order. |
| Known sources | Every ID must refer to a result from this retrieval call. |
| Non-abstaining answer | At least one valid citation is required. |
| Insufficient evidence | The model must set `abstained: true`; an empty retrieval abstains without calling the LLM. |

The model generates only `answer` and `abstained`; it does not regenerate a parallel source-ID
list. Unknown inline IDs, invalid JSON, or an uncited non-abstaining answer fail closed with
`GenerationError`. Validation proves traceability; it does not claim that an LLM can independently
prove the semantic truth of every sentence.

The response is JSON by default and contains:

```json
{
  "answer": "Fault E17 indicates ... [S1]",
  "abstained": false,
  "sources": [
    {
      "id": "S1",
      "retrieval_result_id": "...",
      "document_id": "...",
      "citation": {"source_name": "manual.md", "start_line": 42}
    }
  ],
  "retrieval": {"query": "...", "results": []},
  "strategy": "single_pass",
  "source_shortfall": false,
  "minimum_sources": 5,
  "source_count": 5,
  "metrics": {"model_calls": 1, "estimated_prompt_tokens": 1800}
}
```

The abbreviated nested objects above show the shape, not a literal complete response. The real
`retrieval` value preserves queries, filters, complete results, citations, parent ranges, matched
children, and ranking traces.

## Keep every available source

Generation requests at least five final results even if a lower `--top-k` is supplied. When five
or more results exist, at least five complete candidates reach the generation planner. When the
filtered collection contains fewer, RAGLab uses every available result and sets
`source_shortfall: true`; it does not invent filler sources or abstain merely because the count is
below five.

```mermaid
flowchart TD
    QUERY["One question"] --> RETRIEVE["Typed RetrievalPipeline"]
    RETRIEVE --> CHECK["Validate collection model + dimension"]
    CHECK --> SOURCES["Stable S1..Sn over complete results"]
    SOURCES --> FIT{"System prompt + all sources + output reserve fit num_ctx?"}
    FIT -->|Yes| SINGLE["Single-pass cited synthesis"]
    FIT -->|No| BATCH["Sequential complete-source batches"]
    BATCH --> FACTS["Validate source-linked facts per batch"]
    FACTS --> SYNTH["Final cited synthesis"]
    SINGLE --> GUARD["Validate JSON and citations"]
    SYNTH --> GUARD
    GUARD --> JSON["Answer + sources + strategy + original retrieval"]
```

`single_pass` is the shortest path. Its conservative planner budgets the separate system policy,
user prompt, JSON Schema, Qwen `/no_think` control when applicable, output reserve, and a template
margin—not just source bytes. If the complete invocation would exceed `num_ctx`, `hierarchical`
groups complete candidates into sequential extraction calls, validates that every fact refers
only to its batch IDs, and synthesizes from those source-linked facts. Source IDs stay stable end
to end.

If an extraction exhausts `num_predict`, the fallback recursively splits that batch and retries;
it never silently accepts a partial response. If the first fact layer still cannot fit final
synthesis, bounded reduction rounds compress source-linked fact groups and recheck the complete
invocation budget at every level. Lack of progress, a single oversized source/fact, or final
synthesis exhaustion fails explicitly. The planner does **not** truncate evidence to force it
through. A length-limited single-pass attempt enters the same hierarchical path, and the failed
attempt remains visible in call/token metrics.

## Validated 8 GB local profile

The default was exercised on an NVIDIA RTX 3060 Ti with 8192 MiB VRAM and 15 GiB system RAM.

| Component | Default placement and budget |
| --------- | ---------------------------- |
| Query embedding | `qwen3-embedding:0.6b`, Ollama `num_gpu=0`, `num_ctx=4096` |
| Generation | `qwen3:4b` on GPU, `num_ctx=12288`, `num_predict=512` |
| Calls | One at a time; `parallelism=1` |
| Residency | Positive `keep_alive` TTL, default `5m`; at most these two Ollama models resident |

The measured CPU-embedding plus 12K-generation profile used about 5804 MiB VRAM, left about
2221 MiB free, used no swap, and unloaded both models after their TTL. These figures are a
validated baseline, not a universal guarantee: drivers, Ollama versions, context contents, and
other GPU processes change memory use.

Do not use `keep_alive=0` for this workflow. Ollama 0.13.2 under the tested WSL environment
reported no resident model while VRAM remained occupied until the service restarted. A positive
TTL produced reliable unloading. When diagnosing memory, inspect `nvidia-smi` as well as
`ollama ps`.

## Embedding compatibility comes before generation

Collections persist their embedding model and dimension. `raglab-generate` checks both before it
embeds the query. `--embedding-model` is therefore an experiment control, not permission to query
vectors created in a different vector space. A mismatch fails with an instruction to reindex into
a compatible collection. The current PostgreSQL schema remains fixed at 1024 dimensions.

## `raglab-generate` reference

Only `query` is required. Retrieval flags retain their Chapter 2 meaning; generation raises
`top_k` and `candidate_k` as needed to honor `--minimum-sources`.

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `query` | Required | One question; this version is not a persistent chat session. |
| `--collection` | `documents` | Indexed evidence boundary. |
| `--dsn` | `RAGLAB_DSN`, else Compose DSN | Read-only retrieval connection. |
| `--filter` | None | Repeatable typed prefilter shared by ANN, BM25, and expansion. |
| `--candidate-k` | `50` | Per-channel retrieval candidates; raised to the final source minimum if needed. |
| `--top-k` | `5` | Requested final results; raised to `minimum-sources` if lower. |
| `--minimum-sources` | `5` | Minimum requested when the collection and filters can supply it. |
| `--ef-search` | `100` | HNSW search breadth. |
| `--exact` | Disabled | Exact semantic search for diagnostics. |
| `--rewrite`, `--expansions` | Disabled, `0` | Optional standalone query and up to two expansions. |
| `--history-file` | None | JSON strings or message objects used only to disambiguate this one retrieval query. |
| `--no-rerank`, `--no-mmr`, `--no-small-to-big` | Disabled | Disable one retrieval refinement stage. |
| `--mmr-lambda` | `0.7` | Relevance/diversity balance. |
| `--parent-max-tokens` | `1500` | Complete dynamic-parent budget. |
| `--model` | `RAGLAB_GENERATION_MODEL`, else `qwen3:4b` | Ollama generation model. |
| `--embedding-model` | `RAGLAB_EMBEDDING_MODEL`, else `qwen3-embedding:0.6b` | Must match the collection contract. |
| `--ollama-base-url` | `RAGLAB_OLLAMA_BASE_URL`, else local Ollama | Ollama API root. |
| `--num-ctx` | `RAGLAB_NUM_CTX`, else `12288` | Generation context window. |
| `--num-predict` | `512` | Maximum output tokens reserved by the planner. |
| `--keep-alive` | `RAGLAB_KEEP_ALIVE`, else `5m` | Positive Ollama residency TTL. |

The generation adapter sends the grounding policy through Ollama's `system` field, uses
`think=false`, supplies an explicit JSON Schema, adds `/no_think` for Qwen models, and rejects
responses whose `done_reason` is not `stop`. A `length` termination is a typed signal used by the
hierarchical split/reduction path rather than malformed JSON being accepted.

```bash
raglab-generate "What causes fault E17, and what action resolves it?" \
  --collection greenhouse-manuals

raglab-generate "Which one requires replacing the sensor?" \
  --collection greenhouse-manuals \
  --history-file ./history.json \
  --rewrite

# Model experiment: only valid for a collection indexed with the same embedding model.
RAGLAB_GENERATION_MODEL=qwen3:4b \
RAGLAB_EMBEDDING_MODEL=qwen3-embedding:0.6b \
RAGLAB_NUM_CTX=12288 \
RAGLAB_KEEP_ALIVE=5m \
raglab-generate "Summarize the recovery procedure" \
  --collection greenhouse-manuals \
  --no-rerank
```

## `03_generation.ipynb`

The notebook uses package contracts and small deterministic adapters to run hermetically by
default. It demonstrates single-pass generation, citation rejection, source shortfall, and forced
hierarchical fallback without duplicating pipeline logic.

Enable the final live section only after ingesting a compatible collection. It reuses one
`GenerationPipeline` for a grounded E17 question and a Wi-Fi question that the manual cannot
answer.

For an interactive notebook session, change the assignment in the final code cell:

```python
RAGLAB_RUN_GENERATION_NOTEBOOK = "1"
```

For command-line execution, enable the same section through the environment:

```bash
export RAGLAB_RUN_GENERATION_NOTEBOOK=1
export RAGLAB_GENERATION_COLLECTION=greenhouse-manuals
export RAGLAB_GENERATION_QUERY='What causes fault E17, and how should it be resolved?'
export RAGLAB_GENERATION_UNANSWERABLE_QUERY='According to the Aster manual, what is the default Wi-Fi password? If the sources do not state one, abstain.'
jupyter execute notebooks/03_generation.ipynb \
  --output /tmp/raglab-generation-live.ipynb
```

The live output includes each question, answer, abstention flag, strategy, citations, and the full
content of every retrieved result. The assertions require a cited E17 answer and abstention for the
Wi-Fi question. Citation validation proves that source IDs are traceable; inspect the displayed
evidence to decide whether it semantically supports each claim.

Without either switch, the full notebook remains executable without PostgreSQL or Ollama. If
`raglab-generate` is missing after pulling a version that added the command, refresh the existing
editable install without resolving project dependencies again:

```bash
python -m pip install -e . --no-deps
raglab-generate --help
```

# Chapter 4 — Reproducible RAG evaluation

Evaluation closes the loop: **run → inspect → promote**. A candidate is evidence, not a new
baseline. Inspect its hard failures, per-case results, and corpus identity before explicitly
promoting it.

## Quick path

```bash
# 1. Rebuild the protected core collection and create a candidate.
raglab-evaluate run --profile core

# Use the run_id printed above.
RUN_JSON=artifacts/evaluation/20260820T123456.000000Z.json

# 2. Inspect the JSON for machines and the Markdown summary for humans.
cat "${RUN_JSON%.json}.md"

# 3. Promote only after the candidate is understood and accepted.
raglab-evaluate baseline promote "$RUN_JSON"
```

After a code or configuration change, run the profile again and compare it with the approved
baseline:

```bash
raglab-evaluate run --profile core
CANDIDATE=artifacts/evaluation/20260820T133000.000000Z.json
raglab-evaluate compare "$CANDIDATE" \
  --baseline artifacts/evaluation/baseline.json
```

The comparison reports `improved`, `regressed`, `mixed`, or `no_clear_change` across independent
quality axes. It deliberately does not hide tradeoffs inside one weighted score.

## What an evaluation index means

Here, an **index** means the searchable evaluation corpus: its source documents, chunks,
embeddings, and PostgreSQL search structures as one versioned retrieval input. A normal core run
reconstructs dedicated collections whose names start with `raglab-eval-`, so ingestion and
chunking changes are measured too. Personal collections are outside that protected namespace and
are never reset by the evaluator.

This is NOT PostgreSQL `REINDEX`. `REINDEX` rebuilds a database index structure from rows that
already exist; it does not reconvert sources, rechunk documents, regenerate embeddings, or change
the evaluation corpus. Use `--reuse-index` only when isolating a downstream retrieval or
generation change:

```bash
raglab-evaluate run --profile core --reuse-index
```

A reused-index run is marked partial and cannot be promoted, because it did not verify the whole
pipeline.

## Profiles

| Profile | Purpose | External requirements |
| ------- | ------- | --------------------- |
| `core` | Authoritative, reproducible benchmark over tracked fixtures; judge disabled by default | PostgreSQL, Ollama, and BGE |
| `live` | Optional reality check over Attention and one source from each Raspberry Pi domain | Core services, an absolute PDF path, and network access |

The core manifest uses semantic percentile `85`, four Aster questions and chunk controls, three
short technical sources, distractor and abstention cases, multi-evidence retrieval, and two
multi-turn cases. Each case runs approximate and exact retrieval once each, then generates three
times. The core without a judge is designed to finish in less than 15 minutes.

The live profile compares domain-specific Raspberry Pi collections with an aggregate collection.
It hashes every source and rejects a changed live source instead of pretending it is comparable:

```bash
curl -L https://arxiv.org/pdf/1706.03762 -o attention.pdf
export RAGLAB_ATTENTION_PDF="$(realpath attention.pdf)"
raglab-evaluate run --profile live
```

`RAGLAB_ATTENTION_PDF` must resolve to an absolute local path. Live inputs are mutable by nature;
their recorded hashes are part of the evidence.

## Read the result by axis

| Boundary | Recorded evidence |
| -------- | ----------------- |
| Ingestion | Must-separate and must-keep checks, document/chunk counts, token distribution, latency |
| Retrieval | Hit@1/3/5, Recall@5, MRR, nDCG@5, exact-versus-HNSW agreement, latency |
| Generation | Required facts, abstention, source/citation checks, stability across 3 runs, tokens, calls, latency |
| Operation | Hard and advisory errors plus p50/p95 retrieval and generation latency |

Ground truth names evidence with versioned `source_id` values and normalized text anchors, never
ephemeral PostgreSQL UUIDs. Quality comparison requires the same run schema, profile, corpus
fingerprint, and configuration fingerprint. Latency is shown only when hardware fingerprints
match.

Hard failures cover objective contract breaches such as missing expected evidence or failed
fact, abstention, or citation checks. Numeric deltas remain descriptive until tolerances are
calibrated. An optional LLM judge is advisory: it runs after deterministic answers are saved and
the generator is unloaded; judge failure or OOM does not invalidate the deterministic core.
The judge model must differ from the generation model.

```bash
raglab-evaluate run --profile core --judge-model <different-model>
```

RAGLab does not install or pin a second judge model in this version.

## Baseline rules and CLI reference

A baseline is an explicitly approved run, not merely the previous run. Promotion accepts only a
complete, full run from a clean Git worktree with no hard failures.

| Command | Meaning |
| ------- | ------- |
| `raglab-evaluate run --profile core` | Rebuild the core evaluation collection and write versioned JSON and Markdown artifacts. |
| `raglab-evaluate run --profile live` | Build the mutable-source profile after validating source hashes. |
| `raglab-evaluate run --profile core --reuse-index` | Skip rebuilding the collection; mark the candidate partial and non-promotable. |
| `raglab-evaluate run --profile core --judge-model MODEL` | Add non-authoritative judge observations with a model different from the generator. |
| `raglab-evaluate compare CANDIDATE [--baseline BASELINE]` | Compare compatible quality axes; default baseline is `artifacts/evaluation/baseline.json`. |
| `raglab-evaluate baseline promote RUN` | Atomically replace the approved baseline after promotion checks. |
| global `--artifact-dir PATH` | Write/read artifacts somewhere other than `artifacts/evaluation/`; place it before the subcommand. |

Every run artifact records the schema version, Git commit and dirty state, model names, hardware,
configuration and corpus fingerprints, per-source hashes, results, timings, and errors. The JSON
schema is exported as `raglab.evaluation.RUN_JSON_SCHEMA`; application integrations can import
`EvaluationApplication` and optional judges implement `EvaluationJudge`.

## `04_rag_evaluation.ipynb`

The fourth laboratory imports the same `EvaluationApplication` and manifest loader used by the
CLI. It does not shell out. By default it runs a complete hermetic baseline/candidate lesson,
shows ground truth and multi-turn cases, computes deterministic metrics, and explains the
conservative verdict without PostgreSQL or Ollama.

Execute the tracked output-free notebook into `/tmp`:

```bash
jupyter execute notebooks/04_rag_evaluation.ipynb \
  --output /tmp/raglab-evaluation.ipynb
```

Enable its final real core section explicitly. The resulting run JSON and Markdown remain outside
the repository under `/tmp/raglab-evaluation-artifacts/`:

```bash
RAGLAB_RUN_EVALUATION_NOTEBOOK=1 \
jupyter execute notebooks/04_rag_evaluation.ipynb \
  --output /tmp/raglab-evaluation-live.ipynb
```

# Appendix A — Test strategy and suite

The pyramid keeps algorithmic feedback fast and reserves real converters, databases, models, and
PDFs for explicit boundaries. Environment-gated integration and E2E tests skip when their
services or opt-in variables are absent.

## Levels

| Level | What it proves | Requirements |
| ----- | -------------- | ------------ |
| Unit | Conversion, parsing, chunking, embeddings, CLIs, ranking, reranking, SQL, orchestration | None |
| Docling integration | Real non-text conversion and provenance | Conversion extras |
| PostgreSQL integration | Migrations, atomic replacement, vectors, filters, BM25, retrieval SQL | Disposable ParadeDB |
| PDF E2E | PDF → Docling → Ollama → PostgreSQL | PDF, Docling, Ollama, PostgreSQL |
| Retrieval E2E | Real ANN + BM25 + BGE | ParadeDB, Ollama, BGE model |
| Evaluation integration | Protected collection reset and isolation from personal collections | Disposable ParadeDB |
| Evaluation E2E | Minimal native ingestion, retrieval, and generation run | ParadeDB, Ollama, BGE model |
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
pytest -m integration tests/integration/test_evaluation_postgres.py
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

RAG evaluation E2E:

```bash
export RAGLAB_RUN_EVALUATION_E2E=1
export RAGLAB_TEST_DSN='postgresql://raglab:raglab@127.0.0.1:5432/raglab'
pytest -m e2e tests/e2e/test_rag_evaluation.py
```

Always write executed notebooks outside the repository:

```bash
jupyter execute notebooks/01_ingestion_and_indexing.ipynb \
  --output /tmp/raglab-ingestion.ipynb
jupyter execute notebooks/02_retrieval.ipynb \
  --output /tmp/raglab-retrieval.ipynb
jupyter execute notebooks/03_generation.ipynb \
  --output /tmp/raglab-generation.ipynb
jupyter execute notebooks/04_rag_evaluation.ipynb \
  --output /tmp/raglab-evaluation.ipynb
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
| Generation, citations, fallback, and CLI | `tests/test_generation.py`, `tests/test_generation_ollama.py`, `tests/test_generation_cli.py` |
| Evaluation contracts, metrics, application, and CLI | `tests/test_evaluation.py`, `tests/test_evaluation_cli.py` |
| Evaluation storage isolation and native E2E | `tests/integration/test_evaluation_postgres.py`, `tests/e2e/test_rag_evaluation.py` |
| Output-free executable notebooks | `tests/test_evaluation_notebook.py` |

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
| 03 | `03_generation.ipynb` | Implemented |
| 04 | `04_rag_evaluation.ipynb` | Implemented |

Generation consumes `result.content` with `result.citation`, never `embedding_text`.
