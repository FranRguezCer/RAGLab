# RAGLab learning guide

This guide explains how RAGLab turns source material into measured, citable answers. Follow the
four stages in order for the teaching path, or jump to a CLI reference when running an experiment.
All commands assume the repository root and the environment from the [quick start](../README.md#quick-start).

## Learning path

| Stage | Read | Run |
| --- | --- | --- |
| 1. Ingestion | [Ingestion and indexing](#1-ingestion-and-indexing) | [`01_ingestion_and_indexing.ipynb`](../notebooks/01_ingestion_and_indexing.ipynb) |
| 2. Retrieval | [Hybrid retrieval](#2-hybrid-retrieval) | [`02_retrieval.ipynb`](../notebooks/02_retrieval.ipynb) |
| 3. Generation | [Strict RAG generation](#3-strict-rag-generation) | [`03_generation.ipynb`](../notebooks/03_generation.ipynb) |
| 4. Evaluation | [Transparent evaluation](#4-transparent-evaluation) | [`04_rag_evaluation.ipynb`](../notebooks/04_rag_evaluation.ipynb) |

The tracked fictional **Aster Greenhouse Controller Manual** provides a small corpus with known
boundaries and answer anchors. It keeps the core notebook checkpoints deterministic and makes
failures easy to inspect.

## 1. Ingestion and indexing

Ingestion turns heterogeneous sources into one explicit, reproducible retrieval contract. Think
of each chunk as a library card: it must carry enough context to be found, while its quotation
remains faithful to the source.

### Ingestion pipeline

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

#### Canonical Markdown and provenance

Native `.md`, `.markdown`, and `.txt` files use a lossless direct converter. PDF, HTML, DOCX, ODT,
ODS, ODP, and other supported complex formats use local Docling. Public URLs are downloaded and
converted locally by default; the optional Jina path is explicit. Both routes enforce the
[public-source security policy](appendix.md#source-ingestion).

Every successful path emits canonical Markdown plus source identity, converter version, content
hash, and observable line/page provenance. RAGLab never invents page 1 for a source without
meaningful pagination. Optional location enrichment may be `partial` or `unavailable`; conversion,
parsing, embedding, and storage failures remain hard failures.

#### AST, structure, and semantic boundaries

The Markdown parser builds an AST so headings, paragraphs, lists, tables, and code blocks remain
recognizable. Parsing succeeds only when it produces at least one non-empty block; otherwise it
raises `ParsingError`. A heading-only document is valid Markdown and therefore can satisfy this
parser contract, but it still has no body text to retrieve.

These structural units are the first chunk candidates. Maximum-size, heading, and
semantic cuts are hard boundaries, even when either side is smaller than `min_tokens`. Target-size
cuts are flexible: a small trailing group may merge backward only within the same heading and
without exceeding `max_tokens`. This makes `min_tokens` a preferred size rather than permission to
mix unrelated evidence. Temporary embeddings compare neighbouring material; a positive distance
at or above the configured percentile becomes a semantic boundary. These temporary vectors are
discarded.

The chunker preserves faithful `content`. It builds separate `embedding_text` from the title,
heading breadcrumbs, and faithful chunk. That contextual text creates the final vector, while
retrieval returns only `content`. Chunking must produce at least one searchable chunk or it raises
`ChunkingError`. The ingestion pipeline checks this invariant again for injected parser or chunker
implementations before consulting idempotency state, requesting final embeddings, or writing to
PostgreSQL. A document that cannot become searchable is never reported as indexed.

#### Final embeddings and atomic storage

Ollama's `qwen3-embedding:0.6b` produces final 1024-dimensional vectors. Cosine distance compares
their direction rather than magnitude. Every embedding completes before the transaction opens.

PostgreSQL inserts a new document or updates the current UUID, deletes old chunks, inserts the
complete replacement, and commits. A failure rolls back the whole replacement.

### Collections, fingerprints, and idempotency

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

### `raglab-ingest` reference

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
| `--min-tokens` | Stored value; new collection: `120` | Preferred minimum. It may merge a small target-size tail, but never across heading, semantic, or maximum-size boundaries. |
| `--max-tokens` | Stored value; new collection: `768` | Hard upper bound. Raising it increases context and embedding cost. |
| `--semantic-percentile` | Stored value; new collection: `90` | Lower values split more often; higher values require stronger evidence. |
| internal overlap | `0` | Fixed default; avoids duplicated evidence. |
| `--use-jina` | Off | Sends a public URL to Jina Reader; never automatic and uses the same credentials, DNS, and public-address checks as local URL conversion. |
| `-h`, `--help` | N/A | Print parser help and exit. |

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

### Ingestion notebooks

#### `01_ingestion_and_indexing.ipynb`

This 10–15 minute lab follows three checkpoints: source → canonical Markdown, Markdown → AST,
and AST → indexable chunks. It uses the real converter, parser, and chunker against the tracked
Aster manual, but needs neither PostgreSQL nor Ollama.

```bash
jupyter execute notebooks/01_ingestion_and_indexing.ipynb \
  --output /tmp/raglab-ingestion-lab.ipynb
```

Use the service-backed ingestion CLI and the advanced benchmark when you are ready to inspect
embeddings, PostgreSQL storage, HNSW, or a real PDF.

#### `benchmark_ingestion_hyperparameters.ipynb`

This unnumbered workbench compares chunking profiles and HNSW behavior, records PostgreSQL's
chosen plan, and measures recall@K, latency, build time, and index size. Imagine HNSW as a
multilayer network of shortcuts: upper layers approach a promising region and lower layers refine
the neighbourhood.

The migration fixes `m=16` and `ef_construction=64`. These are physical index-build settings, not
per-collection ingestion flags.

### Optional real PDF: *Attention Is All You Need*

Keep the short notebook hermetic. To ingest the paper through the real service boundaries, use
the CLI instead:

```bash
curl -L https://arxiv.org/pdf/1706.03762 -o attention.pdf
raglab-ingest attention.pdf --collection attention-paper
```

The Aster corpus remains the controlled fixture; the paper is an optional service-backed
experiment for real PDF conversion and indexing.

## 2. Hybrid retrieval

Retrieval asks two questions in parallel: “which chunks mean something similar?” and “which chunks
contain the discriminating words?” Semantic search answers the first; BM25 answers the second.

### Retrieval pipeline, in order

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

### Exact search, ANN, HNSW, and PostgreSQL

Exact search computes cosine distance against every eligible vector and is the diagnostic
reference. ANN visits promising HNSW regions, reducing latency at the cost of possibly missing an
exact neighbour. `recall@K` measures agreement with exact top K.

`ef_search` controls HNSW query breadth. Higher values generally improve recall and cost latency.
RAGLab uses `100`; pgvector defaults to `40`.

An index-enabled query does **not** guarantee an HNSW scan. PostgreSQL may prefer exact scan and
sort for a small or selective collection; `EXPLAIN` is authoritative. All collections share one
physical HNSW index, so post-index filtering can hurt recall for a small collection in a large
table. Raising `ef_search` may help but does not remove that tradeoff.

### Filters and history

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

### `raglab-retrieve` reference

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
| `--exact` | Off | Force exact semantic diagnostics; slower at scale. |
| `--rewrite` | Off | Ask local `qwen3:4b` for a standalone query. |
| `--history-file` | None | JSON conversation history; also enables rewriting. |
| `--expansions` | `0` | Additional variants, `0..2`; each adds ANN and BM25 work. |
| `--no-rerank` | Off | BGE stays enabled. This fallback reduces memory and latency. |
| `--no-mmr` | Off | MMR stays enabled; disabling may return redundant parents. |
| `--mmr-lambda` | `0.7` | `0..1`: `1` favors relevance, `0` diversity. |
| `--no-small-to-big` | Off | Expansion stays enabled; disabling returns matched chunks. |
| `--parent-max-tokens` | `1500` | Positive parent budget; more adds context and prompt cost. |
| `-h`, `--help` | N/A | Print parser help and exit. |

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

### `02_retrieval.ipynb`

This 10–15 minute lab follows three checkpoints: compare semantic and BM25 rankings, fuse them
with RRF, and inspect the final citable evidence. It uses RAGLab's real ranking and retrieval
contracts with deterministic data, so PostgreSQL, Ollama, and BGE are not required.

```bash
jupyter execute notebooks/02_retrieval.ipynb \
  --output /tmp/raglab-retrieval-lab.ipynb
```

Use `raglab-retrieve` for service-backed experiments with rewriting, reranking, filters,
small-to-big expansion, and MMR after a collection has been indexed.

## 3. Strict RAG generation

Generation completes the path from source to answer. `raglab-generate` performs one typed
`RetrievalPipeline` call, gives the resulting evidence to a local LLM, validates the model's JSON,
and returns the **original retrieval response** alongside the answer. It never shells out to
`raglab-retrieve` and never asks the model to reconstruct retrieval metadata.

### The grounding contract

RAGLab treats grounding as a checked boundary, not just a prompt instruction.

The Ollama request sends trusted policy through the dedicated `system` field. The question,
retrieval metadata, and source content remain untrusted user data in `prompt`; source text cannot
replace the system policy merely by containing instruction-like prose.

| Contract | Enforced behavior |
| -------- | ----------------- |
| Evidence | The model may use only complete `RetrievalResult.content` values. |
| Structured source IDs | RAGLab derives ordered `source_ids` aliases such as `S1` and `S2` from selected, verified facts. |
| Resolved sources | `sources` contains the retrieval identity and citation metadata for those IDs in the same order. |
| Known sources | Each selected fact must name a source included in the same selection batch. |
| Non-abstaining answer | At least one validated fact is required before final synthesis. |
| Insufficient evidence | Empty retrieval abstains without calling the LLM; an empty or fully rejected selection abstains without synthesis. |

For non-empty retrieval, the model first sees the question and the complete ranked sources that
fit together, then returns `source_id` + `claim` + `evidence_quote` facts. This selector is the
relevance authority: there is no lexical-overlap filter after it. RAGLab validates the source ID,
recovers a quote only when whitespace normalization produces one unique original span, and checks
quote-to-claim entailment with the pinned NLI model. A second model call synthesizes only from the
verified facts. The public `source_ids` tuple is derived from retained fact lineage; the model never
declares attribution in the synthesis response. Legacy `[S#]` markers are removed from `answer`
and never decide attribution.

`sources` is built from the derived tuple and preserves each result's retrieval ID, document ID,
and full citation metadata. Consumers can inspect compact aliases without losing provenance.

```bash
raglab-generate "What causes fault E17, and how should it be resolved?" \
  --collection greenhouse-manuals
```

The response is JSON by default and contains:

```json
{
  "answer": "Fault E17 indicates ...",
  "abstained": false,
  "source_ids": ["S1"],
  "sources": [
    {
      "id": "S1",
      "retrieval_result_id": "...",
      "document_id": "...",
      "citation": {"source_name": "manual.md", "start_line": 42}
    }
  ],
  "retrieval": {"query": "...", "results": []},
  "strategy": "hierarchical",
  "source_shortfall": false,
  "minimum_sources": 5,
  "source_count": 5,
  "metrics": {
    "model_calls": 2,
    "selection_calls": 1,
    "facts_invalid_quotes": 0,
    "facts_nli_rejected": 0
  }
}
```

The abbreviated nested objects above show the shape, not a literal complete response. The real
`retrieval` value preserves queries, filters, complete results, citations, parent ranges, matched
children, and ranking traces.

### Keep every available source

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
    SOURCES --> SELECT["Select query-relevant evidence in bounded batches"]
    SELECT --> FACTS["Validate source IDs, quotes, and NLI"]
    FACTS --> SYNTH["Final structured synthesis"]
    SYNTH --> GUARD["Validate answer JSON"]
    GUARD --> DERIVE["Derive source IDs from fact lineage"]
    DERIVE --> JSON["Answer + source_ids + sources + original retrieval"]
```

`single_pass` is reserved for the no-evidence abstention, which makes no model call. Every
non-empty retrieval follows the hierarchical path. The normal path uses one selection call over
all complete sources and one synthesis call. When the sources do not fit together, greedy batches
preserve their ranking and process every source; a source that cannot fit alone fails explicitly.
If verified facts cannot fit synthesis, bounded reduction compresses source-linked groups while
preserving the full source-ID lineage. The planner never truncates or silently drops evidence.

### Embedding compatibility comes before generation

Collections persist their embedding model and dimension. `raglab-generate` checks both before it
embeds the query. `--embedding-model` is therefore an experiment control, not permission to query
vectors created in a different vector space. A mismatch fails with an instruction to reindex into
a compatible collection. The current PostgreSQL schema remains fixed at 1024 dimensions.

### `raglab-generate` reference

Only `query` is required. Retrieval flags retain their Section 2 meaning; generation raises
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
| `--exact` | Off | Exact semantic search for diagnostics. |
| `--rewrite`, `--expansions` | Off, `0` | Optional standalone query and up to two expansions. |
| `--history-file` | None | JSON strings or message objects used only to disambiguate this one retrieval query. |
| `--no-rerank`, `--no-mmr`, `--no-small-to-big` | Off | Disable one retrieval refinement stage. |
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

Generation requires the pinned local NLI checkpoint; there is no unverified fallback:

```bash
python -m pip install -e '.[generation]'
hf download tasksource/deberta-small-long-nli \
  --revision 9a77395d4d3751be9e2a69c4ae318491d9b3fffb
```

Selection returns atomic `source_id` + `claim` + `evidence_quote` facts, with at most two facts per
source. The pipeline rejects unknown sources and nonexistent or ambiguous quotes, permits only
unique whitespace-only quote recovery, verifies quote-to-claim entailment, and assigns IDs to the
surviving facts. If none survive, generation abstains without synthesis or citations. Synthesis
returns answer units bound to those fact IDs plus an exact list of unused IDs; every final unit is
verified again against its original evidence. `source_ids` and `sources` therefore represent only
evidence actually used in the answer. Metrics separate selection calls, invalid quotes, NLI
rejections, accepted facts, and used facts while `model_calls` remains the total real call count.

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

### `03_generation.ipynb`

This 10–15 minute lab follows three checkpoints: create typed evidence, verify an exact-quote
claim and evidence-bound answer unit, then reject an invented policy and abstain without citations.
A deterministic verifier drives the real `GenerationPipeline`, so the main path needs neither the
NLI checkpoint, Ollama, nor PostgreSQL.

```bash
jupyter execute notebooks/03_generation.ipynb \
  --output /tmp/raglab-generation-lab.ipynb
```

Use `raglab-generate` for live local generation, source-shortfall experiments, and hierarchical
synthesis after a compatible collection has been indexed.

The service-backed CLI output includes the answer, abstention flag, strategy, used-only citations,
full retrieved evidence, and fact lifecycle metrics.

## 4. Transparent evaluation

Evaluation starts with a small, inspectable truth set—not a score whose meaning is hidden. The
versioned Aster dataset contains four questions, their expected facts and acceptable wording,
the evidence that should be retrieved, and phrases that must never appear. `raglab.evaluation`
uses only the Python standard library and keeps retrieval and generation scorecards separate.

### The two scorecards

For the first `k` retrieved results, let `relevant@k` be the expected evidence IDs found in those
positions and let `relevant` be every expected evidence ID for the case:

```text
Precision@k = |relevant@k| / k
Recall@k    = |relevant@k| / |relevant|
MRR@k       = 1 / position of the first relevant result, or 0 when none appears
```

Generation checks facts independently of retrieval ranking:

```text
fact coverage              = expected facts expressed / expected facts
grounded fact coverage     = expressed facts backed by a valid cited source / expected facts
citation precision         = valid cited sources / cited sources
abstention accuracy        = correct answer-or-abstain outcomes / cases
```

Every result also lists found and missing facts, evidence positions, cited sources, forbidden
phrases, and the numerator and denominator behind each value. Aggregate means are useful for
orientation, but they never replace per-case traces and there is deliberately no weighted global
score.

### Run the four controlled cases

`load_cases` validates `data/evaluation/aster_greenhouse_controller_v1.json`. The response
adapters accept the production `RetrievalResponse` and `GenerationResponse` contracts, while the
evaluation functions remain deterministic and service-free. `EvaluationApplication` is the
separate application boundary: it runs every case through a configured `GenerationPipeline` and
returns the full responses, per-case traces, and the same `EvaluationReport`. The CLI and notebook
share this boundary instead of rebuilding PostgreSQL and Ollama orchestration themselves.

Run the live application directly from the CLI:

```bash
raglab-evaluate data/evaluation/aster_greenhouse_controller_v1.json \
  --collection greenhouse-manuals \
  --output /tmp/raglab-evaluation.json
```

Or integrate the same application boundary in Python:

```python
from raglab.evaluation import load_cases
from raglab.evaluation_application import create_live_evaluation_application

application = create_live_evaluation_application()
run = application.run(load_cases("data/evaluation/aster_greenhouse_controller_v1.json"))
print(run.report.generation_summary)
```

### Evaluate the local demo corpus

The Raspberry Pi dataset complements the four pedagogical Aster cases with six local-demo cases: one
answerable question and one expected abstention for each public collection. Every case declares
`expected_outcome: answer | abstain` and its own collection, so a single run exercises the same
boundaries exposed by the local demo.

```bash
raglab-evaluate data/evaluation/raspberry_pi_demo_v1.json \
  --output artifacts/evaluation/raspberry_pi_demo_v1.json

raglab-evaluation-gate artifacts/evaluation/raspberry_pi_demo_v1.json \
  data/evaluation/baselines/raspberry_pi_demo_v1.json
```

The versioned dataset and corpus receipt preserve the expected cases and source hashes. The run
records the dataset hash, retrieval and generation settings, pinned BGE and NLI revisions, build
SHA, duration, responses, and per-case traces. The terminal shows a short scorecard while
`--output` atomically preserves the complete run.
The local gate fails when a case emits a forbidden phrase, cites an invalid source, misses an
expected abstention, or regresses a baseline metric by more than `0.05`.

### `raglab-evaluate` reference

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `dataset` | Aster v1 | Versioned JSON evaluation cases. |
| `--collection` | `documents` | Fallback collection for cases that do not declare one. |
| `--output` | None | Atomically persist the complete JSON run for inspection or gating. |
| `--dsn` | `RAGLAB_DSN`, else Compose DSN | PostgreSQL/ParadeDB connection string. |
| `--candidate-k`, `--top-k` | `50`, `3` | Retrieval candidate pool and evaluated result count. |
| `--minimum-sources` | `1` | Requested evidence-source floor for generation. |
| `--exact`, `--ef-search` | Off, `100` | Exact semantic search or HNSW breadth. |
| `--rewrite`, `--expansions` | Off, `0` | Optional query rewrite and expansion. |
| `--no-rerank`, `--no-mmr`, `--no-small-to-big` | Off | Turn off one retrieval refinement stage. |
| `--model`, `--embedding-model` | Environment or pinned defaults | Generation and compatible embedding models. |
| `--ollama-base-url` | Local Ollama | Ollama service URL. |
| `--num-ctx`, `--num-predict`, `--keep-alive` | `12288`, `512`, `5m` | Generation resource controls. |

`raglab-evaluation-gate RUN BASELINE [--tolerance 0.05]` is deliberately separate from scoring:
evaluation preserves evidence, while the gate owns the comparison policy and exits non-zero on any
violation.

`build_report` produces a versioned, JSON-serializable report. `compare_reports` shows previous,
current, and delta values and rejects reports with different case IDs or `top_k` values instead of
comparing unlike experiments.

### `04_rag_evaluation.ipynb`

The notebook has three checkpoints: define the expected truth, measure a controlled retrieval
ranking, then measure generation and make a deliberate regression visible. Run those checkpoint
cells alone for a controlled, service-free lesson. The cell immediately before the live appendix
explicitly assigns `RAGLAB_RUN_EVALUATION_NOTEBOOK=1`; automated notebook tests neutralize that
activation cell in their in-memory copy before execution. To run the same four questions against
an indexed Aster collection through PostgreSQL and Ollama, configure the normal RAGLab environment
and execute the appendix:

```bash
RAGLAB_RUN_EVALUATION_NOTEBOOK=1 \
RAGLAB_EVALUATION_COLLECTION=greenhouse-manuals \
jupyter execute notebooks/04_rag_evaluation.ipynb \
  --output /tmp/raglab-evaluation-live.ipynb
```

### What the evaluation does not prove

String variants make known facts and regressions explainable, but they do not prove full semantic
equivalence, writing quality, completeness outside the ten versioned cases, or safety in an unseen
domain. The Aster scorecard remains a teaching signal; the Raspberry Pi scorecard is a bounded local
preparation gate for this exact demo corpus, not a general quality guarantee. The next useful steps
are broader human-reviewed cases, semantic or model-based judges, and production tracing under
controlled service versions. [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) and
[DeepEval](https://deepeval.com/docs/metrics-introduction) become worth comparing only when those
model-based tradeoffs are intentional; v1 avoids adding either framework prematurely.

## Next step

Use the [operations appendix](appendix.md) for the supervised demo, test environments, hardware
profile, and troubleshooting. Return to the [README](../README.md) for the shortest working path.
