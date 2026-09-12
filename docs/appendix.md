# RAGLab operations appendix

This appendix holds the operational detail intentionally kept out of the README: the full attended
demo procedure, its security boundary, verification environments, resource guidance, and recovery
notes. Start with the [README quick start](../README.md#quick-start) before using these runbooks.

## Live local demo runbook

The supervised demo uses native Ollama for CUDA inference and Docker only for PostgreSQL. Prepared
evidence is reused until its commit, inputs, models, or maximum age no longer match.

### Install the tunnel client once

The launcher accepts only the pinned Linux AMD64 binary and verifies it again before every share
session. Downloading to `/tmp` first prevents an incomplete download from replacing a working
installation:

```bash
curl --fail --location \
  https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64 \
  --output /tmp/cloudflared
printf '%s  %s\n' \
  '53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134' \
  /tmp/cloudflared | sha256sum --check --strict
mkdir -p "$HOME/.local/bin"
install -m 0755 /tmp/cloudflared "$HOME/.local/bin/cloudflared"
"$HOME/.local/bin/cloudflared" --version
```

### Prepare and present

Run every command from the repository root. `prepare` starts only the PostgreSQL Compose service;
Ollama remains a native host service so it can use CUDA directly.

```bash
source .venv/bin/activate
python -m pip install -e . --no-deps

# Optional preflight diagnostics. Both model names and the NVIDIA GPU must appear.
nvidia-smi
ollama list

# Run once initially, and again after changing tracked code, corpus, dataset,
# evaluation baseline, model identity, or demo configuration.
raglab-demo prepare

# Start the UI and tunnel in the foreground.
raglab-demo share
# Open the printed HTTPS URL from another device or network.
# Press Ctrl-C after the demonstration.
```

Expected behavior:

1. `prepare` creates `artifacts/demo/ingestion.json` and `artifacts/demo/evidence.json` only after
   all nine public documents, six evaluation cases, and the local gate succeed.
2. `share` prints one URL shaped like `https://<random>.trycloudflare.com/#token=...`.
3. The UI offers three Raspberry Pi collections, suggested questions, free-form queries, citations,
   timing, ranking stages, model calls, token counts, and the separate evaluated scorecard.
4. `Ctrl-C` stops both Uvicorn and `cloudflared`; the URL and its 256-bit session token are no
   longer usable.

An existing `raglab-paradedb` volume is safe to keep. Ingestion is idempotent when the receipt,
collection configuration, and exact document counts match. If one of the three manifest-owned demo
collections is incompatible, `prepare` replaces only that collection before ingesting it again;
unrelated collections in the same PostgreSQL volume are not deleted.

Both commands reject staged or unstaged changes to tracked files. This guarantees that the evidence
commit identifies the code actually being demonstrated. Commit or restore tracked changes, then run
`prepare` again.

### Security model

#### Temporary sharing

`share` prints `https://<random>.trycloudflare.com/#token=...`. The browser consumes and removes the
fragment, holds the token only in memory, and sends it as a bearer token for `POST /v1/query`.
Uvicorn binds only to `127.0.0.1`; no inbound port is opened. OpenAPI is disabled, security headers
prevent caching and framing, and one active query occupies the GPU while concurrent queries receive
`429`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health/ready` | Verify PostgreSQL and the current tamper-evident preparation manifest. |
| `GET /v1/demo/status` | Collections, resolved models, dataset, provenance, and aggregate scorecard. |
| `GET /v1/demo/cases` | Summaries of the six evaluated cases. |
| `GET /v1/demo/cases/{case_id}` | Preserved answer, citations, ranking, and deterministic validations. |
| `POST /v1/query` | Authenticated live inference against one selected collection. |

The aggregate recall@3, precision@3, MRR@3, fact coverage, grounding, citation precision, and
abstention accuracy belong only to the six-case evaluation. A free-form query reports duration,
tokens, calls, and an explicit `has_ground_truth: false`; it never inherits the evaluation scores.

Quick Tunnels are temporary and have no SLA, do not support SSE, and allow at most 200 concurrent
requests. RAGLab does not stream and admits only one inference at a time. Use them only for
attended demonstrations; see the
[official Cloudflare Quick Tunnels documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

#### Source ingestion

Remote ingestion accepts only HTTP(S) URLs without embedded credentials. DNS must resolve
exclusively to public, non-multicast addresses, and RAGLab repeats that validation before every
redirect. Localhost, private, loopback, link-local, reserved, multicast, mixed public/private, and
unresolvable targets fail with `UnsafeRemoteURLError`. The same policy protects the original URL
passed to the opt-in Jina route; there is no private-network bypass.

### Verify the demo from the terminal

Run the [CI-equivalent checks](#test-strategy-and-suite), then verify the real local stack and
regenerate the six-case evidence:

```bash
docker compose up -d --wait postgres
curl --fail http://127.0.0.1:11434/api/tags >/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
raglab-demo prepare

python - <<'PY'
import json
from pathlib import Path

evidence = json.loads(Path("artifacts/demo/evidence.json").read_text())
print("commit:", evidence["prepared"]["commit"])
print("created_at:", evidence["prepared"]["created_at"])
print("gate_passed:", evidence["gate"]["passed"])
print("cases:", len(evidence["evaluation"]["traces"]))
print("retrieval:", evidence["evaluation"]["report"]["retrieval_summary"])
print("generation:", evidence["evaluation"]["report"]["generation_summary"])
PY
```

The expected minimum result is `gate_passed: True` and `cases: 6`. Finally, run
`raglab-demo share`, open the printed URL from a different network, submit a query, inspect its
citations and telemetry, press `Ctrl-C`, and confirm that the temporary URL no longer responds.

Useful diagnostics:

```bash
docker compose ps
docker compose logs postgres
ollama ps
raglab-demo --help
```

## Validated 8 GB GPU profile

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

## Test strategy and suite

The pyramid keeps algorithmic feedback fast and reserves real converters, databases, models, and
PDFs for explicit boundaries. Environment-gated integration and E2E tests skip when their
services or opt-in variables are absent.

### Test levels

| Level | What it proves | Requirements |
| ----- | -------------- | ------------ |
| Unit | Conversion, parsing, chunking, embeddings, CLIs, ranking, reranking, SQL, orchestration | None |
| Docling integration | Real non-text conversion and provenance | Conversion extras |
| PostgreSQL integration | Migrations, atomic replacement, vectors, filters, BM25, retrieval SQL | Disposable ParadeDB |
| PDF E2E | PDF → Docling → Ollama → PostgreSQL | PDF, Docling, Ollama, PostgreSQL |
| Retrieval E2E | Real ANN + BM25 + BGE | ParadeDB, Ollama, BGE model |
| Notebooks | Teaching paths remain executable and output-free in Git | None by default; services for live cells |

```bash
# CI-equivalent hermetic suite
python -m pip install -e '.[dev,generation,retrieval,tokenizers]'
ruff check src tests scripts
mypy src/raglab
pytest --cov=raglab --cov-fail-under=85 -m "not integration and not e2e"
docker compose -f compose.yaml config --quiet
docker build --tag raglab:test .

# Full collection, including environment-gated skips
pytest
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
jupyter execute notebooks/03_generation.ipynb \
  --output /tmp/raglab-generation.ipynb
jupyter execute notebooks/04_rag_evaluation.ipynb \
  --output /tmp/raglab-evaluation.ipynb
jupyter execute notebooks/benchmark_ingestion_hyperparameters.ipynb \
  --output /tmp/raglab-benchmark.ipynb
```

Use a **disposable** `RAGLAB_TEST_DSN`: tests migrate the schema and create, replace, or delete test
collections. Never aim destructive fixtures at production.

### Subsystem test matrix

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
| Evaluation and local gate | `tests/test_evaluation.py`, `tests/test_evaluation_application.py`, `tests/test_evaluation_gate.py` |
| Corpus and supervised demo | `tests/test_corpus.py`, `tests/test_demo.py`, `tests/test_demo_command.py`, `tests/test_demo_evaluation.py` |
| Output-free executable notebooks | `tests/test_notebooks.py` |

## Troubleshooting and operational notes

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

## BGE reranker OOM incident

### Symptom and root cause

During a retrieval run, the operating system terminated `raglab-retrieve` when memory usage
exceeded the available RAM. The kernel OOM killer, not an ordinary Python exception, ended the
process.

The original reranker processed 32 query/document pairs as one padded tensor. Sequences reached
864 tokens, the float32 BGE model occupied about 2.2 GiB, and autograd remained active. Peak RSS
reached approximately 14.6 GiB.

Calling `.eval()` was insufficient. It changes training-sensitive layers such as dropout, but it
does **not** disable gradient recording or the autograd graph. Inference correctness and inference
memory are separate controls.

### Implemented correction

`BGEReranker` now:

1. loads tokenizer and model lazily and reuses them;
2. wraps scoring in `torch.inference_mode()`;
3. truncates every pair with `max_length=512`;
4. scores internal microbatches of four;
5. appends scores in stable candidate order.

Reranking remains enabled by default. The fix bounds memory at inference instead of hiding the
defect by reducing `candidate_k`.

### Validation

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

### Diagnosis and fallback

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

## Related documentation

Use the [learning guide](guide.md) for the ingestion, retrieval, generation, and evaluation model.
Return to the [README](../README.md) for the minimal project path.
