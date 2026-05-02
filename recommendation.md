# Production Deployment Recommendation — Wiki-RAG

This document describes what would change if we wanted to take the local, single-user prototype in this repo and turn it into a multi-tenant, production-grade service. We deliberately built the prototype to be transparent and laptop-friendly; production needs different tradeoffs.

We organize the recommendations as: (1) what stays, (2) what we'd swap out, (3) what we'd add, (4) what we'd measure, and (5) a phased rollout plan.

---

## 1. What stays

The overall pipeline shape — **ingest → chunk → embed → store → route → retrieve → generate** — does not need to change. The component boundaries in `src/` (ingest / chunker / embedder / vector_store / retriever / generator / rag) are also what production code would look like. We'd keep:

- The strict prompt that pins the "I don't know" refusal phrasing. This is the single biggest hallucination defense in the system.
- The metadata schema (`entity`, `type`, `chunk_idx`, `source`). It generalizes cleanly to any closed-domain RAG.
- The "one collection, filter by metadata" architecture (Option B). It scales to many entity types without re-architecting; you just add new values to `type`.

---

## 2. What we'd swap out

### 2.1 Embedding model

| Now | Production |
|-----|------------|
| Ollama `nomic-embed-text` (768-d, CPU) | Same model, but served as a dedicated GPU service (e.g. **Text Embeddings Inference** by HuggingFace, or a managed endpoint), with batched embeddings. |

Why: ingestion is the bottleneck at scale, and `nomic-embed-text` is great quality-per-parameter — we just need batching and a GPU. If we eventually need higher recall, swap to `bge-m3` or `jina-embeddings-v3`.

### 2.2 Vector database

| Now | Production |
|-----|------------|
| ChromaDB on local disk | Managed vector DB: **Qdrant Cloud** or **Weaviate Cloud**, or self-hosted Qdrant on Kubernetes with persistent volumes. |

Why: Chroma is excellent for prototyping but its operational story (HA, replication, backup, multi-tenant isolation) is thinner than Qdrant or Weaviate. Cosine semantics, metadata filtering, and the API surface we use all map 1:1, so the code changes are small.

### 2.3 Generation model

| Now | Production |
|-----|------------|
| Ollama `llama3.2:3b` on CPU | Self-hosted larger model (e.g. **Llama 3.1 70B** or **Qwen 2.5 32B**) on **vLLM** with continuous batching, or — if compliance allows — a managed LLM endpoint (Bedrock, Vertex, AWS Bedrock, AzureOpenAI). |

Why: a 3B model on CPU is fine for a course demo but produces noticeably weaker comparisons and longer-form synthesis than a 30B+ model. vLLM gives ~10× higher throughput than naïve serving via continuous batching. If the project's "no external LLM API" constraint is a hard, on-prem requirement, the path is vLLM on GPU.

### 2.4 Wikipedia ingestion

| Now | Production |
|-----|------------|
| `wikipedia-api` library, ad-hoc cache, no scheduling | Scheduled refresh pipeline that pulls **Wikipedia XML dumps** monthly, diffs against the index, and only re-embeds changed entities. |

Why: live Wikipedia API calls are rate-limited and not reproducible. Dumps are reproducible and fast. Diffing avoids re-embedding pages that didn't change.

### 2.5 Router

The rule-based router is fast and deterministic, but for a much larger entity catalog it would not scale (the alias table grows linearly). At ~10k entities we'd switch to:

- A **classifier model** (a tiny fine-tuned BERT, ~30 MB, GPU-free) that predicts `person | place | both | other` from the query.
- Plus an **embedding-based entity linker** that finds candidate entities by nearest neighbour in a separate "entity name" embedding space.

We'd still keep the alias table as a fast first pass — it's cheap and catches the common case.

---

## 3. What we'd add

### 3.1 Reranker (high-impact, low-effort)

After the bi-encoder retrieval (which is what `nomic-embed-text` gives us), run a small **cross-encoder reranker** (e.g. `bge-reranker-v2-m3` or `ms-marco-MiniLM-L-6-v2`) on the top-20 hits and keep the top-5. Empirically this lifts answer faithfulness substantially on RAG benchmarks. Cost: ~50 ms/query on GPU, ~300 ms on CPU.

### 3.2 Caching

Two layers:
- **Embedding cache**: keyed on `hash(text) + model_version` → vector. Eliminates re-embedding on re-ingest.
- **Answer cache**: keyed on `normalize(question) + model_version + index_version` → answer + sources. Repeat questions ("Who was Einstein?") return instantly. Use Redis with a TTL.

### 3.3 Observability

Production RAG without telemetry is flying blind. We'd emit per-query:
- Retrieval latency, generation latency, tokens/sec.
- Route decision and top-K distances.
- A correlation ID linking the request, the retrieved chunks, and the final answer, written to a structured log store (e.g. ClickHouse, Honeycomb).

### 3.4 Evaluation harness

Build a "golden" question set (~200 questions: factoid, comparative, refusal, ambiguous) with expected answers and expected refusals. Run it on every release. Three metrics:
- **Faithfulness** (LLM-as-judge: is the answer entailed by the retrieved chunks?).
- **Answerable accuracy** (exact-match or LLM-judge for answerable questions).
- **Refusal precision** (does the system correctly refuse on out-of-domain questions?).

This is what makes a model swap or a chunking-strategy tweak provable instead of vibes-based.

### 3.5 Safety and abuse layer

- **PII filter** on user input (block obvious PII before it reaches logs).
- **Prompt-injection defense**: when a retrieved chunk contains text like "ignore previous instructions", strip or quote-fence it. This matters more once the corpus is bigger and includes user-generated wiki content.
- **Rate limiting** per IP / per API key.

### 3.6 Multi-turn handling

Rewrite each user turn into a self-contained query before retrieval ("and his wife?" → "Who was Albert Einstein's wife?") using a small LLM call. Keeps retrieval quality stable across long conversations.

### 3.7 Auth, multi-tenancy, deployment

- Containerize each service (embedder, vector DB, generator, API gateway) with Docker.
- Orchestrate on Kubernetes (or ECS / Cloud Run for simpler ops).
- API gateway in front: auth (JWT or OAuth), rate limits, request/response logging.
- Frontend: replace Streamlit (great for prototyping, awkward for prod) with a small React/Next.js app talking to a FastAPI backend.

---

## 4. What we'd measure (SLOs)

| Metric | Target |
|--------|--------|
| p50 end-to-end latency | < 1.5 s |
| p99 end-to-end latency | < 4.0 s |
| Retrieval-only p50 | < 100 ms |
| Refusal precision on adversarial set | > 95% |
| Faithfulness (LLM judge) on golden set | > 90% |
| Answer cache hit rate | > 30% |
| Index freshness | < 30 days behind upstream Wikipedia |

---

## 5. Phased rollout

**Phase 1 — Hardening (1-2 weeks)**
Wrap the existing prototype in FastAPI, containerize, add a reranker, add basic observability. No infra changes.

**Phase 2 — Scale-up (2-4 weeks)**
Move to Qdrant, move generation to vLLM on GPU, switch ingestion to dump-based pipeline, build the eval harness.

**Phase 3 — Productize (4-8 weeks)**
Multi-tenancy, auth, caching layers, prompt-injection defense, real frontend, on-call rotation, dashboards.

**Phase 4 — Iterate**
A/B test rerankers, embedding models, chunking strategies, and prompts using the eval harness as the gating signal. Keep the rule-based router as long as the entity catalog is small; replace with classifier + entity linker when it isn't.

---

## TL;DR

Keep the architecture, swap the implementations: managed vector DB, vLLM-served larger model, dump-based ingestion, add a reranker and cache, instrument everything, gate releases behind a golden-set evaluation. The prototype already gives you the right component boundaries; production is mostly about durability, scale, and measurement.
