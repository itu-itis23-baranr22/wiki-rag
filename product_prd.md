# Product Requirements Document — Wiki-RAG

## 1. Overview

**Wiki-RAG** is a local, retrieval-augmented chat assistant that answers natural-language questions about famous people and famous places using Wikipedia as its only source of knowledge. Everything — the embedding model, the vector database, the language model, and the UI — runs on the user's own laptop. No data leaves the machine.

The system was built for **BLG483E – Project 3** and aims to combine the indexing/retrieval ideas from Project 1 with the AI-driven workflow ideas from Project 2 into one end-to-end application.

## 2. Goals

- G1. Let a user ask a free-form question about any of 40+ canonical entities (20+ famous people, 20+ famous places) and receive a grounded, factual answer.
- G2. Run **fully on localhost** — no external LLM API, no external embedding API.
- G3. Keep the implementation **transparent**: chunking, routing, and orchestration are written from scratch (not delegated to a heavy framework like LangChain) so each component is easy to inspect and tune.
- G4. Provide both a **chat-style UI** (Streamlit) and a **CLI** so the system can be demoed quickly and scripted easily.
- G5. Fail safely: when the indexed data does not contain the answer, the system says **"I don't know based on the available information"** rather than hallucinating.

## 3. Non-goals

- Not a general-purpose web search agent. The knowledge base is intentionally limited to a curated entity list.
- Not a multi-user, internet-deployed service. (See `recommendation.md` for what would change to make it one.)
- Not a multimodal system. Text in, text out.
- Not a multi-turn reasoning system. Each question is answered independently; chat history is shown for the user's benefit but is not fed back into the LLM.

## 4. Users and use cases

The primary users are course evaluators and the developer. Typical use cases:

- "Who was Albert Einstein and what is he known for?" — single-entity factual lookup.
- "Compare Lionel Messi and Cristiano Ronaldo." — multi-entity comparison within one type.
- "Compare the Eiffel Tower and the Statue of Liberty." — multi-entity comparison across the place corpus.
- "Which famous place is located in Turkey?" — semantic search across one type with no named entity in the query.
- "Who is the president of Mars?" — adversarial / out-of-distribution; the system must refuse to fabricate.

## 5. Functional requirements

### 5.1 Ingestion
- **F-1.** The system must ingest at least 20 famous people and 20 famous places from English Wikipedia. Default lists are stored in `src/config.py` and include all entities required by the brief.
- **F-2.** Each Wikipedia page is split into chunks of ~800 characters with ~150 character overlap, paragraph-aware. Sentences and paragraphs are not split unless they exceed `chunk_size` on their own.
- **F-3.** Each chunk is stored with metadata: `entity` (canonical name), `type` (`person` | `place`), `chunk_idx`, `source` (Wikipedia URL).
- **F-4.** Ingestion is idempotent at the entity level: the same entity can be re-ingested with `--reset` to wipe and rebuild.
- **F-5.** Raw Wikipedia text is cached on disk under `data/raw/` so re-runs without `--reset` don't refetch.

### 5.2 Embedding & storage
- **F-6.** Embeddings are produced locally via Ollama's `nomic-embed-text` model (768 dimensions). No external API.
- **F-7.** Embeddings are stored in a single ChromaDB collection persisted at `chroma_db/`. Cosine distance is used.
- **F-8.** No on-disk format other than what Chroma needs; no parallel SQLite store.

### 5.3 Retrieval & routing
- **F-9.** Each user query is routed to one of three targets — `person`, `place`, or `both` — using a deterministic rule-based classifier (entity-name detection with a small alias table, falling back to keyword cues).
- **F-10.** For routed `person` or `place` queries, the system runs a top-K (default K=5) cosine similarity search filtered by `type` metadata.
- **F-11.** For `both` queries, the system runs two parallel top-K searches (one per type) with K=4 each, then merges and re-sorts by distance, yielding up to 8 chunks.

### 5.4 Generation
- **F-12.** Answers are generated locally via Ollama using `llama3.2:3b` by default. The model is configurable via `src/config.py`.
- **F-13.** The prompt includes a fixed system message that instructs the model to (a) use only the supplied context, (b) refuse and reply `"I don't know based on the available information."` if the context is insufficient, and (c) avoid speaking about "the context" by name.
- **F-14.** Tokens are streamed back to the UI for responsiveness.

### 5.5 Interface
- **F-15.** A Streamlit chat UI displays the running conversation, lets the user toggle a "Show source chunks" panel, and exposes a "Clear chat history" button. Each assistant turn shows the route the classifier picked and the retrieved chunks (with Wikipedia links).
- **F-16.** A CLI provides the same core capabilities (ask, show sources, reset, quit) for headless usage.
- **F-17.** Both UIs print the exact retrieval route used for each question, so reviewers can verify the router is doing the right thing.

## 6. Design choices and rationale

### 6.1 Why Option B (one collection + metadata) for the vector store?

The brief offered two options:
- **A.** Two vector stores, one for people, one for places.
- **B.** One vector store with `type` metadata.

We chose **B**. Reasons:
- **Mixed queries are first-class.** Questions like "Compare Einstein and the Eiffel Tower" or "Which person is associated with electricity, and which place is in Turkey?" can be answered with one HNSW index by either dropping the type filter or running two filtered queries. Option A would require manually fanning out and merging.
- **One operational surface.** One persistence directory, one HNSW index to tune, one schema.
- **Filtering is cheap.** Chroma's metadata-filtered search at our scale (~500 chunks total) is effectively free, so we lose nothing by not partitioning physically.
- **The classifier still does the heavy lifting.** When the query is unambiguously about one type, we still pass `where={"type": "person"}` so we don't waste candidate slots on irrelevant chunks. Option A would offer no extra precision here.

### 6.2 Chunking strategy

- **Paragraph-aware**, then **sentence-aware**, then **word-aware** as a last resort. We never split mid-sentence unless the sentence itself is bigger than `CHUNK_SIZE`.
- **800 characters / 150 overlap** by default. Roughly 150–200 tokens, which is in the sweet spot for `nomic-embed-text`. The 150-char overlap recovers facts that straddle a chunk boundary (e.g. "...he was born in 1879. // The young Einstein...").
- The splitter is implemented in `src/chunker.py` from scratch — no LangChain — to honor the project's "language-native" guidance.

### 6.3 Rule-based router

For a closed, finite knowledge base (40 entities), an LLM-based router would be overkill, slow, and harder to debug. Our router:
1. Builds an alias table from `PEOPLE`, `PLACES`, and `ALIASES` once at import time.
2. Looks for the **longest** alias match in the lowered query (so "great wall" beats "wall").
3. Aggregates the matched entities' types: if all matched entities are people → route `person`; all places → `place`; mixed → `both`.
4. Falls back to keyword cues (`who/where/scientist/monument/...`) when no entity name matches; in a tie or total miss, routes to `both`.

This is deterministic, explainable, and adds <1 ms of latency.

### 6.4 Generation prompt

The prompt explicitly forbids fabrication and pins the refusal phrasing. Temperature is set to 0.2 for determinism. We don't pass chat history to the model — each question is answered independently — because (a) the brief doesn't require it, (b) it bounds the prompt size, and (c) it makes the system's behavior much more predictable for evaluation.

### 6.5 Streaming

Both UIs stream tokens. For a 3-billion-parameter model on a CPU laptop, perceived latency is dominated by generation, not retrieval, so streaming is the cheapest UX win available.

## 7. Quality and acceptance criteria

| # | Criterion |
|---|-----------|
| Q-1 | Ingestion covers all 20 required people and 20 required places, plus 10 extras of each. |
| Q-2 | All 14 example questions in the brief produce a sensible answer or a refusal, with no hallucinated facts. |
| Q-3 | Failure-case prompts ("president of Mars", "John Doe") trigger the "I don't know" refusal. |
| Q-4 | The Streamlit UI renders, accepts chat input, streams tokens, and lets the user inspect sources. |
| Q-5 | The system runs end-to-end on a fresh machine by following only the README. |

## 8. Tradeoffs and limitations

- **Closed-domain.** Anything outside the 40-entity corpus produces a refusal. This is by design but limits utility.
- **CPU latency.** A 3B-parameter model on CPU produces ~10–30 tokens/second. Streaming masks but doesn't eliminate this.
- **Naive sentence splitter.** Our regex-based splitter mishandles abbreviations like "Dr." and "U.S.". For our prose-heavy Wikipedia inputs the impact is small.
- **No reranker.** Cosine on `nomic-embed-text` is the only ranking signal; a cross-encoder reranker would improve precision at top-K but would double per-query latency on CPU.
- **No multi-turn reasoning.** Each question is answered independently. Follow-ups like "and his wife?" won't resolve pronouns from the previous turn.

## 9. Future work

See `recommendation.md` for what would change for a production deployment. Lightweight improvements within the local scope:

- Add a tiny reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers) for query-time precision.
- Cache `(question → answer)` and `(question → embedding)` so repeated questions are instant.
- Latency telemetry surfaced in the UI (retrieval time, generation time, tokens/sec).
- Multi-turn rewriting: rewrite each follow-up question into a self-contained one before retrieval.
