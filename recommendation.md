# Taking Wiki-RAG to production

Wiki-RAG is deliberately a local, single-user application. This note records what would change for a multi-user service, and which of those changes are worth making only with evidence. The evaluation in `eval/` already showed that some reasonable-sounding ideas add nothing here, so proposals below are tied to what would have to be measured first.

## What stays

- **The pipeline shape:** ingest, chunk, embed, route, retrieve, generate, attribute. Each stage is a separate module and could become a separate service without redesign.
- **One index with metadata filters.** Adding a new kind of page only adds a value to `type`.
- **The refusal contract:** a pinned sentence the system can recognise, so declining is a real state and not a guess from wording.
- **The golden set as a release gate.** No model, prompt or chunking change ships without running it.

## What would be replaced

| Today | At scale | Why |
|---|---|---|
| Ollama on a laptop CPU | vLLM or a managed endpoint on GPUs | Waiting several seconds for the first word is acceptable for one person and unusable for many. |
| Chroma on disk | Qdrant or Weaviate, self-hosted or managed | Replication, backups and tenant isolation. The filters and cosine distance in use carry over directly. |
| BM25 in process memory | The vector database's hybrid search, or OpenSearch | The in-memory index is rebuilt at startup, which stops being cheap well before a million passages. |
| Rule-based routing with an alias table | The same table plus an entity linker | The table grows with every page. The linker would find candidates by similarity to page names. |
| Wikipedia API fetches | Monthly dumps, re-embedding only changed pages | Reproducible, and it respects Wikipedia's API limits. Page revisions are already stored, so diffing is straightforward. |
| Local FastAPI server | The same app behind an API gateway with auth and rate limits | The server currently refuses non-local hosts by design. |

## What to add, and how to decide

**Worth doing first**

- **Faithfulness checking.** The current evaluation checks expected facts with patterns and verifies numbers against sources. A judge model scoring whether each sentence is entailed by its evidence would catch wrong statements that contain no number.
- **Observability.** Per-question route, ranks, timings and citation counts already exist as events. Writing them to a log store with a correlation ID makes failures debuggable.
- **Answer caching**, keyed on the normalised question, model and index version.

**Only with evidence**

- **A cross-encoder reranker.** On the current questions, search already reaches the right pages, so a reranker would mostly add latency. It becomes worth testing when the evaluation set grows harder or the corpus much larger.
- **A larger generation model.** This is likely where the most answer quality remains, since search already finds the right passages and the remaining errors are the small model's: declining too rarely, and weak written Turkish. Measure it by running the golden set with `WIKI_RAG_GEN_MODEL` set to the candidate.
- **A multilingual embedding model.** Translating Turkish questions locally works on the golden set and needs no re-indexing. A multilingual model would be worth comparing if more languages are needed.

## Safety for a public deployment

Retrieved text is untrusted input to the model, and on a corpus people can edit it could contain instructions. Before exposing this publicly: treat passages as data in the prompt, keep the model without any tools, filter personal data from logs, and add per-client rate limits. Page additions would need moderation and quotas, since each one downloads and embeds content.
