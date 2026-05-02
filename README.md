# Local Wikipedia RAG Assistant

A fully local, retrieval-augmented generation (RAG) system that answers questions about famous people and famous places using Wikipedia data, a local embedding model, a local vector store, and a local LLM. **No external APIs, no cloud — everything runs on your machine.**

This was built for **BLG483E – Project 3**.

---

## Features

- Ingests **20 famous people + 20 famous places** from Wikipedia (configurable; you can add more).
- Custom chunking with paragraph-awareness and configurable size/overlap.
- Local embeddings via **Ollama** (`nomic-embed-text`).
- Local vector store via **ChromaDB** with metadata-based filtering (`type = person | place`).
- Rule-based query router that decides whether to search people, places, or both.
- Local generation via **Ollama** (default: `llama3.2:3b`; swap to `phi3` or `mistral` if you prefer).
- Two interfaces: **Streamlit web UI** and **CLI**.
- Returns "I don't know" when the answer isn't in the indexed data.
- Optionally shows the source chunks used for each answer.

---

## Architecture

```
                    ┌─────────────────┐
   User question →  │  Query Router   │  (classifies: person / place / both)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Retriever     │  → Chroma (filter by metadata)
                    └────────┬────────┘
                             │ top-k chunks
                    ┌────────▼────────┐
                    │   Generator     │  → Ollama (llama3.2:3b)
                    └────────┬────────┘
                             │
                          Answer
```

Pipeline stages (each in its own module under `src/`):

| Stage     | Module               | What it does                                          |
|-----------|----------------------|-------------------------------------------------------|
| Ingest    | `src/ingest.py`      | Pulls Wikipedia summaries+content for each entity     |
| Chunk     | `src/chunker.py`     | Splits docs into overlapping, paragraph-aware chunks  |
| Embed     | `src/embedder.py`    | Calls Ollama to embed text locally                    |
| Store     | `src/vector_store.py`| Wraps Chroma; one collection, metadata `type` field   |
| Retrieve  | `src/retriever.py`   | Routes query → person/place/both, top-k semantic search |
| Generate  | `src/generator.py`   | Builds prompt, calls Ollama, enforces grounded answers |
| Orchestrate | `src/rag.py`       | Glues all stages together                             |

---

## Requirements

- **Python 3.10+**
- **Ollama** installed and running locally — https://ollama.com
- ~3 GB disk space (model + embeddings)

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url> wiki-rag
cd wiki-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull the models

Install Ollama from https://ollama.com, then in a terminal:

```bash
# Start the Ollama server (keep this running)
ollama serve

# In another terminal, pull the models we need
ollama pull nomic-embed-text       # for embeddings (~270 MB)
ollama pull llama3.2:3b            # for generation  (~2 GB)
```

Verify it works:

```bash
curl http://localhost:11434/api/tags
```

### 3. Ingest Wikipedia data

This downloads the Wikipedia pages for the 40 default entities, chunks them, embeds them, and writes them to a local Chroma DB at `./chroma_db/`.

```bash
python -m src.ingest
```

You should see progress like:

```
[1/40] person  : Albert Einstein           (12 chunks)
[2/40] person  : Marie Curie               (10 chunks)
...
[40/40] place  : Mount Everest             (14 chunks)
Done. Total chunks indexed: ~480
```

You only need to do this once.

### 4. Run the chat interface

**Option A — Streamlit web UI (recommended for the demo):**

```bash
streamlit run src/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

**Option B — CLI:**

```bash
python cli.py
```

You'll get a prompt like:

```
Wiki-RAG ready. Type your question (or /help, /reset, /quit).
> Who was Albert Einstein and what is he known for?
```

---

## Example Queries

**People:**
- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- What is Frida Kahlo known for?

**Places:**
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What was the Colosseum used for?
- Where is Mount Everest?

**Mixed / comparison:**
- Compare Albert Einstein and Nikola Tesla.
- Compare the Eiffel Tower and the Statue of Liberty.
- Which famous place is located in Turkey?
- Which person is associated with electricity?

**Failure cases (should say "I don't know"):**
- Who is the president of Mars?
- Tell me about a random unknown person John Doe.

---

## Configuration

Edit `src/config.py` to:

- Change the LLM (`llama3.2:3b` → `phi3` / `mistral`).
- Change the embedding model.
- Change chunk size / overlap.
- Add or remove entities from `PEOPLE` / `PLACES`.
- Change `TOP_K` retrieval count.

---

## Repository Layout

```
wiki-rag/
├── README.md              ← you are here
├── product_prd.md         ← Product Requirements Document
├── recommendation.md      ← Notes on taking this to production
├── requirements.txt
├── cli.py                 ← CLI entry point
└── src/
    ├── __init__.py
    ├── config.py          ← all tunable settings + entity lists
    ├── ingest.py          ← Wikipedia fetch + chunk + embed + store
    ├── chunker.py         ← paragraph-aware overlapping chunker
    ├── embedder.py        ← Ollama embeddings client
    ├── vector_store.py    ← Chroma wrapper (Option B: 1 collection + metadata)
    ├── retriever.py       ← query routing + top-k retrieval
    ├── generator.py       ← grounded answer generation
    ├── rag.py             ← end-to-end orchestrator
    └── app.py             ← Streamlit UI
```

---

## Useful Commands

| Command | What it does |
|---------|-------------|
| `python -m src.ingest`   | Fetch + chunk + embed + store all entities |
| `python -m src.ingest --reset` | Wipe the DB and re-ingest from scratch |
| `python cli.py`          | CLI chat |
| `streamlit run src/app.py` | Web UI |

In the CLI, the following slash commands work:

- `/help`   – list commands
- `/sources` – toggle showing source chunks
- `/reset`  – clear chat history (does not delete the index)
- `/quit`   – exit

---

## Demo Video

A 5-minute walkthrough is here: **<add Loom / unlisted YouTube link before submitting>**

---

## License

MIT.
