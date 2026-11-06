# Wiki-RAG: product requirements

## 1. Overview

Wiki-RAG is a local question-answering application over a curated set of Wikipedia pages. The embedding model, language model, indexes and interface all run on the user's machine. Its distinguishing feature is that answers are checkable: each sentence links to the source sentence it rests on, and when the pages don't hold the answer, it says so.

It began as a course project (BLG483E, Project 3) and was then rebuilt around measured retrieval quality and a purpose-built interface.

## 2. Goals

- **G1.** Answer free-form questions about the indexed pages, in English or Turkish, using only those pages.
- **G2.** Run fully on one machine. Answering a question must not contact any remote service.
- **G3.** Make each answer checkable down to the sentence.
- **G4.** Keep every stage visible and simple enough to inspect, without a framework such as LangChain.
- **G5.** Decline rather than guess when the pages lack the answer.
- **G6.** Back every claim about quality with a reproducible measurement.

## 3. Non-goals

- A general web search agent. The corpus is limited to chosen pages.
- A multi-user internet service. See `recommendation.md`.
- Multimodal input or output.
- Unbounded conversation memory. Follow-ups use the last few turns.

## 4. Users

- **Portfolio reviewers** arrive from GitHub with little time. They need to understand the product and judge its engineering within a minute, often without installing anything. The recorded demo serves them.
- **The owner** uses it for everyday questions in Turkish or English.

## 5. Functional requirements

### Ingestion
- **F-1.** Index at least 20 people and 20 places from English Wikipedia (`src/config.py`).
- **F-2.** Split each page section into passages of at most 800 characters, on sentence boundaries, with up to 150 characters of whole-sentence overlap.
- **F-3.** Exclude sections that hold no facts about the subject, such as references and external links, when they appear as top-level headings.
- **F-4.** Store each passage with its page, type, section, position, source URL, revision and a content hash.
- **F-5.** Cache downloaded pages. Skip pages whose chunks have not changed. An interrupted run can be restarted safely.
- **F-6.** Refuse to query an index built with different embedding or chunking settings, with a message explaining how to rebuild it.

### Retrieval
- **F-7.** Recognise page names and aliases, ignoring case and accents. Words that are also ordinary vocabulary ("jobs", "swift") count only when capitalised.
- **F-8.** Recognise close misspellings of page names.
- **F-9.** Resolve follow-up questions to the previous turn's pages.
- **F-10.** Search with vector similarity and BM25, merged by reciprocal rank fusion.
- **F-11.** When pages are named, search within them and share the passage budget. Overview questions include the page's opening passage.
- **F-12.** When no page is named, allow at most two passages per page unless too few pages match, and let firm keyword cues restrict the page type.
- **F-13.** Search Turkish questions through a local translation to English.

### Generation
- **F-14.** Generate with the local model, streaming tokens. The model is configurable.
- **F-15.** Instruct the model to use only the passages, cite them by number, answer in the question's language and use a pinned sentence to decline.
- **F-16.** Parse citations, drop invalid ones and find the supporting sentence in each cited passage. Sentences the model left uncited are matched by text comparison and marked as such; sentences with no support are marked as having no source.

### Interface
- **F-17.** Show retrieved passages before the model starts writing, with honest progress during the wait, and allow stopping.
- **F-18.** Highlight each answer sentence and its source sentence in a matching color.
- **F-19.** Show how each passage was found, and how the question was interpreted (translation, spelling correction, follow-up).
- **F-20.** List indexed pages, and allow adding and removing pages from the interface.
- **F-21.** Publish evaluation results in the interface.
- **F-22.** Offer English and Turkish interface text, light and dark themes, keyboard operation and reduced motion.
- **F-23.** Provide a command-line interface with the same core behavior.
- **F-25.** Let the user choose the language model and the passage count, as the original interface did.
- **F-24.** Provide a static replay of recorded real answers, labelled as a recording.

## 6. Design rationale

**One collection with metadata, not two stores.** Questions spanning people and places become one search with optional filters, and there is one index to maintain.

**Hybrid search.** Each method fails differently: vector search can drift to pages that merely sound similar, while keyword search finds exact terms but returns many passages from one page. Merging them and capping passages per page raised search success on the tuning questions from 90% to 96%. The missing embedding prefixes, not vector search itself, turned out to be the main reason the original version missed country questions.

**Rule-based routing.** For a known list of pages, deterministic rules are faster and easier to debug than a model, and every decision can be explained on screen.

**Translating questions instead of a multilingual embedding model.** This needs no re-indexing and no extra download, and it performs well on the Turkish questions. The cost is a short model call per Turkish question.

**No distance threshold for refusals.** Measured distances for unanswerable questions overlap with those of valid ones, and so does the share of question words found in the passages, so either check would reject good questions. The model decides, using the pinned sentence. As a second check, an answer with no sentence traceable to any passage is withheld. Asking for citations made the small model less willing to decline (100% to 83% on the held-out questions), a real cost accepted in exchange for checkable answers.

**A custom interface instead of Streamlit.** Sentence-level highlighting, stopping generation and sources appearing before the answer all need control Streamlit doesn't offer. Removing it also removes its default usage statistics, which conflicted with G2.

## 7. Quality criteria

Measured by `eval/`: 112 questions frozen before the improvements, split evenly into a tuning set and a held-out set. Published on the Measurements page and in the README.

| Measure | Definition |
|---|---|
| Search success | The top 5 passages contain the expected pages, and the answer text where defined, without forbidden pages |
| Answer correct | The answer contains the expected fact, checked by pattern |
| Refusal correct | The system declines questions the pages cannot answer |
| Numbers grounded | Every multi-digit number in the answer appears in the passages |
| Cited answers | The answer links to at least one passage |

## 8. Known limitations

- **Small model.** A 3-billion-parameter model on CPU sometimes writes wrong statements even with the right passages, and its written Turkish often mixes in English words. Citations make errors visible but do not prevent them.
- **Pattern-based scoring** can mark an unusually worded correct answer as wrong, and cannot detect every wrong statement. An AI assistant reading the 50 held-out answers found one wrong fact it accepted; no person has repeated that check.
- **The golden set was written for this project,** not sampled from real use, and is probably easier than real traffic. The held-out half guards against tuning to it, not against that bias.
- **Sentence matching is lexical.** Turkish answers share only names and numbers with English sources, so their highlighted evidence is less precise.
- **English Wikipedia only.**
