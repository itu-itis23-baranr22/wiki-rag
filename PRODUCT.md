# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

FastAPI backend with hand-written HTML, CSS and JavaScript served from the same local process. Chosen by the project owner on 2026-11-06, replacing Streamlit. No CDN or hosted fonts: the app must work offline once models are pulled.

## Users

Primary: people reviewing the owner's portfolio (engineers, hiring managers, instructors). They arrive from GitHub with little time, usually look at screenshots or a recorded demo first, and may try a few example questions. They judge whether the builder understands retrieval and cares about quality.

Secondary: the owner, asking everyday questions in Turkish or English.

## Product Purpose

Wiki-RAG answers questions about a curated set of Wikipedia pages using only models running on the user's own machine. It shows where each claim came from and says it does not know when the pages do not contain the answer. Success means a reviewer understands within a minute what it does, sees that its answers are checkable, and believes the engineering because the numbers are measured and published.

## Positioning

A small, fully local retrieval system whose every stage is visible and measured: which of the indexed passages were searched and used, which passage supports which sentence, and how each change moved the evaluation scores.

## Operating Context

Runs on a laptop at `127.0.0.1`, usually CPU-only. Generation is slow: about 5-9 seconds before the first token and 17-19 tokens per second on the reference machine. The interface must make that wait legible instead of hiding it. Reviewers who do not install anything see a replay of recorded real answers, clearly labelled as recorded.

## Capabilities and Constraints

- Questions in Turkish or English; answers follow the question's language.
- Numbered citations linked to source passages and Wikipedia pages.
- Follow-up questions within a conversation.
- Adding a Wikipedia page from the interface.
- Choosing the language model and the passage count, carried over from the original interface. Not available in the recorded demo.
- Interface text in Turkish and English.
- English Wikipedia is the only corpus. Nothing may be sent to a remote service while answering questions. Fetching a Wikipedia page during ingest is the only network use.

## Brand Commitments

Name: Wiki-RAG. Must not imitate Wikipedia's or the Wikimedia Foundation's logo, wordmark or page design; Wikipedia is credited as the source of the text.

## Evidence on Hand

- Evaluation results produced by `eval/` in this repository. Publish only numbers from real runs.
- No users, testimonials or adoption figures exist; never imply any.

## Product Principles

- Show the work: retrieval is part of the product, not a hidden step.
- Checkable over impressive: a claim without a source is treated as a defect.
- Refusing is a feature, and it should look as deliberate as answering.
- Respect the wait: slow local inference gets honest progress, and can be stopped.
- Measured, not asserted: anything claimed about quality links to a reproducible evaluation.

## Accessibility & Inclusion

Keyboard operable, respects reduced motion, and readable contrast in both themes. Interface and answers work in Turkish, including correct handling of Turkish characters.
