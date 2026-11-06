---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: ["web/app.css","web/app.js"]
---

Scope: the Wiki-RAG local app shell: Ask, Library and Measurements views. Mode: Operate.

Audience and job: portfolio reviewers arriving from GitHub with a minute to spare, then the owner asking everyday questions. Task: ask a question, check an answer against its sources, understand how good the system measurably is. Constraints: self-hosted fonts, no CDN, Turkish and English interface text, usable while a CPU model takes 5-9 seconds to start answering. A static replay of recorded real answers serves reviewers without Ollama and is labelled as recorded.

## Direction contract

THESIS: Every citation is a highlighter stroke across the exact source sentence it rests on, so checking an answer is as physical as checking a marked-up reading packet. It refuses the category default of a centered chat column with small source pills.

OWN-WORLD: Photocopy grey ground, toner black rules and a black top bar, fluorescent inks that each belong to one source number (yellow, pink, green, orange, blue) wherever sources are shown. Outside that job, yellow marks the current tab, model activity and, only on screens showing no sources, the row under the pointer; Measurements, where no source numbers appear, draws a gain in yellow and a loss in pink. Setup problems and errors use a separate danger color, and selected text is plain toner. Model-written text is a heavy, friendly sans; source text is a photocopied book serif. Standard web controls with square corners and 2px toner borders. Dark theme is a negative photostat, where the same inks stay fluorescent.

STORY: The reviewer sees a question answered with colored strokes, sees those same colors on the source sentences, understands that the claim is checkable, tries an example including one the system refuses, then opens Measurements to see before-and-after scores.

FIRST VIEWPORT: Black 52px bar with Wiki-RAG, Ask, Library, Measurements, then status, settings, language and theme. Settings carries over the model and passage-count controls of the original interface, and is hidden in the recorded demo, where they do nothing. Two equal panes split by a 2px rule. Left: a large question prompt, example questions for every behaviour including a refusal, and the input pinned at the bottom. Right: the source sheet, showing the packet's contents page (the indexed pages) until a question replaces it with numbered passages. Signature interaction: when an answer finishes, one highlighter sweep runs across the answer sentences and their matching source sentences; hovering or focusing a citation leaves only that pair inked and takes every other stroke away, with no animation because it happens constantly. A refusal leaves the source sheet with no highlights and states it plainly on a solid toner block.

FORM: Reading packet with highlighter, first on the ordered list, chosen by the owner on the decision page. seed:7fef2adce77e266c

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
