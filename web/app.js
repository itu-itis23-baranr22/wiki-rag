// The Wiki-RAG interface. Everything model-written or fetched is inserted as text
// nodes, never as HTML.
import * as api from "./api.js";
import {
  EXAMPLES,
  FOLLOWUPS,
  countOf,
  detectLanguage,
  formatDuration,
  formatNumber,
  formatPercent,
  language,
  setLanguage,
  t,
} from "./i18n.js";

const REPO_URL = "https://github.com/itu-itis23-baranr22/wiki-rag";
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (selector, root = document) => root.querySelector(selector);

const state = {
  status: null,
  library: null,
  libraryError: false,
  metrics: undefined,
  metricsSplit: "test",
  turns: [],
  active: null,
  streaming: null,
  view: "ask",
  topK: null,
  model: null,
  librarySort: { key: "title", direction: 1 },
  libraryFilter: "",
  addMessage: null,
  addDraft: { title: "", type: "person" },
  adding: false,
  specimen: null,
};

let turnCounter = 0;

function h(tag, props = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "dataset") Object.assign(element.dataset, value);
    else if (key === "style") Object.entries(value).forEach(([k, v]) => element.style.setProperty(k, v));
    else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else element.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return element;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

const inkClass = (number) => `ink-${((number - 1) % 5) + 1}`;
const pageType = (type) => t(`type_${type}`);

// ------------------------------------------------------------------ static text
function applyStaticText() {
  document.querySelectorAll("[data-t]").forEach((element) => {
    element.textContent = t(element.dataset.t);
  });
  document.querySelectorAll("[data-t-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.tLabel));
  });
  const toggle = $("#lang-toggle");
  toggle.textContent = language() === "tr" ? "EN" : "TR";
  toggle.setAttribute("aria-label", language() === "tr" ? "Switch to English" : "Türkçeye geç");
  toggle.setAttribute("lang", language() === "tr" ? "en" : "tr");
  $("#settings-toggle").setAttribute("title", t("settings"));
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme-icon").setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
  $("#theme-toggle").setAttribute("title", t("toggle_theme"));
  const color = theme === "dark" ? "#E8E9E6" : "#111111";
  document.querySelectorAll('meta[name="theme-color"]').forEach((meta) => meta.setAttribute("content", color));
}

// ---------------------------------------------------------------------- status
function setupProblem(status) {
  if (!status || status.demo) return null;
  if (status.offline || !status.ollama?.reachable) return { text: "notice_ollama", command: "ollama serve" };
  if (status.index?.needs_rebuild) return { text: "notice_rebuild", command: "python -m src.ingest --reset" };
  if (status.missing_models?.length) return { text: "notice_models", command: `ollama pull ${status.missing_models[0]}` };
  if (!status.index?.passages) return { text: "notice_index", command: "python -m src.ingest" };
  return null;
}

function renderStatus() {
  const status = state.status;
  const element = $("#status");
  document.body.classList.toggle("is-demo", Boolean(status?.demo));
  element.classList.remove("is-problem");
  if (!status) {
    element.textContent = t("status_checking");
  } else if (status.demo) {
    element.textContent = t("status_demo");
  } else if (!status.ollama?.reachable) {
    element.textContent = t("status_ollama_down");
    element.classList.add("is-problem");
  } else if (!status.ready) {
    element.textContent = t("status_not_ready");
    element.classList.add("is-problem");
  } else {
    element.textContent = t("status_local", { model: state.model || status.defaults?.model });
  }
  renderNotice();
  syncComposer();
  syncSettings();
}

function renderNotice() {
  const notice = $("#notice");
  const status = state.status;
  notice.replaceChildren();
  notice.classList.remove("is-problem");
  if (!status) {
    notice.hidden = true;
    return;
  }
  if (status.demo) {
    notice.hidden = false;
    notice.append(
      h("span", {}, t("notice_demo")),
      h("a", { href: `${REPO_URL}#quick-start` }, t("notice_demo_link"))
    );
    return;
  }
  const problem = setupProblem(status);
  if (!problem) {
    notice.hidden = true;
    return;
  }
  notice.hidden = false;
  notice.classList.add("is-problem");
  const copy = h("button", { class: "copy", type: "button" }, icon("copy"), h("span", {}, t("copy")));
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(problem.command);
      $("span", copy).textContent = t("copied");
    } catch {
      /* clipboard can be unavailable; the command stays visible to copy by hand */
    }
  });
  notice.append(h("span", {}, t(problem.text)), h("code", {}, problem.command), copy);
}

async function refreshStatus() {
  const wasReady = state.status?.ready;
  state.status = await api.getStatus();
  if (!state.status.ready && !state.status.demo) setTimeout(refreshStatus, 5000);
  if (!state.topK) state.topK = state.status.defaults?.top_k || 5;
  if (!state.model) state.model = state.status.defaults?.model || null;
  renderStatus();
  if (!state.library?.total_passages) {
    await loadLibrary();
  } else if (wasReady !== state.status.ready && !state.streaming) {
    // Ollama came back or went away, so example and contents rows change their availability.
    renderThread();
    renderSheet();
  }
}

async function loadLibrary() {
  try {
    state.library = await api.getLibrary();
    state.libraryError = false;
  } catch {
    state.libraryError = true;
  }
  renderThread();
  renderSheet();
  if (state.view === "library") renderLibrary();
}

// -------------------------------------------------------------------- composer
function canAsk() {
  const status = state.status;
  return Boolean(status && status.ready && !status.demo);
}

function syncComposer() {
  const input = $("#question");
  const send = $("#send");
  const hint = $("#composer-hint");
  const streaming = Boolean(state.streaming);
  const status = state.status;

  input.disabled = !canAsk();
  input.placeholder = status?.demo
    ? t("disabled_demo")
    : status && !status.ready
      ? t("disabled_setup")
      : state.turns.length
        ? t("placeholder_followup")
        : t("placeholder");

  send.classList.toggle("is-stop", streaming);
  send.type = streaming ? "button" : "submit";
  send.disabled = streaming ? false : !canAsk();
  send.replaceChildren(icon(streaming ? "stop" : "send"), h("span", {}, t(streaming ? "stop" : "send")));

  hint.textContent = streaming ? t("hint_stop") : canAsk() ? t("hint_keys") : "";
  $("#new-chat").hidden = !state.turns.length;
}

function syncSettings() {
  const status = state.status;
  const select = $("#model-select");
  const models = (status?.ollama?.models || []).filter((name) => !name.includes("embed"));
  const current = state.model || status?.defaults?.model;
  const options = models.includes(current) || !current ? models : [current, ...models];
  select.replaceChildren(...options.map((name) => h("option", { value: name, selected: name === current }, name)));
  select.disabled = !canAsk();
  const range = $("#topk-range");
  range.max = status?.defaults?.max_top_k || 10;
  range.value = state.topK || 5;
  range.disabled = !canAsk();
  $("#topk-value").textContent = range.value;
}

function autosize() {
  const input = $("#question");
  if (CSS.supports("field-sizing", "content")) return;
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

// ---------------------------------------------------------------------- asking
function historyFor() {
  return state.turns
    .filter((turn) => turn.phase === "done")
    .slice(-3)
    .map((turn) => ({
      question: turn.question,
      answer: turn.done?.answer || "",
      entities: turn.done?.entities || [],
    }));
}

async function ask(question) {
  question = question.trim();
  if (!question || state.streaming) return;
  if (state.view !== "ask") location.hash = "#ask";

  const history = historyFor();
  const turn = {
    id: ++turnCounter,
    question,
    phase: "searching",
    route: null,
    sources: [],
    text: "",
    done: null,
    error: null,
    startedAt: performance.now(),
    fresh: true,
  };
  state.turns.push(turn);
  state.active = turn;
  state.streaming = turn;
  const controller = new AbortController();
  turn.controller = controller;
  $("#question").value = "";
  autosize();
  renderThread();
  renderSheet();
  syncComposer();
  scrollToTurn(turn);
  const ticker = setInterval(() => updatePhase(turn), 250);

  try {
    const body = { question, history };
    if (state.topK) body.top_k = Number(state.topK);
    if (state.model) body.model = state.model;
    for await (const event of api.chat(body, controller.signal)) handleEvent(turn, event);
    if (!["done", "error"].includes(turn.phase)) {
      turn.phase = "error";
      turn.error = { code: "network" };
    }
  } catch (error) {
    if (error.name === "AbortError") {
      turn.phase = "stopped";
    } else {
      turn.phase = "error";
      turn.error = { code: "network" };
    }
  } finally {
    clearInterval(ticker);
    state.streaming = null;
    renderThread();
    renderSheet();
    syncComposer();
    if (turn.phase === "done") runSweep(turn);
    if (matchMedia("(pointer: fine)").matches && canAsk()) $("#question").focus();
  }
}

function handleEvent(turn, event) {
  switch (event.event) {
    case "route":
      turn.route = event;
      renderTurnInPlace(turn);
      break;
    case "sources":
      turn.sources = event.sources;
      turn.phase = "reading";
      renderTurnInPlace(turn);
      renderSheet();
      break;
    case "token":
      if (turn.phase !== "writing") {
        turn.phase = "writing";
        renderTurnInPlace(turn);
      }
      turn.text += event.token;
      scheduleStreamPaint(turn);
      break;
    case "done":
      turn.done = event;
      turn.phase = "done";
      break;
    case "error":
      turn.phase = "error";
      turn.error = event;
      break;
  }
}

let paintQueued = false;
function scheduleStreamPaint(turn) {
  if (paintQueued) return;
  paintQueued = true;
  requestAnimationFrame(() => {
    paintQueued = false;
    const answer = $(`#turn-${turn.id} .answer`);
    if (!answer || turn.phase !== "writing") return;
    const thread = $("#thread");
    const pinned = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
    const caret = h("span", { class: "caret", "aria-hidden": "true" });
    answer.replaceChildren(...streamingNodes(turn.text), caret);
    if (matchMedia("(max-width: 860px)").matches) {
      // On a phone the whole page scrolls. Keep following until the reader scrolls away.
      if (!turn.readerScrolled) caret.scrollIntoView({ block: "nearest" });
    } else if (pinned) {
      thread.scrollTop = thread.scrollHeight;
    }
  });
}

function streamingNodes(text) {
  // A citation still being typed ("[1") stays hidden until it is complete.
  const safe = text.replace(/\[[^\]]*$/, "");
  return safe.split(/(\[(?:passage |source )?\d+(?:\s*,\s*\d+)*\])/i).map((part, index) => {
    if (index % 2 === 0) return document.createTextNode(part);
    const numbers = part.match(/\d+/g).map(Number);
    return h(
      "span",
      {},
      numbers.map((number) => h("span", { class: `cite ${inkClass(number)}`, "aria-hidden": "true" }, number))
    );
  });
}

function scrollToTurn(turn) {
  const element = $(`#turn-${turn.id}`);
  if (element) element.scrollIntoView({ block: "start", behavior: "auto" });
}

// ----------------------------------------------------------------- thread view
function renderThread() {
  const thread = $("#thread");
  if (!state.turns.length) {
    thread.replaceChildren(renderEmpty());
    return;
  }
  thread.replaceChildren(...state.turns.map(renderTurn));
}

function exampleList(examples) {
  return h(
    "ul",
    { class: "examples" },
    examples.map((example) =>
      h(
        "li",
        {},
        h(
          "button",
          {
            class: "example",
            type: "button",
            disabled: !(canAsk() || state.status?.demo),
            onclick: () => ask(example.question),
          },
          h("span", { class: "example-question", lang: /nerede/.test(example.question) ? "tr" : "en" }, example.question),
          h("span", { class: "example-kind" }, t(example.kind))
        )
      )
    )
  );
}

function renderSpecimen() {
  const specimen = state.specimen;
  if (!specimen) return null;
  const ink = inkClass(specimen.number);
  return h(
    "figure",
    { class: "specimen" },
    h(
      "p",
      { class: "specimen-answer", lang: "en" },
      h("mark", { class: ink }, specimen.answer),
      h("span", { class: `cite ${ink}`, "aria-hidden": "true" }, specimen.number)
    ),
    h("blockquote", { class: "specimen-source", lang: "en" }, h("mark", { class: ink }, specimen.source)),
    h("figcaption", { class: "specimen-credit" }, t("specimen_credit", { entity: specimen.entity }))
  );
}

function emptyBody() {
  const pages = state.library ? state.library.pages.filter((page) => page.passages).length : state.status?.index?.pages;
  if (pages === undefined) return null;
  if (!pages) return t("empty_body_unbuilt");
  return t(state.status?.demo ? "empty_body_demo" : "empty_body", { pages: formatNumber(pages) });
}

function renderEmpty() {
  const body = emptyBody();
  return h(
    "div",
    { class: "empty" },
    h("h1", {}, t("empty_title")),
    body && h("p", {}, body),
    renderSpecimen(),
    h("h2", { class: "examples-title" }, t("examples")),
    exampleList(EXAMPLES)
  );
}

function moreExamples() {
  // In the recording nothing can be typed, so the next question has to be on screen.
  const asked = new Set(state.turns.map((turn) => turn.question));
  const remaining = EXAMPLES.filter((example) => !asked.has(example.question));
  if (!state.status?.demo || !remaining.length) return null;
  return h("div", { class: "more" }, h("h2", { class: "examples-title" }, t("try_another")), exampleList(remaining));
}

function routeNote(turn) {
  const route = turn.route;
  if (!route) return null;
  const parts = [];
  if (route.reason === "fuzzy") parts.push(t("note_fuzzy", { names: route.entities.join(", ") }));
  if (route.reason === "followup") parts.push(t("note_followup", { names: route.entities.join(", ") }));
  if (route.translated) parts.push(t("note_translated", { q: route.translated }));
  return parts.length ? h("p", { class: "turn-note" }, parts.join(" · ")) : null;
}

function renderTurn(turn) {
  const isPast = turn !== state.turns[state.turns.length - 1];
  return h(
    "article",
    { class: `turn${isPast ? " is-past" : ""}`, id: `turn-${turn.id}`, dataset: { turn: turn.id } },
    h("h1", { class: "turn-question", lang: turn.route?.language || null }, turn.question),
    h("div", { class: "turn-body" }, turnBody(turn, isPast))
  );
}

function renderTurnInPlace(turn) {
  const element = $(`#turn-${turn.id} .turn-body`);
  if (!element) return;
  element.replaceChildren(...turnBody(turn, false));
}

function phaseLine(turn) {
  const elapsed = formatDuration(performance.now() - turn.startedAt);
  if (turn.phase === "searching") return [t("phase_searching"), null];
  if (turn.phase === "reading") {
    return [t("phase_reading", { n: formatNumber(turn.sources.length), s: elapsed }), t("phase_reading_why")];
  }
  return [t("phase_writing", { s: elapsed }), null];
}

function updatePhase(turn) {
  const element = $(`#turn-${turn.id} .phase-text`);
  if (element) element.textContent = phaseLine(turn)[0];
}

function turnBody(turn, isPast) {
  const nodes = [routeNote(turn)];
  const answer = (extra = "") => h("div", { class: `answer ${extra}`, lang: turn.route?.language || null });

  if (["searching", "reading"].includes(turn.phase)) {
    const [text, why] = phaseLine(turn);
    nodes.push(
      h(
        "p",
        { class: "phase", role: "status" },
        h("span", { class: "phase-mark", "aria-hidden": "true" }),
        h("span", { class: "phase-text" }, text),
        why && h("small", {}, why)
      )
    );
  } else if (turn.phase === "writing") {
    const element = answer("is-streaming");
    element.append(...streamingNodes(turn.text), h("span", { class: "caret", "aria-hidden": "true" }));
    nodes.push(element, h("p", { class: "turn-meta phase-text" }, phaseLine(turn)[0]));
  } else if (turn.phase === "stopped") {
    if (turn.text) {
      const element = answer("is-streaming");
      element.append(...streamingNodes(turn.text));
      nodes.push(element);
    }
    nodes.push(
      h(
        "div",
        { class: "turn-actions" },
        h("span", {}, t("phase_stopped")),
        h("button", { class: "link", type: "button", onclick: () => ask(turn.question) }, t("retry"))
      )
    );
  } else if (turn.phase === "error") {
    nodes.push(renderError(turn));
  } else if (turn.done?.refused) {
    nodes.push(
      h(
        "div",
        { class: "refusal", lang: turn.done.language },
        h("strong", {}, turn.done.answer),
        h("span", { lang: language() }, t(turn.done.withheld ? "refusal_withheld" : "refusal_detail", { n: formatNumber(turn.sources.length) }))
      ),
      metaLine(turn)
    );
  } else if (turn.done) {
    nodes.push(renderAnswer(turn), metaLine(turn));
    const followup = !isPast && FOLLOWUPS[turn.question];
    if (followup && (canAsk() || state.status?.demo)) {
      nodes.push(
        h(
          "p",
          { class: "suggest" },
          h("span", {}, t("try_followup")),
          h("button", { class: "link", type: "button", lang: "en", onclick: () => ask(followup) }, followup)
        )
      );
    }
  }

  if (isPast && turn.sources.length) {
    nodes.push(
      h(
        "div",
        { class: "turn-actions" },
        state.active === turn
          ? h("span", { class: "current-sources" }, t("showing_sources"))
          : h("button", { class: "link", type: "button", onclick: () => activate(turn) }, t("show_sources"))
      )
    );
  }
  if (!isPast && turn.phase === "done") nodes.push(moreExamples());
  return nodes.filter(Boolean);
}

function renderError(turn) {
  const error = turn.error || {};
  const key = { ollama: "error_ollama", network: "error_network", demo: "error_demo" }[error.code] || "error_internal";
  return h(
    "div",
    { class: "turn-error", role: "alert" },
    h("strong", {}, t("error_title")),
    h("p", {}, t(key, { message: error.message || "" })),
    error.code !== "demo" && h("button", { class: "link", type: "button", onclick: () => ask(turn.question) }, t("retry"))
  );
}

function doneMeta(turn) {
  const timings = turn.done.timings || {};
  const indexed = formatNumber(turn.route?.indexed_passages || 0);
  const retrieve = formatDuration(timings.retrieve_ms || 0);
  if (turn.done.refused) {
    return t("done_meta_refused", { indexed, retrieve, total: formatDuration(timings.total_ms || 0) });
  }
  const used = turn.done.attribution.used_sources;
  const pages = new Set(turn.sources.filter((s) => used.includes(s.number)).map((s) => s.entity)).size;
  return t("done_meta", {
    passages: countOf(used.length, "passage", "passages", "pasaj"),
    pages: countOf(pages, "page", "pages", "sayfa"),
    indexed,
    retrieve,
    first: formatDuration(timings.first_token_ms || 0),
    speed: t("speed", { n: formatNumber(timings.tokens_per_second || 0) }),
  });
}

function metaLine(turn) {
  // Each fact is its own unit, so a wrapped line never starts or ends with a separator.
  return h("p", { class: "turn-meta" }, doneMeta(turn).split(" · ").map((part) => h("span", {}, part)));
}

function renderAnswer(turn) {
  const element = h("div", { class: "answer", lang: turn.done.language });
  let sentenceIndex = 0;
  for (const paragraph of turn.done.attribution.paragraphs) {
    const p = h("p");
    paragraph.forEach((sentence, index) => {
      const id = sentenceIndex++;
      sentence.id = id;
      if (index) p.append(" ");
      if (sentence.sources.length) {
        p.append(
          h(
            "mark",
            {
              class: inkClass(sentence.sources[0]),
              style: { "--order": id },
              dataset: { sentence: id },
            },
            sentence.text
          )
        );
        for (const number of sentence.sources) {
          const source = turn.sources.find((s) => s.number === number);
          p.append(
            h(
              "button",
              {
                class: `cite ${inkClass(number)}${sentence.inferred ? " is-inferred" : ""}`,
                type: "button",
                title: sentence.inferred ? t("cite_inferred") : null,
                "aria-label": t("cite_label", { n: number, entity: source?.entity || "" }),
                dataset: { source: number, sentence: id, turn: turn.id },
              },
              number
            )
          );
        }
      } else {
        p.append(sentence.text, h("span", { class: "no-source" }, t("no_source")));
      }
    });
    element.append(p);
  }
  element.classList.toggle("will-sweep", Boolean(turn.fresh) && !turn.swept);
  return element;
}

// --------------------------------------------------------------- the highlight
function runSweep(turn) {
  if (!turn.fresh || turn.swept) return;
  turn.swept = true;
  const targets = [$(`#turn-${turn.id} .answer`), $("#sheet")].filter(Boolean);
  // Two frames, so the browser paints the empty state before the stroke starts.
  requestAnimationFrame(() =>
    requestAnimationFrame(() =>
      targets.forEach((element) => {
        element.classList.add("sweep");
        element.classList.remove("will-sweep");
      })
    )
  );
  // Once the strokes have landed the transition must go, or its delays would replay on
  // every hover, leaving dark text on the dark ground for a moment.
  const sentences = turn.done.attribution.paragraphs.flat().length;
  setTimeout(() => targets.forEach((element) => element.classList.remove("sweep")), sentences * 70 + 500);
}

function clearLit() {
  document.querySelectorAll(".is-lit").forEach((element) => element.classList.remove("is-lit"));
  $("#view-ask").classList.remove("is-dimming");
}

function lightSentence(turnId, sentenceId, sourceNumber = null) {
  clearLit();
  if (sentenceId === undefined) return;
  const turn = state.turns.find((item) => item.id === Number(turnId));
  if (!turn || turn !== state.active) return;
  $("#view-ask").classList.add("is-dimming");
  $(`#turn-${turn.id} mark[data-sentence="${sentenceId}"]`)?.classList.add("is-lit");
  document.querySelectorAll("#sheet mark[data-sentences]").forEach((mark) => {
    const matches = mark.dataset.sentences.split(" ").includes(String(sentenceId));
    const rightSource = sourceNumber === null || mark.dataset.source === String(sourceNumber);
    if (matches && rightSource) {
      mark.classList.add("is-lit");
      mark.closest(".passage")?.classList.add("is-lit");
    }
  });
  const sentence = turn.done?.attribution.paragraphs.flat().find((item) => item.id === Number(sentenceId));
  for (const number of sentence?.sources || []) {
    if (sourceNumber === null || number === Number(sourceNumber)) {
      $(`#sheet .passage[data-source="${number}"]`)?.classList.add("is-lit");
    }
  }
}

function revealSource(number) {
  const passage = $(`#sheet .passage[data-source="${number}"]`);
  if (!passage) return;
  const details = passage.closest("details");
  if (details) details.open = true;
  passage.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
}

function activate(turn) {
  state.active = turn;
  clearLit();
  renderThread();
  renderSheet();
}

function wireHighlights() {
  const view = $("#view-ask");
  view.addEventListener("click", (event) => {
    const cite = event.target.closest("button.cite");
    if (cite) {
      const turn = state.turns.find((item) => item.id === Number(cite.dataset.turn));
      if (turn && turn !== state.active) activate(turn);
      lightSentence(cite.dataset.turn, cite.dataset.sentence, cite.dataset.source);
      revealSource(Number(cite.dataset.source));
      return;
    }
    if (event.target.closest(".back-to-answer")) {
      clearLit();
      $(`#turn-${state.active?.id}`)?.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
    }
  });

  const focusOn = (event) => {
    const target = event.target.closest("button.cite, .answer mark");
    if (!target) return false;
    const turnElement = target.closest(".turn");
    lightSentence(turnElement?.dataset.turn, target.dataset.sentence, target.dataset.source ?? null);
    if (matchMedia("(min-width: 861px)").matches) {
      const number = target.dataset.source ?? state.active?.done?.attribution.paragraphs.flat().find((s) => s.id === Number(target.dataset.sentence))?.sources[0];
      if (number) revealSource(Number(number));
    }
    return true;
  };

  if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
    view.addEventListener("pointerover", (event) => {
      if (event.pointerType === "mouse") focusOn(event);
    });
    view.addEventListener("pointerout", (event) => {
      if (event.target.closest("button.cite, .answer mark") && !event.relatedTarget?.closest?.("button.cite, .answer mark")) clearLit();
    });
  }
  view.addEventListener("focusin", (event) => {
    if (event.target.matches("button.cite") && event.target.matches(":focus-visible")) focusOn(event);
  });
  view.addEventListener("focusout", (event) => {
    if (event.target.matches("button.cite")) clearLit();
  });
}

// ---------------------------------------------------------------- source sheet
function renderSheet() {
  const sheet = $("#sheet");
  const turn = state.active;
  sheet.classList.remove("sweep", "will-sweep");
  if (!turn || !turn.sources.length) {
    if (turn && ["searching"].includes(turn.phase)) {
      sheet.replaceChildren(h("h2", { class: "sheet-title" }, t("sheet_title_reading")), ...skeletons(9));
    } else {
      sheet.replaceChildren(renderContents());
    }
    return;
  }

  if (turn.phase !== "done") {
    const title = turn.phase === "stopped" ? "sheet_title_stopped" : "sheet_title_reading";
    sheet.replaceChildren(
      h("h2", { class: "sheet-title" }, t(title)),
      ...turn.sources.map((source) => renderPassage(turn, source, "is-pending"))
    );
    return;
  }

  if (turn.done.refused) {
    sheet.replaceChildren(
      h("h2", { class: "sheet-title" }, t("sheet_title_refused")),
      ...turn.sources.map((source) => renderPassage(turn, source, "is-unused"))
    );
    return;
  }

  const used = turn.done.attribution.used_sources;
  const usedSources = turn.sources.filter((s) => used.includes(s.number));
  const unused = turn.sources.filter((s) => !used.includes(s.number));
  const children = [
    h("h2", { class: "sheet-title" }, t("sheet_title_used")),
    ...usedSources.map((source) => renderPassage(turn, source, "")),
  ];
  if (unused.length) {
    children.push(
      h(
        "details",
        { class: "unused" },
        h("summary", {}, icon("chevron"), t("sheet_unused", { n: formatNumber(unused.length) })),
        unused.map((source) => renderPassage(turn, source, "is-unused"))
      )
    );
  }
  sheet.replaceChildren(...children);
  if (turn.fresh && !turn.swept) sheet.classList.add("will-sweep");
}

function skeletons(count) {
  return Array.from({ length: count }, (_, index) =>
    h("div", { class: "skeleton", style: { width: `${70 + ((index * 37) % 30)}%` } })
  );
}

function renderContents() {
  if (state.libraryError) return h("p", {}, t("contents_error"));
  const library = state.library;
  if (!library) return h("div", {}, ...skeletons(14));
  const indexed = library.pages.filter((page) => page.passages);
  const groups = [
    ["contents_people", "person"],
    ["contents_places", "place"],
    ["contents_other", "other"],
  ];
  const askAbout = (page) => {
    const template = page.type === "person" ? "ask_person" : "ask_place";
    ask(t(template, { title: page.title.replace(/\s*\(.*\)$/, "") }));
  };
  const interactive = canAsk();
  const head = h(
    "div",
    { class: "contents-head" },
    h("h2", {}, t("contents")),
    h(
      "span",
      {},
      t("contents_summary", {
        pages: formatNumber(indexed.length),
        passages: formatNumber(library.total_passages),
      })
    )
  );
  if (!indexed.length) return h("div", {}, head, h("p", { class: "contents-empty" }, t("contents_empty")));
  return h(
    "div",
    {},
    head,
    h(
      "div",
      { class: "contents-groups" },
      groups.map(([label, type]) => {
        const pages = indexed.filter((page) => page.type === type);
        if (!pages.length) return null;
        return h(
          "section",
          { class: "contents-group" },
          h("h3", {}, t(label)),
          pages.map((page) =>
            h(
              interactive ? "button" : "div",
              {
                class: "contents-row",
                type: interactive ? "button" : null,
                lang: "en",
                onclick: interactive ? () => askAbout(page) : null,
              },
              h("span", { class: "contents-title" }, page.title),
              h("span", { class: "contents-dots", "aria-hidden": "true" }),
              h("span", { class: "contents-count" }, formatNumber(page.passages))
            )
          )
        );
      })
    )
  );
}

function evidenceFor(turn, number) {
  // Merge evidence from every answer sentence citing this passage, in text order.
  const spans = new Map();
  for (const sentence of turn.done?.attribution.paragraphs.flat() || []) {
    for (const evidence of sentence.evidence) {
      if (evidence.source !== number) continue;
      const key = `${evidence.start}:${evidence.end}`;
      if (!spans.has(key)) spans.set(key, { start: evidence.start, end: evidence.end, sentences: [], order: sentence.id });
      spans.get(key).sentences.push(sentence.id);
    }
  }
  return [...spans.values()].sort((a, b) => a.start - b.start);
}

function passageText(turn, source) {
  const nodes = [];
  let cursor = 0;
  for (const span of evidenceFor(turn, source.number)) {
    if (span.start < cursor) continue;
    if (span.start > cursor) nodes.push(source.text.slice(cursor, span.start));
    nodes.push(
      h(
        "mark",
        {
          class: inkClass(source.number),
          style: { "--order": span.order },
          dataset: { sentences: span.sentences.join(" "), source: source.number },
        },
        source.text.slice(span.start, span.end)
      )
    );
    cursor = span.end;
  }
  if (cursor < source.text.length) nodes.push(source.text.slice(cursor));
  return nodes;
}

function renderPassage(turn, source, stateClass) {
  const position =
    source.chunk_index !== null && source.chunk_index !== undefined && source.page_passages
      ? t("passage_position", { i: formatNumber(source.chunk_index + 1), n: formatNumber(source.page_passages) })
      : null;
  const ranks = [
    source.vector_rank && t("rank_vector", { n: source.vector_rank }),
    source.lexical_rank && t("rank_lexical", { n: source.lexical_rank }),
  ].filter(Boolean);
  const section = source.section ? source.section.split(" > ").pop() : t("lead_section");
  return h(
    "article",
    {
      class: `passage ${stateClass} ${stateClass ? "" : inkClass(source.number)}`,
      dataset: { source: source.number },
    },
    h(
      "header",
      { class: "passage-head" },
      h("span", { class: "passage-num" }, source.number),
      h(
        "h3",
        { class: "passage-title" },
        h("span", { lang: "en" }, source.entity),
        h("small", { lang: source.section ? "en" : null }, section)
      ),
      source.url &&
        h(
          "a",
          { class: "passage-link", href: source.url, target: "_blank", rel: "noopener noreferrer" },
          t("open_wikipedia"),
          icon("out")
        )
    ),
    h("p", { class: "passage-body", lang: "en" }, passageText(turn, source)),
    h("div", { class: "passage-foot" }, position && h("span", {}, position), ranks.length ? h("span", {}, ranks.join(" · ")) : null),
    !stateClass && h("button", { class: "link back-to-answer", type: "button" }, t("back_to_answer"))
  );
}

// --------------------------------------------------------------------- library
function sortedPages() {
  const { key, direction } = state.librarySort;
  const query = state.libraryFilter.trim().toLocaleLowerCase(language());
  const pages = state.library.pages.filter((page) => !query || page.title.toLocaleLowerCase(language()).includes(query));
  return [...pages].sort((a, b) => {
    const first = a[key];
    const second = b[key];
    const order = typeof first === "number" ? first - second : String(first).localeCompare(String(second), language());
    return order * direction;
  });
}

function renderLibrary() {
  const root = $("#view-library");
  if (state.libraryError) {
    root.replaceChildren(h("div", { class: "page-inner" }, h("h1", {}, t("library_title")), h("p", { class: "page-intro" }, t("library_error"))));
    return;
  }
  if (!state.library) {
    root.replaceChildren(h("div", { class: "page-inner" }, h("h1", {}, t("library_title")), ...skeletons(10)));
    return;
  }
  const library = state.library;
  const indexed = library.pages.filter((page) => page.passages).length;
  const demo = state.status?.demo;
  root.replaceChildren(
    h(
      "div",
      { class: "page-inner" },
      h("h1", {}, t("library_title")),
      h(
        "p",
        { class: "page-intro" },
        t("library_summary", { pages: formatNumber(indexed), passages: formatNumber(library.total_passages) })
      ),
      renderAddForm(demo),
      h(
        "section",
        { class: "section" },
        h(
          "label",
          { class: "field table-tools" },
          h("span", {}, t("filter_label")),
          h("input", {
            type: "text",
            placeholder: t("filter_placeholder"),
            value: state.libraryFilter,
            oninput: (event) => {
              state.libraryFilter = event.target.value;
              renderLibraryTable();
            },
          })
        ),
        h("div", { class: "table-wrap", id: "library-table" })
      )
    )
  );
  renderLibraryTable();
}

function renderAddForm(demo) {
  const message = state.addMessage;
  return h(
    "section",
    { class: "section" },
    h("h2", {}, t("add_title")),
    demo && h("p", { class: "section-help" }, t("add_demo")),
    h(
      "form",
      {
        class: "add-form",
        onsubmit: (event) => {
          event.preventDefault();
          const data = new FormData(event.target);
          state.addDraft = { title: String(data.get("title")).trim(), type: String(data.get("type")) };
          addPage(state.addDraft.title, state.addDraft.type);
        },
      },
      h(
        "label",
        { class: "field" },
        h("span", {}, t("add_field")),
        h("input", {
          type: "text",
          name: "title",
          required: true,
          maxlength: 256,
          placeholder: t("add_placeholder"),
          value: state.addDraft.title,
          disabled: demo || state.adding,
        })
      ),
      h(
        "label",
        { class: "field" },
        h("span", {}, t("add_type")),
        h(
          "select",
          { name: "type", disabled: demo || state.adding },
          ["person", "place", "other"].map((type) =>
            h("option", { value: type, selected: state.addDraft.type === type }, pageType(type))
          )
        )
      ),
      h("button", { class: "button", type: "submit", disabled: demo || state.adding }, t("add_button"))
    ),
    h("p", { class: `form-message${message?.error ? " is-error" : ""}`, role: "status" }, message?.text || "")
  );
}

function sortHeader(key, label, numeric = false) {
  const { key: currentKey, direction } = state.librarySort;
  const sorted = currentKey === key;
  return h(
    "th",
    { class: numeric ? "num" : null, "aria-sort": sorted ? (direction > 0 ? "ascending" : "descending") : null, scope: "col" },
    h(
      "button",
      {
        type: "button",
        onclick: () => {
          state.librarySort = { key, direction: sorted ? -direction : numeric ? -1 : 1 };
          renderLibraryTable();
        },
      },
      label,
      icon("up")
    )
  );
}

function renderLibraryTable() {
  const wrap = $("#library-table");
  if (!wrap) return;
  const pages = sortedPages();
  if (!pages.length) {
    wrap.replaceChildren(h("p", { class: "muted" }, t("no_filter_results", { q: state.libraryFilter })));
    return;
  }
  const largest = Math.max(...state.library.pages.map((page) => page.passages), 1);
  wrap.replaceChildren(
    h(
      "table",
      {},
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          sortHeader("title", t("col_page")),
          sortHeader("type", t("col_type")),
          sortHeader("passages", t("col_passages"), true),
          h("th", { class: "cell-bar", "aria-hidden": "true" }),
          h("th", { "aria-hidden": "true" })
        )
      ),
      h(
        "tbody",
        {},
        pages.map((page) =>
          h(
            "tr",
            {},
            h(
              "td",
              { class: "page-title-cell", lang: "en" },
              page.url ? h("a", { href: page.url, target: "_blank", rel: "noopener noreferrer" }, page.title) : page.title,
              page.custom && h("span", { class: "tag-added", lang: language() }, "+")
            ),
            h("td", {}, pageType(page.type)),
            h("td", { class: "num" }, page.passages ? formatNumber(page.passages) : h("span", { class: "muted" }, t("not_indexed"))),
            h(
              "td",
              { class: "cell-bar", "aria-hidden": "true" },
              page.passages ? h("span", { class: "mini-bar", style: { width: `${(page.passages / largest) * 100}%` } }) : null
            ),
            h("td", { class: "num" }, page.custom && !state.status?.demo ? removeButton(page) : null)
          )
        )
      )
    )
  );
}

function removeButton(page) {
  const button = h("button", { class: "link", type: "button" }, t("remove"));
  let armed = false;
  button.addEventListener("click", async () => {
    if (!armed) {
      armed = true;
      button.textContent = t("remove_confirm");
      return;
    }
    await api.removePage(page.title);
    await loadLibrary();
  });
  button.addEventListener("blur", () => {
    armed = false;
    button.textContent = t("remove");
  });
  return button;
}

async function addPage(title, type) {
  if (!title || state.adding) return;
  state.adding = true;
  state.addMessage = { text: t("add_fetching", { title }) };
  renderLibrary();
  try {
    for await (const event of api.addPage(title, type)) {
      if (event.event === "progress" && event.stage === "fetching") {
        state.addMessage = { text: t("add_fetching", { title }) };
      } else if (event.event === "progress") {
        state.addMessage = { text: t("add_embedding", { title }) };
      } else if (event.event === "done") {
        state.addMessage = { text: t("add_done", { title, n: formatNumber(event.passages) }) };
        state.addDraft = { title: "", type };
      } else if (event.event === "error") {
        const key = { not_found: "add_not_found", ambiguous: "add_ambiguous", busy: "add_busy" }[event.code] || "add_failed";
        state.addMessage = { error: true, text: t(key, { title, message: event.message }) };
      }
      updateAddMessage();
    }
  } catch (error) {
    state.addMessage = { error: true, text: t("add_failed", { message: error.message }) };
  }
  state.adding = false;
  await loadLibrary();
  renderLibrary();
}

function updateAddMessage() {
  const element = $("#view-library .form-message");
  if (!element) return;
  element.textContent = state.addMessage?.text || "";
  element.classList.toggle("is-error", Boolean(state.addMessage?.error));
}

// ---------------------------------------------------------------- measurements
const CATEGORY_ORDER = ["anonymous", "named", "list", "typo", "ambiguous", "compare", "turkish", "followup", "refusal"];

function dumbbell(split, metric, title, helpKey) {
  const before = split.baseline.categories;
  const after = split.current.categories;
  const rows = CATEGORY_ORDER.filter((name) => before[name]?.[metric] !== undefined && after[name]?.[metric] !== undefined);
  if (!rows.length) return null;
  const scored = rows.reduce((total, name) => total + after[name].questions, 0);
  const unscored = split.current.overall.questions - scored;
  return h(
    "section",
    { class: "section" },
    h("h2", {}, title),
    h("p", { class: "section-help" }, t(helpKey, { n: formatNumber(scored), m: formatNumber(unscored) })),
    h(
      "div",
      { class: "legend" },
      h("span", {}, h("i", { class: "dot" }), t("legend_before")),
      h("span", {}, h("i", { class: "dot is-after" }), t("legend_after"))
    ),
    h(
      "div",
      { class: "dumbbell" },
      rows.map((name) => {
        const from = before[name][metric];
        const to = after[name][metric];
        const low = Math.min(from, to);
        const tip = `${t(`cat_${name}`)}: ${formatPercent(from)} → ${formatPercent(to)} · ${t("tip_questions", { n: after[name].questions })}`;
        return h(
          "div",
          { class: "db-row", dataset: { tip } },
          h("span", { class: "db-name" }, t(`cat_${name}`), h("small", { class: "db-count" }, formatNumber(after[name].questions))),
          h(
            "div",
            { class: "db-track", "aria-hidden": "true" },
            from !== to &&
              h("i", {
                class: `db-stroke${to < from ? " is-worse" : ""}`,
                style: { left: `${low * 100}%`, width: `${Math.abs(to - from) * 100}%` },
              }),
            h("i", { class: "db-point", style: { left: `${from * 100}%` } }),
            h("i", { class: "db-point is-after", style: { left: `${to * 100}%` } })
          ),
          h("span", { class: "db-values" }, `${formatPercent(from)} → `, h("b", {}, formatPercent(to)))
        );
      }),
      h(
        "div",
        { class: "db-row is-axis", "aria-hidden": "true" },
        h("span"),
        h(
          "div",
          { class: "db-axis" },
          [0, 0.5, 1].map((tick) => h("span", { style: { left: `${tick * 100}%` } }, formatPercent(tick)))
        ),
        h("span")
      )
    )
  );
}

function qualityTable(split) {
  const before = split.baseline.overall;
  const after = split.current.overall;
  const rows = [
    ["q_answer_correct", "answer_correct", formatPercent],
    ["q_refusal_correct", "refusal_correct", formatPercent],
    ["q_false_refusal", "false_refusal", formatPercent],
    ["q_numbers_grounded", "numbers_grounded", formatPercent],
    ["q_cited_answers", "cited_answers", formatPercent],
  ];
  return h(
    "section",
    { class: "section" },
    h("h2", {}, t("quality_title")),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { style: { "margin-top": "10px" } },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { scope: "col" }, t("q_metric")),
            h("th", { class: "num", scope: "col" }, t("q_before")),
            h("th", { class: "num", scope: "col" }, t("q_after"))
          )
        ),
        h(
          "tbody",
          {},
          rows.map(([label, key, format]) =>
            h(
              "tr",
              {},
              h("td", {}, t(label)),
              h("td", { class: "num" }, format(before[key])),
              h("td", { class: "num" }, h("b", {}, format(after[key])))
            )
          )
        )
      )
    )
  );
}

function manualReview(review) {
  if (!review) return null;
  return h(
    "section",
    { class: "section" },
    h("h2", {}, t("review_title")),
    h(
      "p",
      { class: "section-help" },
      t("review_body", {
        read: formatNumber(review.answers_read),
        wrong: formatNumber(review.wrong_fact),
        turkish: formatNumber(review.turkish_answers),
        mixed: formatNumber(review.turkish_mixed_language),
        confused: formatNumber(review.turkish_confused),
      })
    )
  );
}

function speedTable(speed) {
  if (!speed) return null;
  const rows = [
    ["speed_search", "search_s"],
    ["speed_first", "first_word_s"],
    ["speed_total", "total_s"],
  ];
  return h(
    "section",
    { class: "section" },
    h("h2", {}, t("speed_title")),
    h("p", { class: "section-help" }, t("speed_help", { n: formatNumber(speed.questions) })),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { style: { "margin-top": "10px" } },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { scope: "col" }, t("q_metric")),
            h("th", { class: "num", scope: "col" }, t("q_before")),
            h("th", { class: "num", scope: "col" }, t("q_after"))
          )
        ),
        h(
          "tbody",
          {},
          rows.map(([label, key]) =>
            h(
              "tr",
              {},
              h("td", {}, t(label)),
              h("td", { class: "num" }, formatDuration(speed.baseline[key] * 1000)),
              h("td", { class: "num" }, h("b", {}, formatDuration(speed.current[key] * 1000)))
            )
          )
        )
      )
    )
  );
}

function ablationTable(steps) {
  if (!steps?.length) return null;
  return h(
    "section",
    { class: "section" },
    h("h2", {}, t("ablation_title")),
    h("p", { class: "section-help" }, t("ablation_help")),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { style: { "margin-top": "10px" } },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { scope: "col" }, t("ab_step")),
            h("th", { class: "num", scope: "col" }, t("ab_score")),
            h("th", { class: "num", scope: "col" }, t("ab_first"))
          )
        ),
        h(
          "tbody",
          {},
          steps.map((step) =>
            h(
              "tr",
              {},
              h("td", {}, step[language()] || step.en),
              h("td", { class: "num" }, formatPercent(step.retrieval_success)),
              h("td", { class: "num" }, formatPercent(step.hit_at_1))
            )
          )
        )
      )
    )
  );
}

function renderMetrics() {
  const root = $("#view-metrics");
  if (state.metrics === undefined) {
    root.replaceChildren(h("div", { class: "page-inner" }, h("h1", {}, t("metrics_title")), ...skeletons(10)));
    api.getMetrics().then((metrics) => {
      state.metrics = metrics;
      renderMetrics();
    });
    return;
  }
  const metrics = state.metrics;
  if (!metrics) {
    root.replaceChildren(h("div", { class: "page-inner" }, h("h1", {}, t("metrics_title")), h("p", { class: "page-intro" }, t("metrics_missing"))));
    return;
  }
  const split = metrics.splits[state.metricsSplit];
  const segment = (key, label) =>
    h(
      "button",
      {
        type: "button",
        "aria-pressed": String(state.metricsSplit === key),
        onclick: () => {
          state.metricsSplit = key;
          renderMetrics();
        },
      },
      label
    );
  root.replaceChildren(
    h(
      "div",
      { class: "page-inner" },
      h("h1", {}, t("metrics_title")),
      h(
        "p",
        { class: "page-intro" },
        t("metrics_intro", { total: formatNumber(metrics.questions), held: formatNumber(metrics.splits.test.current.overall.questions) })
      ),
      h("div", { class: "segmented", role: "group", "aria-label": t("metrics_split") }, segment("test", t("split_test")), segment("dev", t("split_dev"))),
      dumbbell(split, "retrieval_success", t("chart_retrieval"), "chart_retrieval_help"),
      dumbbell(split, "answer_rate", t("chart_answers"), "chart_answers_help"),
      qualityTable(split),
      manualReview(metrics.manual_review),
      speedTable(metrics.speed),
      ablationTable(metrics.ablation),
      h(
        "section",
        { class: "section" },
        h("h2", {}, t("method_title")),
        h("p", { class: "section-help" }, t("method_body")),
        h("pre", { class: "code-block" }, metrics.commands.join("\n")),
        h(
          "p",
          { class: "measured" },
          t("measured_on", {
            date: new Intl.DateTimeFormat(language(), { dateStyle: "long" }).format(new Date(metrics.date)),
            passages: formatNumber(metrics.passages),
            model: metrics.model,
            embed: metrics.embed_model,
          })
        )
      )
    )
  );
}

function wireTooltip() {
  const tip = $("#tooltip");
  document.addEventListener("pointermove", (event) => {
    const row = event.target.closest?.("[data-tip]");
    if (!row || event.pointerType !== "mouse") {
      tip.hidden = true;
      return;
    }
    tip.textContent = row.dataset.tip;
    tip.hidden = false;
    const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.max(event.clientY - tip.offsetHeight - 10, 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
}

// ------------------------------------------------------------------- navigation
function showView() {
  const view = ["library", "metrics"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "ask";
  state.view = view;
  $("#view-ask").hidden = view !== "ask";
  $("#view-library").hidden = view !== "library";
  $("#view-metrics").hidden = view !== "metrics";
  document.querySelectorAll(".nav a").forEach((link) => {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (view === "library") renderLibrary();
  if (view === "metrics") renderMetrics();
  document.title = view === "ask" ? "Wiki-RAG" : `${t(`nav_${view}`)} · Wiki-RAG`;
}

function renderEverything() {
  applyStaticText();
  renderStatus();
  renderThread();
  renderSheet();
  showView();
}

function wireControls() {
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    ask($("#question").value);
  });
  $("#send").addEventListener("click", () => {
    if (state.streaming) state.streaming.controller.abort();
  });
  const input = $("#question");
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(input.value);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.streaming) state.streaming.controller.abort();
    if (event.key === "/" && !event.target.matches("input, textarea, select") && !input.disabled) {
      event.preventDefault();
      location.hash = "#ask";
      input.focus();
    }
  });
  $("#new-chat").addEventListener("click", () => {
    state.streaming?.controller.abort();
    state.turns = [];
    state.active = null;
    renderThread();
    renderSheet();
    syncComposer();
    $("#question").focus();
  });
  $("#lang-toggle").addEventListener("click", () => {
    setLanguage(language() === "tr" ? "en" : "tr");
    renderEverything();
  });
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem("wiki-rag-theme", next);
    } catch {
      /* the choice just won't survive a reload */
    }
  });
  $("#model-select").addEventListener("change", (event) => {
    state.model = event.target.value;
    renderStatus();
  });
  $("#topk-range").addEventListener("input", (event) => {
    state.topK = Number(event.target.value);
    $("#topk-value").textContent = event.target.value;
  });
  window.addEventListener("hashchange", showView);
  const noteReaderScroll = () => {
    if (state.streaming) state.streaming.readerScrolled = true;
  };
  window.addEventListener("touchmove", noteReaderScroll, { passive: true });
  window.addEventListener("wheel", noteReaderScroll, { passive: true });
}

// ------------------------------------------------------------------------ start
setLanguage(detectLanguage());
applyTheme(document.documentElement.dataset.theme || "light");
wireControls();
wireHighlights();
wireTooltip();
renderEverything();
refreshStatus();
api.getSpecimen().then((specimen) => {
  state.specimen = specimen;
  if (!state.turns.length) renderThread();
});
