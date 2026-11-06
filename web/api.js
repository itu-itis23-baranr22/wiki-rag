// Talks to the local server, or replays recorded answers when there is no server.

let demo = null;

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

async function loadDemo() {
  if (!demo) {
    const response = await fetch("data/demo.json");
    if (!response.ok) throw new Error("No recording available");
    demo = await response.json();
  }
  return demo;
}

export const isDemo = () => demo !== null;

export async function getStatus() {
  const forced = new URLSearchParams(location.search).has("demo");
  if (!forced) {
    try {
      const response = await fetch("api/status");
      if (response.ok && (response.headers.get("content-type") || "").includes("json")) {
        return { ...(await response.json()), demo: false };
      }
    } catch {
      /* no server here: fall through to the recording */
    }
  }
  try {
    const recording = await loadDemo();
    return { ...recording.status, demo: true };
  } catch {
    return { demo: false, ready: false, offline: true, ollama: { reachable: false, models: [] }, index: {}, defaults: {} };
  }
}

export async function getLibrary() {
  if (isDemo()) return (await loadDemo()).library;
  const response = await fetch("api/library");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// One real answer sentence and the source sentence behind it, taken from the recording,
// so the first screen shows the product's idea without inventing an example.
export async function getSpecimen() {
  let recording;
  try {
    const response = await fetch("data/demo.json");
    if (!response.ok) return null;
    recording = await response.json();
  } catch {
    return null;
  }
  for (const entry of recording.answers) {
    const done = entry.events.find((event) => event.event === "done");
    const sources = entry.events.find((event) => event.event === "sources")?.sources || [];
    if (!done || done.refused || done.language !== "en") continue;
    for (const sentence of done.attribution.paragraphs.flat()) {
      const evidence = sentence.evidence[0];
      if (sentence.inferred || !evidence || evidence.score < 0.6 || sentence.text.length > 110) continue;
      const source = sources.find((item) => item.number === evidence.source);
      const quote = source?.text.slice(evidence.start, evidence.end);
      if (quote && quote.length <= 180) {
        return { number: evidence.source, answer: sentence.text, source: quote, entity: source.entity, question: entry.question };
      }
    }
  }
  return null;
}

export async function getMetrics() {
  const response = await fetch("data/metrics.json");
  if (!response.ok) return null;
  return response.json();
}

async function* readEvents(response) {
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (block.startsWith("data: ")) yield JSON.parse(block.slice(6));
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

async function* replay(question, signal) {
  const recording = await loadDemo();
  const entry = recording.answers.find((a) => a.question === question);
  if (!entry) {
    yield { event: "error", code: "demo" };
    return;
  }
  // The wait is shortened so the replay stays watchable. The meta line still reports
  // the real timings, which come from the recording.
  for (const event of entry.events) {
    if (event.event === "route") await sleep(240, signal);
    if (event.event === "sources") await sleep(120, signal);
    if (event.event === "token") await sleep(16, signal);
    yield event;
    if (event.event === "sources") await sleep(900, signal);
  }
}

export async function* chat(body, signal) {
  if (isDemo()) {
    yield* replay(body.question, signal);
    return;
  }
  const response = await fetch("api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  yield* readEvents(response);
}

export async function* addPage(title, type, signal) {
  const response = await fetch("api/pages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, type }),
    signal,
  });
  yield* readEvents(response);
}

export async function removePage(title) {
  const response = await fetch(`api/pages/${encodeURIComponent(title)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
}
