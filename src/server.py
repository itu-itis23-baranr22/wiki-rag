"""Local web server: the JSON and streaming API, plus the interface in `web/`.

It binds to 127.0.0.1 and rejects requests addressed to any other host name, so a
web page open in the same browser cannot reach the API through DNS rebinding.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.catalog import PAGE_TYPES
from src.config import EMBED_MODEL, MAX_TOP_K, OLLAMA_HOST, TOP_K, WEB_DIR
from src.ingest import index_page
from src.ollama_client import OllamaError
from src.rag import RagSystem
from src.vector_store import IndexMismatch
from src.wiki import AmbiguousPage, PageNotFound, load_page

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
_END = object()


class HistoryTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(default="", max_length=8000)
    entities: List[str] = Field(default_factory=list, max_length=20)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=MAX_TOP_K)
    model: Optional[str] = Field(default=None, max_length=200)
    history: List[HistoryTurn] = Field(default_factory=list, max_length=20)


class AddPageRequest(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    type: str = "other"


class LocalHostOnly:
    """Refuse requests addressed to any other host name.

    Written as plain ASGI so it stays out of the streaming path: the Stop button depends on
    a client disconnect reaching the response untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            host = dict(scope["headers"]).get(b"host", b"").decode("latin-1").rsplit(":", 1)[0].strip("[]")
            if host not in ALLOWED_HOSTS:
                response = JSONResponse({"detail": "This server only answers on the local machine."}, status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _sse(event: Dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _error_event(exc: Exception) -> Dict:
    code = "ollama" if isinstance(exc, OllamaError) else "internal"
    return {"event": "error", "code": code, "message": str(exc)}


async def _stream(request: Request, events: Iterator[Dict]):
    """Relay a blocking generator as server-sent events.

    Closing the generator when the browser disconnects also closes the request to
    Ollama, which is what makes the Stop button actually stop the model.
    """
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await run_in_threadpool(next, events, _END)
            except Exception as exc:  # reported to the page, which shows a recovery message
                yield _sse(_error_event(exc))
                break
            if event is _END:
                break
            yield _sse(event)
    finally:
        # A disconnect cancels this coroutine, and a plain await here would be cancelled
        # too, leaving the model writing to nobody. The shield lets the close finish.
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(events.close)


def create_app(rag_factory: Callable[[], RagSystem] = RagSystem, web_dir: Path = WEB_DIR) -> FastAPI:
    app = FastAPI(title="Wiki-RAG", docs_url=None, redoc_url=None, openapi_url=None)
    state: Dict = {"rag": None, "error": None}
    lock = threading.Lock()
    ingest_lock = threading.Lock()

    def get_rag() -> RagSystem:
        with lock:
            if state["rag"] is None:
                state["rag"] = rag_factory()
            return state["rag"]

    app.add_middleware(LocalHostOnly)

    @app.get("/api/status")
    def status() -> Dict:
        report: Dict = {
            "ollama": {"reachable": False, "host": OLLAMA_HOST, "models": []},
            "index": {"passages": 0, "pages": 0, "needs_rebuild": False},
            "defaults": {"top_k": TOP_K, "max_top_k": MAX_TOP_K, "model": None, "embed_model": EMBED_MODEL},
            "ready": False,
        }
        try:
            rag = get_rag()
        except IndexMismatch:
            report["index"]["needs_rebuild"] = True
            return report
        report["defaults"]["model"] = rag.generator.model
        stats = rag.store.entity_stats()
        report["index"].update(passages=sum(s["passages"] for s in stats.values()), pages=len(stats))
        try:
            installed = rag.client.installed_models()
        except OllamaError:
            return report
        report["ollama"].update(reachable=True, models=installed)
        wanted = {rag.generator.model, rag.embedder.model}
        have = {name for name in installed} | {name.split(":")[0] for name in installed}
        report["missing_models"] = sorted(wanted - have)
        report["ready"] = bool(stats) and not report["missing_models"]
        return report

    @app.get("/api/library")
    def library() -> Dict:
        rag = get_rag()
        stats = rag.store.entity_stats()
        pages = []
        for entry in rag.catalog.entries():
            info = stats.get(entry.title, {})
            pages.append(
                {
                    "title": entry.title,
                    "type": entry.type,
                    "custom": entry.custom,
                    "passages": info.get("passages", 0),
                    "url": info.get("url", ""),
                    "revision": info.get("revision"),
                }
            )
        return {"pages": pages, "total_passages": sum(p["passages"] for p in pages)}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        rag = await run_in_threadpool(get_rag)
        events = rag.ask_stream(
            body.question,
            top_k=body.top_k,
            model=body.model,
            history=[turn.model_dump() for turn in body.history],
        )
        return StreamingResponse(_stream(request, events), media_type="text/event-stream")

    def _add_page_events(title: str, page_type: str) -> Iterator[Dict]:
        rag = get_rag()
        if not ingest_lock.acquire(blocking=False):
            yield {"event": "error", "code": "busy", "message": "Another page is being added."}
            return
        try:
            yield {"event": "progress", "stage": "fetching", "title": title}
            try:
                page = load_page(title, refresh=True)
            except PageNotFound:
                yield {"event": "error", "code": "not_found", "message": title}
                return
            except AmbiguousPage:
                yield {"event": "error", "code": "ambiguous", "message": title}
                return

            entry = rag.catalog.get(title) or rag.catalog.add(title, page_type)
            updates: List[Dict] = []
            report = index_page(rag.store, rag.embedder, entry, page, on_progress=updates.append)
            yield {"event": "progress", "stage": "indexed", "title": title, "passages": report.passages}
            rag.retriever.refresh()
            yield {"event": "done", "title": title, "passages": report.passages}
        finally:
            ingest_lock.release()

    @app.post("/api/pages")
    async def add_page(body: AddPageRequest, request: Request) -> StreamingResponse:
        if body.type not in PAGE_TYPES:
            raise HTTPException(status_code=422, detail="Unknown page type")
        return StreamingResponse(
            _stream(request, _add_page_events(body.title.strip(), body.type)),
            media_type="text/event-stream",
        )

    @app.delete("/api/pages/{title:path}")
    def delete_page(title: str) -> Dict:
        rag = get_rag()
        entry = rag.catalog.get(title)
        if entry is None or not entry.custom:
            raise HTTPException(status_code=403, detail="Only pages you added can be removed.")
        with ingest_lock:
            rag.store.delete_entity(title)
            rag.catalog.remove(title)
            rag.retriever.refresh()
        return {"removed": title}

    if Path(web_dir).exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app()
