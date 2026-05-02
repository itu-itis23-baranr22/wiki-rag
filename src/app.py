"""Streamlit chat UI for the Wiki-RAG system.

Run:
    streamlit run src/app.py

Design notes
------------
ChatGPT-style minimal layout:
  * Welcome screen is intentionally empty — just a centered title and the
    input. No example chips, no clutter.
  * Sidebar holds the four runtime settings (LLM, Top-K, Temperature, Chunk
    size info) plus the Clear-chat button.
  * Source chunks are tucked behind a single collapsed "Sources" expander
    under each assistant turn. Toggling it does NOT interrupt streaming
    because the chunks are only rendered AFTER the answer finishes.
  * Status while generating: a clean "Searching the index…" → "Generating
    answer…" status indicator.
"""
from __future__ import annotations

import time
from typing import List

import requests
import streamlit as st

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    GEN_MODEL,
    OLLAMA_HOST,
    TOP_K,
)
from src.rag import RagSystem


# --------------------------------------------------------------------------- #
# Page setup                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Wiki-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force a dark, ChatGPT-flavoured palette regardless of the user's system
# theme. We use Streamlit's theming via injected CSS — fewer surprises than
# relying on config.toml at the repo level.
st.markdown(
    """
<style>
    /* ------- Global dark palette ------- */
    .stApp {
        background-color: #212121;
        color: #ececec;
    }
    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2a2a2a;
    }

    /* Tighten the main column */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 8rem;
        max-width: 820px;
    }

    /* ------- Headings ------- */
    h1, h2, h3, h4 { color: #ececec !important; }
    .app-header {
        text-align: center;
        margin-top: 12vh;
        margin-bottom: 1.5rem;
    }
    .app-logo {
        font-size: 3.2rem;
        line-height: 1;
        margin-bottom: 0.6rem;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 600;
        margin: 0;
        color: #ececec;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #9b9b9b;
        margin-top: 6px;
    }

    /* ------- Chat messages ------- */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
    }
    /* Subtle bubble for user messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #2f2f2f !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Assistant messages stay flat (no bubble), like ChatGPT */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        padding: 0.4rem 0 !important;
    }

    /* ------- Input bar ------- */
    [data-testid="stChatInput"] {
        background: #2f2f2f;
        border-radius: 14px;
        border: 1px solid #404040;
    }
    [data-testid="stChatInput"] textarea {
        color: #ececec !important;
    }

    /* ------- Sidebar ------- */
    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem !important;
        font-weight: 700;
        color: #9b9b9b !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2a2a2a;
        color: #ececec;
        border: 1px solid #3a3a3a;
        font-size: 0.86rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #353535;
        border-color: #505050;
    }
    /* Sliders */
    section[data-testid="stSidebar"] [data-baseweb="slider"] {
        margin-top: 4px;
    }
    /* Selectbox */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
    }
    /* Captions in sidebar */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.78rem;
        color: #8a8a8a;
    }

    /* ------- Status indicator ------- */
    [data-testid="stStatus"] {
        background-color: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
    }

    /* ------- Sources expander ------- */
    [data-testid="stExpander"] {
        background: transparent;
        border: 1px solid #2f2f2f;
        border-radius: 8px;
        margin-top: 0.6rem;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.84rem;
        color: #9b9b9b;
    }

    .source-card {
        background: #2a2a2a;
        border-left: 2px solid #4a4a4a;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 6px;
    }
    .source-meta {
        font-size: 0.75rem;
        color: #9b9b9b;
        margin-bottom: 6px;
    }
    .source-text {
        font-size: 0.84rem;
        line-height: 1.55;
        color: #d0d0d0;
    }
    .source-link {
        margin-top: 6px;
        font-size: 0.74rem;
    }
    .source-link a {
        color: #7aa2f7;
        text-decoration: none;
    }
    .source-link a:hover { text-decoration: underline; }

    /* Hide Streamlit's hamburger menu and footer for a cleaner demo,
       but leave the header toolbar alone — that's where the sidebar
       collapse/expand button lives. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    /* Make the header strip blend into the dark background */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the local RAG system…")
def get_rag() -> RagSystem:
    return RagSystem()


@st.cache_data(ttl=60, show_spinner=False)
def list_ollama_models() -> List[str]:
    """Fetch installed models from Ollama. Falls back to a sensible default
    list if the API is not reachable."""
    try:
        r = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        # Filter out embedding-only models — heuristic by name.
        models = [m for m in models if "embed" not in m.lower()]
        if not models:
            models = [GEN_MODEL]
        # Make sure the configured default appears and is first.
        if GEN_MODEL in models:
            models = [GEN_MODEL] + [m for m in models if m != GEN_MODEL]
        else:
            models = [GEN_MODEL] + models
        return models
    except Exception:
        return [GEN_MODEL]


def render_sources(chunks: list) -> None:
    """Render retrieved chunks under an assistant turn, behind a collapsed
    expander. No-op if there are no chunks."""
    if not chunks:
        return
    with st.expander(f"📎 Sources ({len(chunks)})", expanded=False):
        for i, c in enumerate(chunks, start=1):
            meta = c.get("metadata") or {}
            etype = meta.get("type", "?")
            entity = meta.get("entity", "?")
            dist = c.get("distance", 0.0)
            src = meta.get("source", "")

            preview = c["document"][:500]
            if len(c["document"]) > 500:
                preview += "…"
            preview = preview.replace("<", "&lt;").replace(">", "&gt;")

            st.markdown(
                f"<div class='source-card'>"
                f"<div class='source-meta'>"
                f"<b>#{i}</b> &nbsp;·&nbsp; "
                f"<span style='color:#bdbdbd'>{etype}</span> · "
                f"<b>{entity}</b> &nbsp;·&nbsp; "
                f"distance {dist:.3f}"
                f"</div>"
                f"<div class='source-text'>{preview}</div>"
                + (
                    f"<div class='source-link'>"
                    f"<a href='{src}' target='_blank'>Read on Wikipedia →</a>"
                    f"</div>"
                    if src
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Boot the RAG system                                                         #
# --------------------------------------------------------------------------- #
try:
    rag = get_rag()
    db_count = rag.store.count()
except Exception as e:
    st.error(f"Could not initialise the RAG system: {e}")
    st.info("Make sure `ollama serve` is running and the models are pulled.")
    st.stop()


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Session state (initialise before the sidebar so the sidebar can read it)    #
# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# True while the script is in the middle of producing an answer. We use this
# to disable the sidebar inputs so they can't trigger a Streamlit rerun
# mid-generation (which would silently abort the in-flight answer).
is_generating = bool(st.session_state.get("awaiting_answer"))


# --------------------------------------------------------------------------- #
# Sidebar — settings                                                          #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Settings")

    if is_generating:
        st.caption("🔒 _Settings locked while an answer is being generated._")

    available_models = list_ollama_models()
    model_choice = st.selectbox(
        "Language model",
        options=available_models,
        index=0,
        help="Local LLM used to generate the answer. Pulled from Ollama.",
        disabled=is_generating,
    )

    top_k_choice = st.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=15,
        value=TOP_K,
        step=1,
        help="How many context chunks to retrieve per query.",
        disabled=is_generating,
    )

    st.markdown("**Chunk size**")
    st.caption(
        f"`{CHUNK_SIZE}` chars · overlap `{CHUNK_OVERLAP}` "
        f"_(set at ingest time — change in `src/config.py` and re-ingest to modify)_"
    )

    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True, disabled=is_generating):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        f"📊 **{db_count}** chunks indexed · embeddings `{EMBED_MODEL}`"
    )
    st.caption("💡 Runs entirely on your machine. No external APIs.")


# --------------------------------------------------------------------------- #
# Welcome header — only shown when there is no chat history                   #
# --------------------------------------------------------------------------- #
if not st.session_state.history:
    st.markdown(
        """
<div class='app-header'>
    <div class='app-logo'>📚</div>
    <h1 class='app-title'>Wiki-RAG</h1>
    <p class='app-subtitle'>
        Ask anything about famous people and famous places.<br>
        Local retrieval, local generation — nothing leaves your machine.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Render history                                                              #
# --------------------------------------------------------------------------- #
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("chunks", []))


# --------------------------------------------------------------------------- #
# Chat input                                                                  #
# --------------------------------------------------------------------------- #
user_query = st.chat_input(
    "Ask about a famous person or place…",
    disabled=is_generating,
)

# A chip click sets pending_query in session state and reruns. Pick that up
# here so the click acts like the user typed the question themselves.
if not user_query and st.session_state.get("pending_query"):
    user_query = st.session_state.pop("pending_query")


# Two-phase render to keep messages in the right visual order.
#
# Phase 1: a fresh user message has just arrived. Append it to history,
# stash it as the question to answer, and rerun. The next render will
# replay the entire history (including the new user message) in order.
#
# Phase 2: the rerun pass picks up `awaiting_answer` and runs the assistant
# turn at the bottom of the conversation, where it belongs.
if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.awaiting_answer = user_query
    st.rerun()


pending_question = st.session_state.pop("awaiting_answer", None)
if pending_question:
    with st.chat_message("assistant"):
        status_holder = st.empty()
        answer_holder = st.empty()

        chunks: list = []
        answer_parts: list[str] = []

        try:
            with status_holder.status(
                "🔍 Searching the index…", expanded=False
            ) as status:
                t0 = time.time()
                gen_started = False
                gen_t0 = 0.0
                retrieve_ms = 0.0

                for ev in rag.ask_stream(
                    pending_question,
                    top_k=top_k_choice,
                    model=model_choice,
                ):
                    if ev["event"] == "retrieved":
                        retrieve_ms = (time.time() - t0) * 1000
                        chunks = ev["chunks"]
                        status.update(
                            label=(
                                f"💭 Generating answer… "
                                f"_(retrieved {len(chunks)} chunks "
                                f"in {retrieve_ms:.0f} ms)_"
                            ),
                            state="running",
                        )
                        gen_t0 = time.time()
                        gen_started = True
                    elif ev["event"] == "token":
                        answer_parts.append(ev["token"])
                        answer_holder.markdown("".join(answer_parts) + "▌")
                    elif ev["event"] == "done":
                        gen_ms = (time.time() - gen_t0) if gen_started else 0.0
                        answer_holder.markdown("".join(answer_parts))
                        status.update(
                            label=(
                                f"✅ Done · retrieved in {retrieve_ms:.0f} ms · "
                                f"generated in {gen_ms:.1f} s"
                            ),
                            state="complete",
                        )
        except Exception as e:
            answer_holder.error(
                f"Generation failed: {e}\n\n"
                f"Make sure `ollama serve` is running."
            )
            answer_parts = [f"_Error: {e}_"]

        render_sources(chunks)

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "".join(answer_parts),
                "chunks": chunks,
            }
        )

    # Force a clean re-render so the sidebar inputs go back to enabled
    # (they were disabled while is_generating was True).
    st.rerun()