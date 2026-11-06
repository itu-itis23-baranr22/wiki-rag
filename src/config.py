"""Central configuration for Wiki-RAG.

Tunable parameters, model names, paths and the default page list live here so
they can be changed in one place. Values that vary per machine can be set with
environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


def normalize_ollama_host(value: str) -> str:
    """Return a usable base URL for an Ollama address.

    Accepts what people actually put in OLLAMA_HOST, such as "0.0.0.0:11434",
    and rewrites `localhost` to 127.0.0.1. On Windows, opening a new connection
    to `localhost` costs about two seconds when the server only listens on IPv4,
    which is the default for Ollama.
    """
    value = value.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parts = urlsplit(value)
    host = parts.hostname or "127.0.0.1"
    if host in {"localhost", "0.0.0.0", "::"}:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    port = parts.port or 11434
    return f"{parts.scheme}://{host}:{port}{parts.path}"


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("WIKI_RAG_DATA_DIR", PROJECT_ROOT / "data"))
INDEX_DIR = DATA_DIR / "index"
PAGES_DIR = DATA_DIR / "pages"
CUSTOM_PAGES_PATH = DATA_DIR / "custom_pages.json"
WEB_DIR = PROJECT_ROOT / "web"

# --------------------------------------------------------------------------- #
# Ollama (local LLM + embeddings)                                             #
# --------------------------------------------------------------------------- #
OLLAMA_HOST = normalize_ollama_host(
    os.getenv("WIKI_RAG_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
)
EMBED_MODEL = os.getenv("WIKI_RAG_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("WIKI_RAG_GEN_MODEL", "llama3.2:3b")
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS = 400
GEN_NUM_CTX = 4096  # set explicitly: older Ollama versions default to 2048 and cut the prompt
# How long Ollama keeps a model loaded after use. The server can also hold several GB of
# cached prompts while loaded, so a long value here costs real memory on a laptop.
KEEP_ALIVE = "5m"

# nomic-embed-text was trained with task prefixes and expects them on every input.
EMBED_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}
# Only the evaluation turns this off, to measure what the prefixes contribute.
EMBED_USE_PREFIXES = os.getenv("WIKI_RAG_EMBED_PREFIXES", "1") != "0"
EMBED_BATCH_SIZE = 32

# --------------------------------------------------------------------------- #
# Vector store                                                                #
# --------------------------------------------------------------------------- #
# One collection, with `type` metadata, so a question can span people and places.
COLLECTION_NAME = "wiki_rag_v2"

# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 800  # characters, hard limit including overlap
CHUNK_OVERLAP = 150  # whole sentences repeated from the previous chunk, up to this many characters
CHUNK_MIN = 120  # shorter trailing chunks merge into the previous one

# Wikipedia sections that hold no facts about the subject.
BOILERPLATE_SECTIONS = {
    "see also", "references", "notes", "footnotes", "citations", "bibliography",
    "further reading", "external links", "sources", "works cited", "explanatory notes",
    "notes and references", "general and cited sources", "primary sources",
}

# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
TOP_K = 5  # passages given to the model
MAX_TOP_K = 10  # the most the UI allows; keeps the prompt inside GEN_NUM_CTX
CANDIDATE_POOL = 40  # candidates each search method contributes before fusion
RRF_K = 60  # standard reciprocal rank fusion constant
VECTOR_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0
ENTITY_CAP = 2  # for questions naming no page: most passages from any one page

# --------------------------------------------------------------------------- #
# Default pages                                                               #
# --------------------------------------------------------------------------- #
# Wikipedia page titles. They double as names the router recognises.
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Mustafa Kemal Atatürk",
    "Cleopatra",
    "Vincent van Gogh",
    "Steve Jobs",
    "Alan Turing",
    "Stephen Hawking",
    "Mozart",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",
    "Mount Everest",
    "Stonehenge",
    "Petra",
    "Christ the Redeemer (statue)",
    "Niagara Falls",
    "Mount Fuji",
    "Sagrada Família",
    "Tower of London",
    "Acropolis of Athens",
    "Angkor Wat",
    "Cappadocia",
]

# --------------------------------------------------------------------------- #
# Names people type for those pages                                           #
# --------------------------------------------------------------------------- #
# Lower-case alias -> page title. Accents are ignored when matching, so "ataturk"
# and "atatürk" need only one entry.
ALIASES = {
    "messi": "Lionel Messi",
    "ronaldo": "Cristiano Ronaldo",
    "einstein": "Albert Einstein",
    "tesla": "Nikola Tesla",
    "curie": "Marie Curie",
    "da vinci": "Leonardo da Vinci",
    "leonardo": "Leonardo da Vinci",
    "shakespeare": "William Shakespeare",
    "ada": "Ada Lovelace",
    "lovelace": "Ada Lovelace",
    "kahlo": "Frida Kahlo",
    "newton": "Isaac Newton",
    "darwin": "Charles Darwin",
    "gandhi": "Mahatma Gandhi",
    "ghandi": "Mahatma Gandhi",
    "gandi": "Mahatma Gandhi",
    "ataturk": "Mustafa Kemal Atatürk",
    "kemal ataturk": "Mustafa Kemal Atatürk",
    "van gogh": "Vincent van Gogh",
    "jobs": "Steve Jobs",
    "turing": "Alan Turing",
    "hawking": "Stephen Hawking",
    "wolfgang amadeus mozart": "Mozart",
    "kleopatra": "Cleopatra",
    "swift": "Taylor Swift",
    "eiffel": "Eiffel Tower",
    "eyfel": "Eiffel Tower",
    "eyfel kulesi": "Eiffel Tower",
    "great wall": "Great Wall of China",
    "cin seddi": "Great Wall of China",
    "tac mahal": "Taj Mahal",
    "buyuk kanyon": "Grand Canyon",
    "kolezyum": "Colosseum",
    "ayasofya": "Hagia Sophia",
    "ozgurluk heykeli": "Statue of Liberty",
    "ozgurluk aniti": "Statue of Liberty",
    "pyramids of giza": "Giza pyramid complex",
    "pyramid of giza": "Giza pyramid complex",
    "great pyramid": "Giza pyramid complex",
    "pyramids": "Giza pyramid complex",
    "giza": "Giza pyramid complex",
    "giza piramitleri": "Giza pyramid complex",
    "keops piramidi": "Giza pyramid complex",
    "everest": "Mount Everest",
    "everest dagi": "Mount Everest",
    "christ the redeemer": "Christ the Redeemer (statue)",
    "cristo redentor": "Christ the Redeemer (statue)",
    "kurtarici isa": "Christ the Redeemer (statue)",
    "niagara": "Niagara Falls",
    "niagara selalesi": "Niagara Falls",
    "fuji": "Mount Fuji",
    "fuji dagi": "Mount Fuji",
    "sagrada": "Sagrada Família",
    "londra kulesi": "Tower of London",
    "acropolis": "Acropolis of Athens",
    "akropol": "Acropolis of Athens",
    "angkor": "Angkor Wat",
    "kapadokya": "Cappadocia",
}

# These aliases are also ordinary words ("what jobs did she have", "how swift",
# and "ada" means island in Turkish), so they only count when capitalised.
CASE_SENSITIVE_ALIASES = {"jobs": "Jobs", "swift": "Swift", "ada": "Ada"}

# --------------------------------------------------------------------------- #
# Router keyword cues                                                         #
# --------------------------------------------------------------------------- #
# Words that hint at "person" or "place" when a question names no page.
PERSON_KEYWORDS = {
    "who", "person", "people", "scientist", "scientists", "artist", "artists",
    "writer", "writers", "physicist", "physicists", "mathematician", "mathematicians",
    "biologist", "biologists", "engineer", "engineers", "musician", "musicians",
    "singer", "singers", "actor", "actors", "actress", "philosopher",
    "philosophers", "leader", "president", "queen", "king", "born", "died",
    "discovered", "invented", "wrote", "painted", "composed", "played",
    "footballer", "footballers", "painter", "composer",
}
PLACE_KEYWORDS = {
    "where", "place", "places", "city", "country", "located", "location",
    "monument", "monuments", "landmark", "landmarks", "wonder", "wonders",
    "tower", "wall", "temple", "pyramid", "pyramids", "canyon", "mountain",
    "ruins", "site", "building", "tomb", "cathedral", "mosque", "church",
    "geography", "river", "valley", "waterfall", "waterfalls", "statue", "mausoleum",
}
# Cues that suggest a type but are too weak to exclude pages: "where did the inventor
# of X work?" is about a person, even though it starts with "where".
WEAK_KEYWORDS = {
    "who", "where", "located", "location", "born", "died", "discovered", "invented",
    "wrote", "painted", "composed", "played", "site", "wall", "tower", "building", "river",
}
