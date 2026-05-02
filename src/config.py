"""Central configuration for the Wiki-RAG system.

All tunable parameters, model names, paths, and the canonical lists of
entities to ingest live here so they can be changed in one place.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# --------------------------------------------------------------------------- #
# Ollama (local LLM + embeddings)                                             #
# --------------------------------------------------------------------------- #
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"          # 768-dim, fast, local
GEN_MODEL = "llama3.2:3b"                 # swap for "phi3" or "mistral" if you prefer
GEN_TEMPERATURE = 0.2                     # low — we want grounded, factual answers
GEN_MAX_TOKENS = 512

# --------------------------------------------------------------------------- #
# Vector store                                                                #
# --------------------------------------------------------------------------- #
# Design choice (Option B): one collection, with a `type` metadata field set
# to either "person" or "place". This lets us
#   1. handle "compare X and Y" queries that span both types in a single search,
#   2. share index infrastructure (one HNSW index, one connection),
#   3. still filter by type when the query is clearly about one or the other.
# See product_prd.md for the full rationale.
COLLECTION_NAME = "wiki_rag"

# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #
# Paragraph-aware fixed-size chunking with overlap.
# Chunk size is a soft target measured in characters (not tokens) — it's the
# greedy paragraph packer's budget.
CHUNK_SIZE = 800            # ~150-200 tokens for English
CHUNK_OVERLAP = 150         # ~30 tokens of overlap to preserve context across boundaries

# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
TOP_K = 5                   # chunks per type for routed queries
TOP_K_BOTH = 4              # chunks per type when query touches both (8 total)

# --------------------------------------------------------------------------- #
# Entities to ingest                                                          #
# --------------------------------------------------------------------------- #
# These are the Wikipedia page titles. They double as canonical names that the
# router uses for keyword-based query routing.
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
    # extras beyond the required 10:
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
    "Giza pyramid complex",   # Wikipedia title for "Pyramids of Giza"
    "Mount Everest",
    # extras beyond the required 10:
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
# Entity-name aliases used by the router                                      #
# --------------------------------------------------------------------------- #
# Maps lower-cased aliases the user might type → canonical Wikipedia title.
# Helps the router catch e.g. "Pyramids of Giza" → "Giza pyramid complex",
# or "Messi" → "Lionel Messi".
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
    "kahlo": "Frida Kahlo",
    "newton": "Isaac Newton",
    "darwin": "Charles Darwin",
    "gandhi": "Mahatma Gandhi",
    "ataturk": "Mustafa Kemal Atatürk",
    "atatürk": "Mustafa Kemal Atatürk",
    "van gogh": "Vincent van Gogh",
    "jobs": "Steve Jobs",
    "turing": "Alan Turing",
    "hawking": "Stephen Hawking",
    "mozart": "Mozart",
    "cleopatra": "Cleopatra",
    "swift": "Taylor Swift",

    "eiffel": "Eiffel Tower",
    "great wall": "Great Wall of China",
    "taj mahal": "Taj Mahal",
    "grand canyon": "Grand Canyon",
    "machu picchu": "Machu Picchu",
    "colosseum": "Colosseum",
    "hagia sophia": "Hagia Sophia",
    "ayasofya": "Hagia Sophia",
    "statue of liberty": "Statue of Liberty",
    "pyramids of giza": "Giza pyramid complex",
    "pyramids": "Giza pyramid complex",
    "everest": "Mount Everest",
    "stonehenge": "Stonehenge",
    "petra": "Petra",
    "christ the redeemer": "Christ the Redeemer (statue)",
    "niagara": "Niagara Falls",
    "fuji": "Mount Fuji",
    "sagrada": "Sagrada Família",
    "tower of london": "Tower of London",
    "acropolis": "Acropolis of Athens",
    "angkor": "Angkor Wat",
    "cappadocia": "Cappadocia",
    "kapadokya": "Cappadocia",
}

# --------------------------------------------------------------------------- #
# Router keyword cues                                                         #
# --------------------------------------------------------------------------- #
# Words that, in the absence of any matched entity name, hint at "person"
# vs "place". Used as a fallback signal.
PERSON_KEYWORDS = {
    "who", "person", "people", "scientist", "scientists", "artist", "artists",
    "writer", "writers", "physicist", "physicists", "mathematician", "mathematicians",
    "biologist", "biologists", "engineer", "engineers", "musician", "musicians",
    "singer", "singers", "actor", "actors", "actress", "philosopher",
    "philosophers", "leader", "president", "queen", "king", "born", "died",
    "discovered", "invented", "wrote", "painted", "composed", "played",
}
PLACE_KEYWORDS = {
    "where", "place", "places", "city", "country", "located", "location",
    "monument", "monuments", "landmark", "landmarks", "wonder", "wonders",
    "tower", "wall", "temple", "pyramid", "pyramids", "canyon", "mountain",
    "ruins", "site", "building", "tomb", "cathedral", "mosque", "church",
    "geography", "river", "valley",
}
