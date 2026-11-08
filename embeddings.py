"""Local embedding client using sentence-transformers.

Embeddings run fully locally (no API, no network) with a multilingual E5
model. E5 models are trained for asymmetric retrieval with text prefixes:
documents are embedded with "passage: " and queries with "query: ", which
maps directly onto the app's db/query embedding types.

The model (~450 MB) is downloaded from the Hugging Face Hub on first use and
cached locally after that. Loading is lazy so the web server starts fast.
"""

import threading
from typing import List

from config import load_config

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32

_model = None
_model_name = None
_model_lock = threading.Lock()


class EmbeddingError(Exception):
    pass


def _get_model():
    global _model, _model_name
    cfg = load_config()
    name = cfg.get("embedding_model") or DEFAULT_MODEL
    # legacy configs may still carry the MiniMax model name
    if name == "embo-01":
        name = DEFAULT_MODEL
    with _model_lock:
        if _model is None or _model_name != name:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            try:
                _model = SentenceTransformer(name)
                _model_name = name
            except Exception as exc:
                raise EmbeddingError(f"Failed to load embedding model '{name}': {exc}") from exc
        return _model


def _truncate(text: str, max_chars: int = 4000) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _encode(texts: List[str], prefix: str) -> List[List[float]]:
    model = _get_model()
    prepared = [prefix + _truncate(t.strip() or " ") for t in texts]
    try:
        vectors = model.encode(
            prepared,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
    return [[float(x) for x in vec] for vec in vectors]


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document-side texts (chunks, entities, relationships)."""
    return _encode(texts, prefix="passage: ")


def embed_query(text: str) -> List[float]:
    """Embed a user query."""
    return _encode([text], prefix="query: ")[0]
