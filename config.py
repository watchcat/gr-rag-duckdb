"""Configuration management for LightRAG-DuckDB.

The MiniMax API key and tunable parameters are persisted to a local JSON
config file (config.json) so they survive restarts. Environment variables
are used only as an initial fallback when the config file does not yet
contain a key.
"""

import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DB_PATH = BASE_DIR / "lightrag.duckdb"

_DEFAUL_CONFIG = {
    "minimax_api_key": "",
    "minimax_group_id": "",
    "llm_model": "MiniMax-M3",
    "embedding_model": "intfloat/multilingual-e5-small",
    "api_base_url": "https://api.minimax.io/v1",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "top_k_entities": 8,
    "top_k_relationships": 8,
    "top_k_chunks": 6,
    "max_context_chars": 24000,
}

_lock = threading.Lock()


def load_config() -> dict:
    """Load config from the JSON file, merged over defaults."""
    cfg = dict(_DEFAUL_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    if not cfg.get("minimax_api_key"):
        env_key = os.environ.get("MINIMAX_API_KEY", "")
        if env_key:
            cfg["minimax_api_key"] = env_key
    if not cfg.get("minimax_group_id"):
        env_group = os.environ.get("MINIMAX_GROUP_ID", "")
        if env_group:
            cfg["minimax_group_id"] = env_group
    return cfg


def save_config(updates: dict) -> dict:
    """Persist updates into the JSON config file and return the new config."""
    with _lock:
        cfg = load_config()
        for key, value in updates.items():
            if key in _DEFAUL_CONFIG:
                cfg[key] = value
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({k: cfg[k] for k in _DEFAUL_CONFIG}, f, indent=2)
    return cfg


def get_api_key() -> str:
    return load_config().get("minimax_api_key", "")
