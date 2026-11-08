#!/usr/bin/env python3
"""Run a real query against the indexed database and print the retrieval trace.

Unlike test_pipeline.py this uses the real embedding model and the real
LLM, so it needs a configured MiniMax API key and the embedding model
downloaded (about 450 MB, cached afterwards).

Usage:
    python3 tests/test_retrieval_live.py
    python3 tests/test_retrieval_live.py "Who rebuilt the Fjord Line?"

DuckDB takes a single-writer lock, so run this while the server is stopped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import database as db  # noqa: E402
import retriever  # noqa: E402

DEFAULT_QUESTION = "Which company led the signalling upgrade?"

# Words that carry the meaning of the question, so the printed result shows
# the retrieved material rather than every word in common.
STOPWORDS = {
    "which", "who", "what", "when", "where", "why", "how", "did", "does",
    "was", "were", "the", "and", "for", "that", "this", "with", "from",
    "into", "about", "their", "there", "they", "them", "has", "have",
}


def _keywords(question):
    words = [w.lower().strip(".,?!") for w in question.split()]
    return [w for w in words if w and w not in STOPWORDS and len(w) > 2]


def run(question):
    stats = db.get_stats()
    if not stats["documents"]:
        print("The database holds no documents.")
        print("Seed it first:  python3 tests/seed_demo.py")
        return 1
    if not config.get_api_key():
        print("No MiniMax API key configured. Set one on the Settings page")
        print("or in config.json, or export MINIMAX_API_KEY.")
        return 1

    print(f"Question: {question}")
    print(f"Corpus: {stats['documents']} documents, {stats['entities']} entities, "
          f"{stats['relationships']} relationships")
    print()

    retrieval = retriever.retrieve(question)
    names = {e["id"]: e["name"] for e in retrieval["entities"]}

    def show(label, ids):
        found = [names.get(i, f"#{i}") for i in ids]
        print(f"  {label}: {', '.join(found) if found else '(none)'}")

    print("Retrieval trace")
    show("direct vector matches", retrieval["seed_ids"])
    show("one-hop neighbours", retrieval["hop1_ids"])
    show("two-hop neighbours", retrieval["hop2_ids"])
    print(f"  source chunks: {len(retrieval['chunks'])}")

    context = retriever._build_context(retrieval, config.load_config()["max_context_chars"])
    keywords = _keywords(question)
    hits = sum(1 for word in keywords if word in context.lower())
    print()
    print(f"Context: {len(context)} characters, "
          f"{hits}/{len(keywords)} question keywords present")

    answer = retriever.answer_question(question)
    print()
    print("Answer")
    print(answer["answer"])
    return 0


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    sys.exit(run(question))
