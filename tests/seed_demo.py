#!/usr/bin/env python3
"""Seed the database with demo documents so the UI has something to show.

Usage:
    python3 tests/seed_demo.py          # add the demo documents
    python3 tests/seed_demo.py clean    # remove them again

The demo set is a small set of documents that mention each other, so the
knowledge graph has cross-document connections and the one-hop and two-hop
layers of the retrieval have something to traverse.

DuckDB takes a single-writer lock, so run this while the server is stopped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import database as db  # noqa: E402
import embeddings as emb  # noqa: E402
import indexer  # noqa: E402
import llm  # noqa: E402

DEMO_FILES = [
    ("demo-cities.md", """# The Northern Rail Corridor

The Northern Rail Corridor links Oslo and Bergen. It opened in 2004 after
seven years of construction. The Bergen Line carries most of the freight
between the two cities, while the Fjord Line handles passenger traffic.
Oslo is the capital of Norway, and Bergen is a UNESCO World Heritage city.
"""),
    ("demo-engineering.md", """# The Fjord Line Rebuild

The Fjord Line was rebuilt between 2019 and 2023. Engineers replaced the
original signalling with an ETCS Level 2 system supplied by a consortium
led by Siemens. The project was led by Norwegian consultant Norsk Rådgiv.
The Bergen Line shares its signalling with the Fjord Line since the
rebuild, which is why both lines are now described together in the
engineering documentation.
"""),
    ("demo-tourism.md", """# Travelling to Bergen

Bergen is reached by the Bergen Line from Oslo, or by ferry from
Stavanger. The railway was built by the Norwegian engineer Ole Bull, who
also worked on the Hardanger Line. Most visitors use the Bergen Railway
Museum, which is housed in the former station building near the harbour.
The Hardanger Line was never completed, so the Bergen Line remains the
only direct connection to the city.
"""),
]


def clean():
    """Remove the demo documents and anything that becomes orphaned."""
    rows = db.get_conn().execute(
        "SELECT id FROM documents WHERE filename LIKE 'demo-%'"
    ).fetchall()
    for (doc_id,) in rows:
        db.delete_document(doc_id)
    print(f"Removed {len(rows)} demo document(s).")


def seed():
    """Index the demo documents, reusing the real indexing pipeline."""
    existing = {
        r[0]
        for r in db.get_conn().execute("SELECT filename FROM documents").fetchall()
    }
    added = 0
    for filename, content in DEMO_FILES:
        if filename in existing:
            print(f"Skipping {filename} (already indexed).")
            continue
        print(f"Indexing {filename} ...")
        indexer.index_document(filename, content, "seed-demo")
        added += 1
    stats = db.get_stats()
    print(
        f"Done. {added} document(s) added. "
        f"Now: {stats['documents']} documents, {stats['entities']} entities, "
        f"{stats['relationships']} relationships."
    )


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if action == "clean":
        clean()
    elif action == "seed":
        seed()
    else:
        print(f"Unknown action {action!r}. Use 'seed' or 'clean'.", file=sys.stderr)
        sys.exit(2)
