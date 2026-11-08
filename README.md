# LightRAG-DuckDB

A graph-based Retrieval-Augmented Generation (RAG) web application: a LightRAG
clone built entirely on **DuckDB**, using **MiniMax-M3** for language modeling
and **local sentence-transformers** (`intfloat/multilingual-e5-small`) for
embeddings.

Unlike naive RAG systems that only retrieve isolated text chunks, this
application builds a **knowledge graph** from your documents and traverses it
at query time to discover **cross-document connections**.

![stack](https://img.shields.io/badge/python-3.9%2B-blue) ![db](https://img.shields.io/badge/storage-DuckDB-orange) ![license](https://img.shields.io/badge/license-MIT-green)

## How it works

### Indexing pipeline (`indexer.py`)

When you upload a document, the pipeline performs four steps. First, the text
is split into overlapping chunks that respect paragraph and sentence
boundaries. Second, MiniMax-M3 extracts entities (people, organizations,
concepts, technologies, ...) and the relationships between them from each
chunk. Third, every chunk, entity, and relationship is embedded locally with
the multilingual E5 model (`passage:` prefix). Fourth, everything is stored in
DuckDB with full graph connections, including `entity_chunks` provenance links
from each entity back to the chunks it was mentioned in.

The index is **incremental**: when a new document mentions an entity that
already exists in the graph (case-insensitive name match), its description is
enriched and re-embedded instead of creating a duplicate node. This entity
merging is what connects information *across* documents, so a relationship
extracted from document B can attach to an entity first seen in document A.
Relationship weights accumulate across documents, and deleting a document
removes only the entities that no other document still references.

### Dual-level retrieval (`retriever.py`)

Queries are answered with two cooperating retrieval levels:

| Level | Mechanism | What it finds |
|---|---|---|
| Low (local) | Cosine similarity search over entity and relationship embeddings, executed as SQL in DuckDB | Specific entities and their direct relations that match the question |
| High (global) | Graph traversal from the matched entities: 1-hop neighbors, then 2-hop neighbors | Broader connections and cross-document interdependencies that vector search alone would miss |

The retriever then collects entity descriptions, relationship descriptions, and
the original source chunks linked to the involved entities, assembles them into
a single context, and asks MiniMax-M3 to generate an answer. The answer cites
the entities and relationships it used with inline `[E#]` / `[R#]` markers, and
the UI shows the full retrieval trace (direct matches vs. 1-hop vs. 2-hop
discoveries).

### How a query is executed, step by step

There is **no BM25, keyword, or full-text search** in this application. All
text matching is dense **vector similarity only** (cosine similarity over
local E5 embeddings), combined with knowledge-graph traversal. A question
posted to `/api/query` goes through the following stages
(`retriever.retrieve`):

1. **Embed the question.** The query is embedded locally with the `query:`
   prefix of the E5 model (`embeddings.embed_query`), producing a 384-dim
   vector. No API call is involved.
2. **Low-level vector search (seeds).** Two brute-force cosine-similarity scans
   run as SQL inside DuckDB using `list_cosine_similarity` (no ANN index, every
   row is scored, which is fine at this scale):
   - top-K entities by entity-embedding similarity (`top_k_entities`, default 8);
   - top-K relationships by relationship-embedding similarity
     (`top_k_relationships`, default 8).

   The matched entities, plus both endpoints of every matched relationship,
   become the **seed set**.
3. **High-level graph traversal.** From the seeds, all touching relationships
   are followed to collect **1-hop** neighbor entities, and from those the
   traversal repeats once more for **2-hop** neighbors. This is pure SQL over
   the `relationships` table, with no similarity involved, and it is what
   surfaces cross-document connections that vector search alone would miss.
4. **Capping.** Entities are ranked seed -> 1-hop -> 2-hop and capped at
   `top_k_entities * 5`; relationships whose endpoints both survived are kept,
   sorted by descending weight, capped at `top_k_relationships * 5`.
5. **Source chunks (two paths, merged).** Chunks come from the union of:
   - **provenance**: chunks linked via `entity_chunks` to the seed and 1-hop
     entities, ordered by how many of those entities each chunk mentions;
   - **direct vector search**: top-K cosine scan over chunk embeddings
     (`top_k_chunks`, default 6).

   Duplicates are merged (provenance wins), capped at `top_k_chunks * 2`.
6. **Context assembly and answer.** Entities (labeled `direct match` /
   `1-hop neighbor` / `2-hop neighbor`), relationships, and chunk texts are
   formatted into a single context, truncated to `max_context_chars` (default
   24 000), and sent to MiniMax-M3, which writes the answer with inline
   `[E#]` / `[R#]` citations.

If hybrid retrieval (BM25 + vectors) is ever needed, DuckDB's `fts` extension
could add a keyword leg over `chunks.content`, but currently ranking is
entirely embedding-based.

### Storage (`database.py`)

All state lives in a single DuckDB file (`lightrag.duckdb`):

| Table | Columns |
|---|---|
| `documents` | id, filename, content, created_at |
| `chunks` | id, document_id, content, chunk_index, embedding `FLOAT[]` |
| `entities` | id, name, type, description, embedding `FLOAT[]` |
| `relationships` | id, source_entity_id, target_entity_id, description, weight, embedding `FLOAT[]` |
| `entity_chunks` | entity_id, chunk_id |

Vector search uses DuckDB's built-in `list_cosine_similarity` function directly
in SQL, so no external vector database is needed.

## Setup

Requirements: Python 3.9+ and a MiniMax API key (get one at
[platform.minimax.io](https://platform.minimax.io)).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp config.example.json config.json
# then paste your MiniMax API key into config.json,
# or export MINIMAX_API_KEY and set it on the Settings page instead

# 3. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
# or: python app.py

# 4. Open http://localhost:8000
```

`config.json` holds the API key and the tunable parameters and is **not** part
of version control. The Settings page writes to it, so the key can also be
entered from the browser after the first start.

## Using the app

1. **Documents** (`/upload`) - upload `.txt`/`.md` files or paste raw text.
   Indexing runs in the background; the page shows live progress (chunking ->
   embedding -> entity extraction per chunk). Documents can be deleted;
   orphaned entities and relationships are cleaned up automatically while
   shared entities survive.
2. **Query** (`/`) - ask a question. You get a Markdown answer with
   `[E#]`/`[R#]` citations plus a retrieval trace showing which entities were
   direct vector matches, 1-hop neighbors, or 2-hop discoveries, along with the
   relationships and source chunks used.
3. **Knowledge Graph** (`/graph`) - interactive vis.js visualization. Node
   color encodes entity type, node size encodes mention count, edge width
   encodes relationship weight, and entities appearing in more than one
   document are highlighted with an amber ring. Click a node for its
   description and source documents.
4. **Settings** (`/settings`) - API key, chunk size/overlap, and retrieval
   top-K parameters.

## MiniMax API integration

The LLM client (`llm.py`) uses the OpenAI Python SDK pointed at
`https://api.minimax.io/v1` with model `MiniMax-M3`, and disables thinking mode
via `extra_body={"thinking": {"type": "disabled"}}` for faster responses.
Extraction requests JSON mode, because M3 otherwise tends to drop keys.

Embeddings (`embeddings.py`) run fully locally with sentence-transformers using
`intfloat/multilingual-e5-small` - documents are embedded with the `passage:`
prefix and queries with the `query:` prefix, matching how E5 models were
trained for asymmetric retrieval. The model (~450 MB) is downloaded from the
Hugging Face Hub on first use and cached; no embeddings API or key is needed.

The client also works around a recurring M3 defect: it emits the value of a
missing `description` key as a bare string, so `_repair_missing_keys` puts the
key back before parsing, and a malformed response is retried once at a higher
temperature rather than replayed verbatim.

## Project structure

```
lightrag-duckdb/
├── app.py              # Main FastAPI app (pages + JSON APIs)
├── config.py           # JSON-file configuration management
├── database.py         # DuckDB schema, vector search, graph traversal SQL
├── embeddings.py       # Local sentence-transformers embeddings (passage/query modes)
├── llm.py              # MiniMax M3 client (extraction + answer generation)
├── indexer.py          # Document processing pipeline (chunk -> extract -> embed -> store)
├── retriever.py        # Dual-level retrieval with 1-hop/2-hop graph traversal
├── templates/          # Jinja2 HTML templates
│   ├── base.html       # Shared layout with sidebar navigation
│   ├── index.html      # Query interface
│   ├── upload.html     # Document upload & management
│   ├── graph.html      # vis.js knowledge graph viewer
│   └── settings.html   # API key & parameter settings
├── static/             # Front-end assets
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_pipeline.py       # Offline end-to-end test (mocked MiniMax APIs)
│   ├── test_retrieval_live.py # Retrieval test against the current database
│   └── seed_demo.py           # Seed/clean demo graph data for UI exploration
├── config.example.json        # Template for config.json (holds the API key)
├── requirements.txt
├── LICENSE
└── README.md
```

## Tests

The offline test suite mocks both the MiniMax API and the embedding model, so
you can verify the whole pipeline (chunking, entity merging, cosine search,
1-hop/2-hop traversal, provenance, context assembly, deletion cleanup) without
a key and without network access:

```bash
python3 tests/test_pipeline.py
```

The live test runs a real query against the indexed database with the real
model and prints the retrieval trace:

```bash
python3 tests/test_retrieval_live.py "Who rebuilt the Fjord Line?"
```

To explore the UI with sample data, stop the server, run
`python3 tests/seed_demo.py`, and start the server again
(`python3 tests/seed_demo.py clean` removes the demo data). Note that DuckDB
uses a single-writer lock, so scripts that open the database must run while the
server is stopped.

## License

MIT. See [LICENSE](LICENSE).
