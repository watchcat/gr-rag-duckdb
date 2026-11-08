"""DuckDB schema and operations for LightRAG-DuckDB.

All persistent state lives in a single DuckDB file:
  - documents:      uploaded source documents
  - chunks:         text chunks with FLOAT[] embeddings
  - entities:       knowledge-graph nodes with FLOAT[] embeddings
  - relationships:  knowledge-graph edges with FLOAT[] embeddings
  - entity_chunks:  provenance links from entities back to source chunks

Cosine similarity search is executed directly in SQL using
`list_cosine_similarity`, so no external vector store is required.
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import duckdb

from config import DB_PATH

_conn: Optional[duckdb.DuckDBPyConnection] = None
_lock = threading.Lock()


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(DB_PATH))
        _init_schema(_conn)
    return _conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_documents START 1;
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_chunks START 1;
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_entities START 1;
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_relationships START 1;
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_documents'),
            filename VARCHAR NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT current_timestamp
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_chunks'),
            document_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding FLOAT[]
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_entities'),
            name VARCHAR NOT NULL,
            type VARCHAR,
            description TEXT,
            embedding FLOAT[]
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_relationships'),
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            description TEXT,
            weight DOUBLE DEFAULT 1.0,
            embedding FLOAT[]
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL
        );
    """)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_document(filename: str, content: str) -> int:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "INSERT INTO documents (filename, content, created_at) VALUES (?, ?, ?) RETURNING id",
            [filename, content, datetime.utcnow()],
        ).fetchone()
        return row[0]


def insert_chunk(document_id: int, content: str, chunk_index: int, embedding: List[float]) -> int:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "INSERT INTO chunks (document_id, content, chunk_index, embedding) VALUES (?, ?, ?, ?) RETURNING id",
            [document_id, content, chunk_index, embedding],
        ).fetchone()
        return row[0]


def find_entity_by_name(name: str):
    conn = get_conn()
    return conn.execute(
        "SELECT id, name, type, description FROM entities WHERE lower(name) = lower(?)",
        [name],
    ).fetchone()


def insert_entity(name: str, etype: str, description: str, embedding: List[float]) -> int:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "INSERT INTO entities (name, type, description, embedding) VALUES (?, ?, ?, ?) RETURNING id",
            [name, etype, description, embedding],
        ).fetchone()
        return row[0]


def update_entity(entity_id: int, description: str, embedding: List[float]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE entities SET description = ?, embedding = ? WHERE id = ?",
            [description, embedding, entity_id],
        )


def find_relationship(source_id: int, target_id: int):
    conn = get_conn()
    return conn.execute(
        """
        SELECT id, description, weight FROM relationships
        WHERE (source_entity_id = ? AND target_entity_id = ?)
           OR (source_entity_id = ? AND target_entity_id = ?)
        """,
        [source_id, target_id, target_id, source_id],
    ).fetchone()


def insert_relationship(source_id: int, target_id: int, description: str,
                        weight: float, embedding: List[float]) -> int:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            """INSERT INTO relationships (source_entity_id, target_entity_id, description, weight, embedding)
               VALUES (?, ?, ?, ?, ?) RETURNING id""",
            [source_id, target_id, description, weight, embedding],
        ).fetchone()
        return row[0]


def update_relationship(rel_id: int, description: str, weight: float, embedding: List[float]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE relationships SET description = ?, weight = ?, embedding = ? WHERE id = ?",
            [description, weight, embedding, rel_id],
        )


def link_entity_chunk(entity_id: int, chunk_id: int) -> None:
    with _lock:
        conn = get_conn()
        exists = conn.execute(
            "SELECT 1 FROM entity_chunks WHERE entity_id = ? AND chunk_id = ?",
            [entity_id, chunk_id],
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
                [entity_id, chunk_id],
            )


# ---------------------------------------------------------------------------
# Vector search (cosine similarity in DuckDB SQL)
# ---------------------------------------------------------------------------

def search_entities(query_embedding: List[float], top_k: int) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, name, type, description,
               list_cosine_similarity(embedding, ?::FLOAT[]) AS score
        FROM entities
        WHERE embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [query_embedding, top_k],
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "type": r[2], "description": r[3], "score": r[4]}
        for r in rows
    ]


def search_relationships(query_embedding: List[float], top_k: int) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT r.id, r.source_entity_id, r.target_entity_id, r.description, r.weight,
               es.name AS source_name, et.name AS target_name,
               list_cosine_similarity(r.embedding, ?::FLOAT[]) AS score
        FROM relationships r
        JOIN entities es ON es.id = r.source_entity_id
        JOIN entities et ON et.id = r.target_entity_id
        WHERE r.embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [query_embedding, top_k],
    ).fetchall()
    return [
        {
            "id": r[0], "source_entity_id": r[1], "target_entity_id": r[2],
            "description": r[3], "weight": r[4],
            "source_name": r[5], "target_name": r[6], "score": r[7],
        }
        for r in rows
    ]


def search_chunks(query_embedding: List[float], top_k: int) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, c.content, d.filename,
               list_cosine_similarity(c.embedding, ?::FLOAT[]) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [query_embedding, top_k],
    ).fetchall()
    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "filename": r[3], "score": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

def get_neighbors(entity_ids: List[int]) -> List[dict]:
    """Return all relationships touching any of the given entities (1 hop)."""
    if not entity_ids:
        return []
    conn = get_conn()
    placeholders = ",".join(["?"] * len(entity_ids))
    rows = conn.execute(
        f"""
        SELECT r.id, r.source_entity_id, r.target_entity_id, r.description, r.weight,
               es.name AS source_name, et.name AS target_name
        FROM relationships r
        JOIN entities es ON es.id = r.source_entity_id
        JOIN entities et ON et.id = r.target_entity_id
        WHERE r.source_entity_id IN ({placeholders})
           OR r.target_entity_id IN ({placeholders})
        """,
        entity_ids + entity_ids,
    ).fetchall()
    return [
        {
            "id": r[0], "source_entity_id": r[1], "target_entity_id": r[2],
            "description": r[3], "weight": r[4],
            "source_name": r[5], "target_name": r[6],
        }
        for r in rows
    ]


def get_entities_by_ids(entity_ids: List[int]) -> List[dict]:
    if not entity_ids:
        return []
    conn = get_conn()
    placeholders = ",".join(["?"] * len(entity_ids))
    rows = conn.execute(
        f"SELECT id, name, type, description FROM entities WHERE id IN ({placeholders})",
        entity_ids,
    ).fetchall()
    return [{"id": r[0], "name": r[1], "type": r[2], "description": r[3]} for r in rows]


def get_chunks_for_entities(entity_ids: List[int], limit: int = 10) -> List[dict]:
    """Retrieve source chunks linked to the given entities via entity_chunks."""
    if not entity_ids:
        return []
    conn = get_conn()
    placeholders = ",".join(["?"] * len(entity_ids))
    rows = conn.execute(
        f"""
        SELECT DISTINCT c.id, c.document_id, c.content, d.filename,
               count(ec.entity_id) OVER (PARTITION BY c.id) AS entity_hits
        FROM entity_chunks ec
        JOIN chunks c ON c.id = ec.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ec.entity_id IN ({placeholders})
        ORDER BY entity_hits DESC
        LIMIT ?
        """,
        entity_ids + [limit],
    ).fetchall()
    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "filename": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Listing / stats / graph export
# ---------------------------------------------------------------------------

def list_documents() -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT d.id, d.filename, length(d.content) AS size, d.created_at,
               (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
        FROM documents d ORDER BY d.created_at DESC
        """
    ).fetchall()
    return [
        {"id": r[0], "filename": r[1], "size": r[2],
         "created_at": str(r[3]), "chunk_count": r[4]}
        for r in rows
    ]


def delete_document(document_id: int) -> None:
    """Remove a document, its chunks and entity-chunk links.

    Entities/relationships are kept if still referenced by other documents;
    orphaned entities (no remaining chunk links) and their relationships
    are removed.
    """
    with _lock:
        conn = get_conn()
        conn.execute(
            "DELETE FROM entity_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
            [document_id],
        )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", [document_id])
        conn.execute("DELETE FROM documents WHERE id = ?", [document_id])
        # Remove orphaned entities and their relationships
        orphan_rows = conn.execute(
            "SELECT id FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM entity_chunks)"
        ).fetchall()
        orphans = [r[0] for r in orphan_rows]
        if orphans:
            ph = ",".join(["?"] * len(orphans))
            conn.execute(
                f"DELETE FROM relationships WHERE source_entity_id IN ({ph}) OR target_entity_id IN ({ph})",
                orphans + orphans,
            )
            conn.execute(f"DELETE FROM entities WHERE id IN ({ph})", orphans)


def get_stats() -> dict:
    conn = get_conn()
    stats = {}
    for table in ["documents", "chunks", "entities", "relationships"]:
        stats[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return stats


def get_full_graph() -> dict:
    """Export the whole knowledge graph for visualization.

    Each entity node carries the set of document ids it appears in so the
    frontend can highlight cross-document entities (appearing in >1 doc).
    """
    conn = get_conn()
    node_rows = conn.execute(
        """
        SELECT e.id, e.name, e.type, e.description,
               coalesce(list(DISTINCT c.document_id) FILTER (c.document_id IS NOT NULL), []) AS doc_ids,
               count(DISTINCT ec.chunk_id) AS mention_count
        FROM entities e
        LEFT JOIN entity_chunks ec ON ec.entity_id = e.id
        LEFT JOIN chunks c ON c.id = ec.chunk_id
        GROUP BY e.id, e.name, e.type, e.description
        """
    ).fetchall()
    edge_rows = conn.execute(
        "SELECT id, source_entity_id, target_entity_id, description, weight FROM relationships"
    ).fetchall()
    doc_rows = conn.execute("SELECT id, filename FROM documents").fetchall()
    return {
        "nodes": [
            {"id": r[0], "name": r[1], "type": r[2], "description": r[3],
             "doc_ids": sorted(r[4]) if r[4] else [], "mentions": r[5]}
            for r in node_rows
        ],
        "edges": [
            {"id": r[0], "source": r[1], "target": r[2], "description": r[3], "weight": r[4]}
            for r in edge_rows
        ],
        "documents": [{"id": r[0], "filename": r[1]} for r in doc_rows],
    }
