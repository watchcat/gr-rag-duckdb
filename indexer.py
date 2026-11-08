"""Document indexing pipeline for LightRAG-DuckDB.

Pipeline: chunk text -> extract entities & relationships per chunk via the
LLM -> merge duplicates into the existing knowledge graph (incremental
update) -> embed chunks, entities, relationships -> store in DuckDB.

Entity merging is the key to cross-document connections: when a new document
mentions an entity that already exists (case-insensitive name match), we
enrich its description instead of creating a duplicate node. Relationships
then naturally connect information across documents.
"""

import threading
import time
import uuid
from typing import Dict, List, Optional

import database as db
import embeddings as emb
import llm
from config import load_config

# ---------------------------------------------------------------------------
# In-memory job registry so the UI can poll processing status
# ---------------------------------------------------------------------------

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph and sentence
    boundaries near the target size."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # search backwards for a natural boundary in the last 40% of the window
            window_start = start + int(chunk_size * 0.6)
            best = -1
            for boundary in ["\n\n", "\n", ". ", "! ", "? ", "; "]:
                pos = text.rfind(boundary, window_start, end)
                if pos > best:
                    best = pos + len(boundary)
                    break
            if best > start:
                end = best
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Graph merge helpers (incremental updates)
# ---------------------------------------------------------------------------

def _merge_description(existing: str, new: str, max_len: int = 1500) -> str:
    existing = (existing or "").strip()
    new = (new or "").strip()
    if not existing:
        return new
    if not new or new.lower() in existing.lower():
        return existing
    merged = f"{existing} {new}"
    return merged[:max_len]


def _upsert_entity(ent: dict, entity_ids: Dict[str, int]) -> int:
    """Insert a new entity or merge into an existing one. Returns entity id."""
    key = ent["name"].lower()
    if key in entity_ids:
        return entity_ids[key]
    existing = db.find_entity_by_name(ent["name"])
    if existing:
        eid, _, _, old_desc = existing
        merged = _merge_description(old_desc, ent["description"])
        if merged != (old_desc or ""):
            vec = emb.embed_documents([f"{ent['name']}: {merged}"])[0]
            db.update_entity(eid, merged, vec)
        entity_ids[key] = eid
        return eid
    vec = emb.embed_documents([f"{ent['name']}: {ent['description']}"])[0]
    eid = db.insert_entity(ent["name"], ent["type"], ent["description"], vec)
    entity_ids[key] = eid
    return eid


def _upsert_relationship(src_id: int, tgt_id: int, src_name: str, tgt_name: str,
                         description: str, strength: float) -> None:
    existing = db.find_relationship(src_id, tgt_id)
    if existing:
        rid, old_desc, old_weight = existing
        merged = _merge_description(old_desc, description)
        new_weight = float(old_weight or 1.0) + strength
        vec = emb.embed_documents([f"{src_name} -> {tgt_name}: {merged}"])[0]
        db.update_relationship(rid, merged, new_weight, vec)
    else:
        vec = emb.embed_documents([f"{src_name} -> {tgt_name}: {description}"])[0]
        db.insert_relationship(src_id, tgt_id, description, strength, vec)


# ---------------------------------------------------------------------------
# Main indexing pipeline
# ---------------------------------------------------------------------------

def index_document(filename: str, content: str, job_id: str) -> None:
    """Run the full indexing pipeline for one document (called in a thread)."""
    cfg = load_config()
    try:
        _update_job(job_id, status="chunking", message="Splitting document into chunks...")
        chunks = chunk_text(content, cfg["chunk_size"], cfg["chunk_overlap"])
        if not chunks:
            _update_job(job_id, status="error", message="Document is empty.")
            return

        doc_id = db.insert_document(filename, content)
        _update_job(job_id, document_id=doc_id, total_chunks=len(chunks),
                    status="embedding_chunks", message=f"Embedding {len(chunks)} chunks...")

        chunk_vecs = emb.embed_documents(chunks)
        chunk_ids = []
        for i, (text, vec) in enumerate(zip(chunks, chunk_vecs)):
            chunk_ids.append(db.insert_chunk(doc_id, text, i, vec))

        entity_ids: Dict[str, int] = {}  # lower-name -> id cache for this document
        entities_found = 0
        relationships_found = 0

        for i, (chunk_id, text) in enumerate(zip(chunk_ids, chunks)):
            _update_job(
                job_id, status="extracting", processed_chunks=i,
                message=f"Extracting entities from chunk {i + 1}/{len(chunks)}...",
                entities=entities_found, relationships=relationships_found,
            )
            try:
                graph = llm.extract_entities_relationships(text)
            except llm.LLMError as exc:
                # Soft-fail on a single chunk; continue with the rest
                _update_job(job_id, message=f"Chunk {i + 1} extraction failed ({exc}); continuing...")
                continue

            chunk_entity_ids = {}
            for ent in graph["entities"]:
                eid = _upsert_entity(ent, entity_ids)
                chunk_entity_ids[ent["name"].lower()] = eid
                db.link_entity_chunk(eid, chunk_id)
            entities_found += len(chunk_entity_ids)

            for rel in graph["relationships"]:
                src_id = chunk_entity_ids.get(rel["source"].lower())
                tgt_id = chunk_entity_ids.get(rel["target"].lower())
                if src_id and tgt_id and src_id != tgt_id:
                    _upsert_relationship(src_id, tgt_id, rel["source"], rel["target"],
                                         rel["description"], rel["strength"])
                    relationships_found += 1

        _update_job(
            job_id, status="done", processed_chunks=len(chunks),
            entities=entities_found, relationships=relationships_found,
            message=(f"Indexed {len(chunks)} chunks, {entities_found} entity mentions, "
                     f"{relationships_found} relationships."),
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001 — report any failure to the UI
        _update_job(job_id, status="error", message=f"Indexing failed: {exc}")


def start_indexing(filename: str, content: str) -> str:
    """Kick off indexing in a background thread and return a job id."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "filename": filename, "status": "queued",
            "message": "Queued...", "total_chunks": 0, "processed_chunks": 0,
            "entities": 0, "relationships": 0, "started_at": time.time(),
        }
    thread = threading.Thread(target=index_document, args=(filename, content, job_id), daemon=True)
    thread.start()
    return job_id
