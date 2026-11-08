"""Dual-level graph retrieval for LightRAG-DuckDB.

Low level  : vector similarity search over entities and relationships.
High level : graph traversal from the matched entities — 1-hop and 2-hop
             neighbors — to discover broader, cross-document connections.

The combined context (entity descriptions, relationship descriptions, and
linked source chunks) is sent to MiniMax M3 for answer generation with
inline [E#]/[R#] citations.
"""

from typing import Dict, List, Set, Tuple

import database as db
import embeddings as emb
import llm
from config import load_config


def _traverse_graph(seed_entity_ids: List[int]) -> Tuple[List[dict], List[dict], Set[int], Set[int]]:
    """Expand from seed entities to 1-hop and 2-hop neighbors.

    Returns (all_entities, all_relationships, one_hop_ids, two_hop_ids).
    """
    seed_set = set(seed_entity_ids)

    # --- 1-hop: relationships touching the seed entities
    hop1_rels = db.get_neighbors(list(seed_set))
    hop1_ids = set()
    for rel in hop1_rels:
        hop1_ids.add(rel["source_entity_id"])
        hop1_ids.add(rel["target_entity_id"])
    hop1_only = hop1_ids - seed_set

    # --- 2-hop: relationships touching the 1-hop frontier
    hop2_rels = db.get_neighbors(list(hop1_only)) if hop1_only else []
    hop2_ids = set()
    for rel in hop2_rels:
        hop2_ids.add(rel["source_entity_id"])
        hop2_ids.add(rel["target_entity_id"])
    hop2_only = hop2_ids - hop1_ids - seed_set

    # Deduplicate relationships (2-hop set includes some 1-hop edges)
    rels_by_id = {}
    for rel in hop1_rels + hop2_rels:
        rels_by_id[rel["id"]] = rel
    all_rels = sorted(rels_by_id.values(), key=lambda r: -(r["weight"] or 1.0))

    all_entity_ids = seed_set | hop1_ids | hop2_ids
    all_entities = db.get_entities_by_ids(list(all_entity_ids))
    return all_entities, all_rels, hop1_only, hop2_only


def retrieve(question: str) -> dict:
    """Run dual-level retrieval and return the assembled context package."""
    cfg = load_config()
    query_vec = emb.embed_query(question)

    # -- Low level: direct vector matches
    matched_entities = db.search_entities(query_vec, cfg["top_k_entities"])
    matched_rels = db.search_relationships(query_vec, cfg["top_k_relationships"])

    seed_ids = [e["id"] for e in matched_entities]
    for rel in matched_rels:  # matched relationships also seed the traversal
        seed_ids.extend([rel["source_entity_id"], rel["target_entity_id"]])
    seed_ids = list(dict.fromkeys(seed_ids))

    # -- High level: graph traversal (1-hop + 2-hop)
    all_entities, all_rels, hop1_ids, hop2_ids = _traverse_graph(seed_ids)

    # Cap the expansion to keep context manageable: seeds first, then 1-hop, then 2-hop
    seed_set = set(seed_ids)
    rank = {eid: 0 for eid in seed_set}
    rank.update({eid: 1 for eid in hop1_ids})
    rank.update({eid: 2 for eid in hop2_ids})
    all_entities.sort(key=lambda e: rank.get(e["id"], 3))
    max_entities = cfg["top_k_entities"] * 5
    all_entities = all_entities[:max_entities]
    kept_ids = {e["id"] for e in all_entities}

    max_rels = cfg["top_k_relationships"] * 5
    kept_rels = [
        r for r in all_rels
        if r["source_entity_id"] in kept_ids and r["target_entity_id"] in kept_ids
    ][:max_rels]

    # -- Source chunks: linked to seed + 1-hop entities, plus direct chunk search
    provenance_ids = [eid for eid in kept_ids if rank.get(eid, 3) <= 1]
    entity_chunks = db.get_chunks_for_entities(provenance_ids, limit=cfg["top_k_chunks"])
    vector_chunks = db.search_chunks(query_vec, cfg["top_k_chunks"])
    chunks_by_id = {c["id"]: c for c in entity_chunks}
    for c in vector_chunks:
        chunks_by_id.setdefault(c["id"], c)
    chunks = list(chunks_by_id.values())[:cfg["top_k_chunks"] * 2]

    return {
        "matched_entities": matched_entities,
        "matched_relationships": matched_rels,
        "entities": all_entities,
        "relationships": kept_rels,
        "chunks": chunks,
        "hop1_ids": list(hop1_ids & kept_ids),
        "hop2_ids": list(hop2_ids & kept_ids),
        "seed_ids": [eid for eid in seed_ids if eid in kept_ids],
    }


def _build_context(retrieval: dict, max_chars: int) -> str:
    """Format the retrieval package into the LLM context string with citation IDs."""
    lines = ["=== ENTITIES ==="]
    seed_set = set(retrieval["seed_ids"])
    hop1 = set(retrieval["hop1_ids"])
    for ent in retrieval["entities"]:
        origin = ("direct match" if ent["id"] in seed_set
                  else "1-hop neighbor" if ent["id"] in hop1 else "2-hop neighbor")
        lines.append(f"[E{ent['id']}] {ent['name']} ({ent['type']}, {origin}): {ent['description']}")

    lines.append("\n=== RELATIONSHIPS ===")
    for rel in retrieval["relationships"]:
        lines.append(
            f"[R{rel['id']}] {rel['source_name']} -> {rel['target_name']} "
            f"(weight {rel['weight']:.0f}): {rel['description']}"
        )

    lines.append("\n=== SOURCE TEXT ===")
    for chunk in retrieval["chunks"]:
        lines.append(f"[Chunk {chunk['id']} from \"{chunk['filename']}\"]\n{chunk['content']}\n")

    context = "\n".join(lines)
    return context[:max_chars]


def answer_question(question: str) -> dict:
    """Full query pipeline: retrieve context via the graph, then generate an answer."""
    cfg = load_config()
    retrieval = retrieve(question)

    if not retrieval["entities"] and not retrieval["chunks"]:
        return {
            "answer": ("The knowledge graph is empty — no entities or chunks matched your "
                       "question. Upload and index some documents first."),
            "retrieval": retrieval,
        }

    context = _build_context(retrieval, cfg["max_context_chars"])
    answer = llm.generate_answer(question, context)
    return {"answer": answer, "retrieval": retrieval}
