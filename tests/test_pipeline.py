#!/usr/bin/env python3
"""Offline end-to-end test for LightRAG-DuckDB.

Both the LLM and the embedding model are replaced with deterministic
stand-ins, so the whole pipeline runs without an API key and without
network access:

  chunking, entity merging, cosine search, 1-hop and 2-hop graph
  traversal, provenance, context assembly, deletion cleanup.

The DuckDB file is a temporary one, so the real database is never touched.
Run it with:

    python3 tests/test_pipeline.py
"""

import hashlib
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import database as db  # noqa: E402
import embeddings as emb  # noqa: E402
import indexer  # noqa: E402
import llm  # noqa: E402
import retriever  # noqa: E402

EMBED_DIM = 384

# Words that look like proper nouns but are not entities.
STOP = {
    "The", "This", "That", "Then", "When", "While", "After", "Before",
    "Some", "Each", "Both", "Here", "There", "First", "Second", "Once",
    "Under", "Over", "About", "Another", "Note", "Text", "Content",
}

PROPER = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")


def fake_embed(text):
    """Deterministic bag-of-words embedding.

    Tokens are hashed into a fixed number of buckets and the vector is
    normalised, so cosine similarity behaves like a real one for the
    purposes of these tests: identical tokens score high, different
    tokens score low, and nothing is downloaded.
    """
    vec = [0.0] * EMBED_DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        vec[int.from_bytes(digest[:4], "big") % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def fake_extract(chunk_text):
    """Extract proper nouns from a chunk and link consecutive ones.

    Descriptions carry the passage they came from, so the same entity
    mentioned in two documents produces two different descriptions. That
    is what the merging logic has to reconcile.
    """
    names = []
    for match in PROPER.finditer(chunk_text):
        word = match.group(0)
        if word in STOP:
            continue
        if word not in names:
            names.append(word)
    entities = [
        {
            "name": name,
            "type": "CONCEPT",
            "description": f"{name} (seen in: {chunk_text[:40].strip()})",
        }
        for name in names
    ]
    relationships = [
        {
            "source": source,
            "target": target,
            "description": f"{source} relates to {target}",
            "strength": 5,
        }
        for source, target in zip(names, names[1:])
    ]
    return {"entities": entities, "relationships": relationships}


class PipelineTest(unittest.TestCase):
    """Exercises the indexing and retrieval path end to end."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        tmp = Path(self.tmpdir.name)

        # A test config with small limits, so the tests are fast. The
        # top_k values are deliberately 1: a matched entity and a matched
        # relationship both seed the traversal, so a larger top_k would pull
        # every node into the seed set and leave the hop layers empty.
        cfg_path = tmp / "config.json"
        cfg_path.write_text(
            '{"minimax_api_key": "", "chunk_size": 200, "chunk_overlap": 20,'
            ' "top_k_entities": 1, "top_k_relationships": 1,'
            ' "top_k_chunks": 4, "max_context_chars": 2000}',
            encoding="utf-8",
        )

        patches = [
            mock.patch.object(config, "CONFIG_PATH", cfg_path),
            mock.patch.object(db, "DB_PATH", tmp / "test.duckdb"),
            mock.patch.object(emb, "embed_documents",
                              side_effect=lambda texts: [fake_embed(t) for t in texts]),
            mock.patch.object(emb, "embed_query",
                              side_effect=lambda q: fake_embed(q)),
            mock.patch.object(llm, "extract_entities_relationships",
                              side_effect=lambda text: fake_extract(text)),
            mock.patch.object(llm, "chat", side_effect=lambda *a, **k: "mocked"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        # The connection is a module global; drop it so each test gets a
        # fresh database.
        self.addCleanup(self._reset_connection)
        self._reset_connection()

    @staticmethod
    def _reset_connection():
        if db._conn is not None:
            db._conn.close()
            db._conn = None

    def _index(self, filename, content):
        indexer.index_document(filename, content, "test-job")
        stats = db.get_stats()
        self.assertNotEqual(stats["documents"], 0, "document was not indexed")
        return stats

    # ------------------------------------------------------------ chunking --

    def test_chunking_respects_boundaries(self):
        text = "Alpha one two three.\n\nBeta four five six.\n\nGamma seven."
        chunks = indexer.chunk_text(text, 40, 5)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 40)
        # A chunk should not end mid-word on a paragraph break.
        self.assertTrue(any("\n" in c for c in chunks))

    def test_chunking_splits_long_text(self):
        chunks = indexer.chunk_text("word " * 500, 100, 10)
        self.assertGreater(len(chunks), 1)

    def test_chunking_short_text_is_one_chunk(self):
        self.assertEqual(indexer.chunk_text("short text", 100, 10), ["short text"])

    # ------------------------------------------------------------- indexing --

    def test_indexing_stores_entities_and_relationships(self):
        stats = self._index("a.txt", "Acme meets Bolt.")
        self.assertEqual(stats["documents"], 1)
        self.assertGreaterEqual(stats["entities"], 1)
        self.assertGreaterEqual(stats["relationships"], 1)

    def test_entity_is_merged_across_documents(self):
        self._index("one.txt", "Acme meets Bolt.")
        self._index("two.txt", "Acme and Core are discussed.")

        entities = db.get_conn().execute(
            "SELECT name, description FROM entities WHERE name = 'Acme'"
        ).fetchall()
        self.assertEqual(len(entities), 1, "the entity was duplicated")
        # The description from the second document is merged in.
        self.assertIn("and Core", entities[0][1])

    def test_relationship_weight_accumulates(self):
        self._index("one.txt", "Acme meets Bolt.")
        first = db.get_conn().execute(
            "SELECT weight FROM relationships"
        ).fetchone()[0]
        self._index("two.txt", "Acme meets Bolt again.")
        second = db.get_conn().execute(
            "SELECT weight FROM relationships"
        ).fetchone()[0]
        self.assertGreater(second, first, "the weight did not accumulate")

    # ---------------------------------------------------------- vector search --

    def test_cosine_search_finds_the_right_entity(self):
        self._index("a.txt", "Acme meets Bolt.")
        query = fake_embed("Acme")
        hits = db.search_entities(query, 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["name"], "Acme")

    def test_vector_search_skips_rows_without_embeddings(self):
        self._index("a.txt", "Acme meets Bolt.")
        db.get_conn().execute(
            "UPDATE entities SET embedding = NULL WHERE name = 'Bolt'"
        )
        names = [h["name"] for h in db.search_entities(fake_embed("Bolt"), 5)]
        self.assertNotIn("Bolt", names)

    # -------------------------------------------------------- graph traversal --

    def test_graph_traversal_reaches_one_and_two_hops(self):
        self._index("chain.txt", "Acme meets Bolt. Bolt talks Core. Core sees Nova.")
        acme = db.get_conn().execute(
            "SELECT id FROM entities WHERE name = 'Acme'"
        ).fetchone()[0]

        entities, relationships, hop1, hop2 = retriever._traverse_graph([acme])

        hop1_names = {e["name"] for e in entities if e["id"] in hop1}
        hop2_names = {e["name"] for e in entities if e["id"] in hop2}
        self.assertIn("Bolt", hop1_names)
        self.assertIn("Core", hop2_names)
        self.assertNotIn("Nova", hop2_names, "traversal went further than two hops")
        self.assertTrue(relationships)

    def test_retrieve_returns_the_hop_layers(self):
        self._index("chain.txt", "Acme meets Bolt. Bolt talks Core. Core sees Nova.")
        result = retriever.retrieve("Acme")
        for key in ("matched_entities", "seed_ids", "hop1_ids", "hop2_ids",
                    "entities", "relationships", "chunks"):
            self.assertIn(key, result)
        self.assertTrue(result["seed_ids"])
        self.assertTrue(result["hop1_ids"])
        self.assertTrue(result["hop2_ids"])

    # ----------------------------------------------------------- provenance --

    def test_provenance_links_entities_to_chunks(self):
        self._index("a.txt", "Acme meets Bolt.")
        links = db.get_conn().execute(
            "SELECT count(*) FROM entity_chunks"
        ).fetchone()[0]
        self.assertGreater(links, 0, "no provenance links were written")

    def test_chunks_for_entities_returns_the_source(self):
        self._index("a.txt", "Acme meets Bolt.")
        acme = db.get_conn().execute(
            "SELECT id FROM entities WHERE name = 'Acme'"
        ).fetchone()[0]
        chunks = db.get_chunks_for_entities([acme], limit=5)
        self.assertTrue(chunks)
        self.assertIn("Acme", chunks[0]["content"])

    # ------------------------------------------------------ context assembly --

    def test_context_carries_citation_markers(self):
        self._index("a.txt", "Acme meets Bolt.")
        result = retriever.retrieve("Acme")
        context = retriever._build_context(result, 4000)
        self.assertIn("=== ENTITIES ===", context)
        self.assertIn("=== RELATIONSHIPS ===", context)
        self.assertIn("=== SOURCE TEXT ===", context)
        self.assertRegex(context, r"\[E\d+\]")
        self.assertRegex(context, r"\[R\d+\]")

    def test_context_is_truncated_to_the_limit(self):
        self._index("a.txt", "Acme meets Bolt. Bolt talks Core.")
        result = retriever.retrieve("Acme")
        context = retriever._build_context(result, 200)
        self.assertLessEqual(len(context), 200)

    # -------------------------------------------------------------- deletion --

    def test_delete_removes_orphans_and_keeps_shared_entities(self):
        self._index("one.txt", "Acme meets Bolt.")
        self._index("two.txt", "Acme and Core are discussed.")

        one_id = db.get_conn().execute(
            "SELECT id FROM documents WHERE filename = 'one.txt'"
        ).fetchone()[0]
        db.delete_document(one_id)

        names = {r[0] for r in db.get_conn().execute(
            "SELECT name FROM entities"
        ).fetchall()}
        self.assertIn("Acme", names, "a shared entity was deleted")
        self.assertNotIn("Bolt", names, "an orphaned entity survived")

    def test_delete_cleans_up_chunks_and_links(self):
        self._index("one.txt", "Acme meets Bolt.")
        doc_id = db.get_conn().execute(
            "SELECT id FROM documents WHERE filename = 'one.txt'"
        ).fetchone()[0]
        db.delete_document(doc_id)
        stats = db.get_stats()
        self.assertEqual(stats["documents"], 0)
        self.assertEqual(stats["chunks"], 0)
        self.assertEqual(
            db.get_conn().execute("SELECT count(*) FROM entity_chunks").fetchone()[0], 0
        )


class LLMOutputTest(unittest.TestCase):
    """The JSON handling around a model that has a known output defect."""

    def test_repair_inserts_the_missing_description_key(self):
        broken = '{"name": "Acme", "type": "CONCEPT", "some description"}'
        fixed = llm._repair_missing_keys(broken)
        self.assertIn('"description": "', fixed)
        self.assertEqual(llm._parse_json_object(fixed)["description"], "some description")

    def test_parse_accepts_fenced_json(self):
        fenced = '```json\n{"a": 1}\n```'
        self.assertEqual(llm._parse_json_object(fenced), {"a": 1})

    def test_parse_accepts_json_with_surrounding_prose(self):
        text = 'Here is the graph:\n{"a": 1}\nDone.'
        self.assertEqual(llm._parse_json_object(text), {"a": 1})

    def test_parse_raises_on_unparseable_output(self):
        with self.assertRaises(llm.LLMError):
            llm._parse_json_object("no json at all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
