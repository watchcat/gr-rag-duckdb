"""MiniMax M3 LLM client using the OpenAI-compatible SDK.

Used for two purposes:
  1. Entity / relationship extraction from text chunks (JSON output)
  2. Final answer generation from graph-retrieved context

Thinking mode is explicitly disabled via extra_body for faster responses.
"""

import json
import re
from typing import List

from openai import OpenAI

from config import load_config


class LLMError(Exception):
    pass


def _get_client() -> OpenAI:
    cfg = load_config()
    api_key = cfg.get("minimax_api_key", "")
    if not api_key:
        raise LLMError("MiniMax API key not configured. Set it on the Settings page.")
    return OpenAI(api_key=api_key, base_url=cfg.get("api_base_url", "https://api.minimax.io/v1"))


def chat(messages: List[dict], temperature: float = 0.3, max_tokens: int = 4096,
         json_object: bool = False) -> str:
    """Send a chat completion request to MiniMax-M3 with thinking disabled.

    json_object=True requests JSON mode (response_format), which MiniMax needs
    to produce well-formed JSON reliably; without it, M3 tends to drop keys.
    """
    cfg = load_config()
    client = _get_client()
    kwargs = {}
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(
            model=cfg.get("llm_model", "MiniMax-M3"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
            **kwargs,
        )
    except Exception as exc:  # surface API errors cleanly to the UI
        raise LLMError(f"MiniMax LLM request failed: {exc}") from exc
    if not response.choices:
        raise LLMError("MiniMax LLM returned no choices.")
    content = response.choices[0].message.content or ""
    return content.strip()


EXTRACTION_SYSTEM_PROMPT = """You are a knowledge-graph extraction engine. \
Given a text passage, extract the salient entities and the relationships between them.

Rules:
- Entities must be specific, canonical names (people, organizations, places, concepts, technologies, events, products).
- Use a short UPPERCASE type label for each entity, e.g. PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT, PRODUCT.
- Each entity needs a one-to-two sentence description grounded in the passage.
- Relationships connect two extracted entities and describe how they are related in the passage.
- Give each relationship a strength score from 1 (weak/mentioned together) to 10 (strong/explicit causal or structural link).
- Extract at most 15 entities and 20 relationships per passage. Prefer quality over quantity.

Respond with ONLY a JSON object in this exact format, no markdown fences, no commentary:
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relationships": [{"source": "...", "target": "...", "description": "...", "strength": 5}]}
The "source" and "target" fields must exactly match one of the extracted entity names."""


def _repair_missing_keys(text: str) -> str:
    """Fix MiniMax-M3's recurring JSON defect: it drops the "description" key
    and emits the value as a bare string, e.g. {"name": "X", "type": "Y", "..."}.
    Inserts the missing key before a string that directly follows another
    key-value pair without being a key itself."""
    return re.sub(
        r'("(?:type|name|target|source)"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*)"(?![a-zA-Z_]+"\s*:)',
        r'\1"description": "',
        text,
    )


def _parse_json_object(text: str) -> dict:
    """Robustly parse a JSON object from LLM output (strips fences, finds braces)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = [text]
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates += [_repair_missing_keys(c) for c in list(candidates)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"Could not parse JSON from LLM output: {text[:200]}")


def extract_entities_relationships(chunk_text: str) -> dict:
    """Extract entities and relationships from a text chunk. Returns a dict with
    'entities' and 'relationships' lists (empty lists on soft failure)."""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract the knowledge graph from this passage:\n\n{chunk_text}"},
    ]
    data = None
    last_error = None
    # retry with a higher temperature so a malformed output isn't replayed verbatim
    for temperature in (0.1, 0.6):
        raw = chat(messages, temperature=temperature, max_tokens=4096, json_object=True)
        try:
            data = _parse_json_object(raw)
            break
        except LLMError as exc:
            last_error = exc
    if data is None:
        raise last_error
    entities = []
    for ent in data.get("entities", []):
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        entities.append({
            "name": name,
            "type": str(ent.get("type", "CONCEPT")).strip().upper() or "CONCEPT",
            "description": str(ent.get("description", "")).strip(),
        })
    names = {e["name"].lower() for e in entities}
    relationships = []
    for rel in data.get("relationships", []):
        src = str(rel.get("source", "")).strip()
        tgt = str(rel.get("target", "")).strip()
        if not src or not tgt or src.lower() == tgt.lower():
            continue
        if src.lower() not in names or tgt.lower() not in names:
            continue
        try:
            strength = float(rel.get("strength", 5))
        except (TypeError, ValueError):
            strength = 5.0
        relationships.append({
            "source": src,
            "target": tgt,
            "description": str(rel.get("description", "")).strip(),
            "strength": max(1.0, min(10.0, strength)),
        })
    return {"entities": entities, "relationships": relationships}


ANSWER_SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions using a knowledge graph \
retrieved from the user's document collection.

You will receive context in three sections:
- ENTITIES: knowledge-graph nodes with descriptions (each has an ID like [E12])
- RELATIONSHIPS: knowledge-graph edges connecting entities (each has an ID like [R7])
- SOURCE TEXT: original document excerpts the entities were extracted from

Instructions:
- Answer the question accurately and thoroughly using ONLY the provided context.
- Cite the entities and relationships you used inline with their IDs, e.g. "MiniMax released M3 [E3] [R2]".
- Highlight connections that span multiple documents when relevant.
- If the context is insufficient to answer, say so clearly and state what is missing.
- Write in clear Markdown."""


def generate_answer(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    return chat(messages, temperature=0.4, max_tokens=4096)
