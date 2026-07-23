"""Step 5 · Entity Resolver — Layer 3.

Resolves each raw entity surface form extracted by the Memory Extractor
(e.g. "Claude CLI", "Claude", "Anthropic Claude") to a single canonical
entity, per spec Step 5. Strategy, per spec: "embeddings similarity ->
fuzzy string match -> hard rules dict". New entities are added to the
registry (Postgres `entities` table, mirrored into the Qdrant `entities`
collection for future similarity lookups).

Resolution order implemented here (cheapest/most-precise first, each tier
only runs if the previous one found nothing):

  1. Hard rules dict — exact case-insensitive match against a known
     canonical_name or alias (`common.db.find_entity_by_alias`). Handles
     "Claude CLI" == "Claude CLI" trivially and any manually-curated
     aliases.
  2. Embedding similarity — embed the surface form (same LM Studio
     nomic-embed-text-v1.5 endpoint used elsewhere) and search the Qdrant
     `entities` collection. A hit above `EMBEDDING_MATCH_THRESHOLD` is
     treated as the same entity (e.g. "Anthropic Claude" ~ "Claude").
  3. Fuzzy string match (rapidfuzz) — token-based ratio against every
     known canonical_name/alias in the registry snapshot. Catches typos/
     abbreviations embeddings might miss (e.g. "Qdrant" vs "Qdrnat").
  4. No match — register a brand-new canonical entity.

Every resolution (steps 1-3) also appends the surface form as a new alias
on the matched entity if it wasn't already known, so the registry grows
richer over time without manual curation.

Note on ordering vs. the spec's stated order ("embeddings similarity ->
fuzzy string match -> hard rules dict"): the spec lists embeddings first,
but hard rules are strictly cheaper (no embedding call) and strictly more
precise (exact match) than similarity search, so trying them first is a
pure optimization that produces identical resolutions for anything the
hard-rules dict actually covers, and falls through to the spec's stated
order for everything else.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common import db  # noqa: E402
from common.qdrant_client import (  # noqa: E402
    ENTITIES_COLLECTION,
    embed_text,
    ensure_collection,
    get_qdrant,
    point_id_for,
)
from qdrant_client.models import PointStruct  # noqa: E402
from schema.memory_event import Entity  # noqa: E402

logger = logging.getLogger("compiler.resolver")

# Cosine similarity threshold above which two entity surface forms are
# considered the same canonical entity. Not verified as extensively as the
# retrieval endpoint's semantic threshold (services/ingest/retrieval.py) —
# chosen conservatively high since a false-merge (two distinct entities
# treated as one) is worse for the graph than a false-split (duplicate
# entities that a human/future pass can merge later). Revisit with real
# resolver test cases once more entities accumulate.
EMBEDDING_MATCH_THRESHOLD = 0.80

# rapidfuzz token_sort_ratio (0-100) threshold for a fuzzy-match hit —
# catches near-identical strings (typos, abbreviations) that the embedding
# tier might not score highly enough (embeddings capture semantic
# similarity, not surface-form similarity).
FUZZY_MATCH_THRESHOLD = 88


async def _find_via_embedding(surface_form: str) -> str | None:
    try:
        await ensure_collection(ENTITIES_COLLECTION)
        vector = embed_text(surface_form)
        client = get_qdrant()
        response = await client.query_points(
            collection_name=ENTITIES_COLLECTION,
            query=vector,
            limit=1,
            with_payload=True,
            score_threshold=EMBEDDING_MATCH_THRESHOLD,
        )
    except Exception:
        logger.exception(
            "entity resolver: embedding search failed for %r, falling through",
            surface_form,
        )
        return None
    if not response.points:
        return None
    return response.points[0].payload.get("canonical_id")


def _find_via_fuzzy(surface_form: str, registry: list[dict]) -> str | None:
    best_id: str | None = None
    best_score = 0.0
    for entity in registry:
        candidates = [entity["canonical_name"], *(entity.get("aliases") or [])]
        for candidate in candidates:
            score = fuzz.token_sort_ratio(surface_form.lower(), candidate.lower())
            if score > best_score:
                best_score = score
                best_id = entity["canonical_id"]
    if best_score >= FUZZY_MATCH_THRESHOLD:
        return best_id
    return None


async def _register_new_entity(surface_form: str, type_: str | None) -> str:
    canonical_id = f"ent_{uuid.uuid4().hex[:12]}"
    await db.insert_entity(
        canonical_id=canonical_id,
        canonical_name=surface_form,
        type_=type_,
        aliases=[surface_form],
    )
    await _upsert_entity_embedding(canonical_id, surface_form, type_)
    logger.info(
        "entity resolver: registered new canonical entity %s (%r, type=%s)",
        canonical_id,
        surface_form,
        type_,
    )
    return canonical_id


async def _upsert_entity_embedding(
    canonical_id: str, canonical_name: str, type_: str | None
) -> None:
    """Mirror the canonical entity into the Qdrant `entities` collection so
    future surface forms can be resolved against it via embedding
    similarity (see `_find_via_embedding` above).
    """
    try:
        await ensure_collection(ENTITIES_COLLECTION)
        vector = embed_text(canonical_name)
        client = get_qdrant()
        await client.upsert(
            collection_name=ENTITIES_COLLECTION,
            points=[
                PointStruct(
                    id=point_id_for(canonical_id),
                    vector=vector,
                    payload={
                        "canonical_id": canonical_id,
                        "canonical_name": canonical_name,
                        "type": type_,
                    },
                )
            ],
        )
    except Exception:
        # Non-fatal: the entity is still registered in Postgres (source of
        # truth for the registry) even if the Qdrant mirror write fails —
        # worst case is a missed embedding-similarity hit on a future
        # surface form, which falls through to fuzzy match / new-entity
        # registration instead of hard-failing entity resolution.
        logger.exception(
            "entity resolver: failed to upsert entities collection point for %s",
            canonical_id,
        )


async def resolve_entity(entity: Entity) -> str:
    """Resolve one extracted `Entity` to a canonical_id, registering a new
    canonical entity if no existing match is found. Also records the
    surface form as a known alias on whichever entity it resolved to.
    """
    surface_form = entity.name.strip()
    if not surface_form:
        return await _register_new_entity(surface_form or "unknown", entity.type)

    hard_match = await db.find_entity_by_alias(surface_form)
    if hard_match is not None:
        return hard_match["canonical_id"]

    embedding_match_id = await _find_via_embedding(surface_form)
    if embedding_match_id is not None:
        await db.add_entity_alias(embedding_match_id, surface_form)
        return embedding_match_id

    registry = await db.get_all_entities()
    fuzzy_match_id = _find_via_fuzzy(surface_form, registry)
    if fuzzy_match_id is not None:
        await db.add_entity_alias(fuzzy_match_id, surface_form)
        return fuzzy_match_id

    return await _register_new_entity(surface_form, entity.type)


async def resolve_entities(entities: list[Entity]) -> list[Entity]:
    """Resolve every entity in a `MemoryItem.entities` list in place,
    populating `Entity.canonical_id`. Entities are resolved sequentially
    (not gathered concurrently) so that two near-duplicate surface forms
    appearing in the *same* item (e.g. "Qdrant" and "the vector database")
    can't both register as new entities in a race — the second lookup
    always sees the first's registration, at the cost of some latency
    (acceptable: extraction items are small lists, and this mirrors the
    already-sequential nature of the rest of the compiler pipeline).
    """
    resolved: list[Entity] = []
    for entity in entities:
        canonical_id = await resolve_entity(entity)
        resolved.append(entity.model_copy(update={"canonical_id": canonical_id}))
    return resolved
