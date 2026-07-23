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

Session 7 follow-up (see tech-spec implementation log, "known limitation"
from session 6): a static `KNOWN_ALIAS_GROUPS` dict was added as a
*curated* hard-rules layer, checked before the DB-backed hard-rules tier.
It maps well-known surface-form variants (e.g. "Anthropic Claude",
"Claude CLI") to one preferred canonical name ("Claude") so that the
first-ever mention of any variant registers/looks up the *same* canonical
entity, instead of relying on embedding similarity (which failed to merge
"Claude" and "Anthropic Claude" in session 6's live test at the old 0.80
threshold). `EMBEDDING_MATCH_THRESHOLD` was also lowered 0.80 -> 0.70 for
surface forms not covered by the curated dict.
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
# considered the same canonical entity. Originally 0.80 (session 5,
# unverified). Session 6's live test found a real false-split at that
# threshold: "Anthropic Claude" and "Claude" scored below 0.80 and
# registered as two distinct canonical entities. Lowered to 0.70 in
# session 7 — still conservative enough to avoid merging clearly distinct
# tools/concepts, but permissive enough to catch that kind of
# name-prefix/suffix variation. See also `KNOWN_ALIAS_GROUPS` below, which
# handles specific known-problematic pairs deterministically instead of
# relying on the embedding model to score them highly.
EMBEDDING_MATCH_THRESHOLD = 0.70

# rapidfuzz token_sort_ratio (0-100) threshold for a fuzzy-match hit —
# catches near-identical strings (typos, abbreviations) that the embedding
# tier might not score highly enough (embeddings capture semantic
# similarity, not surface-form similarity).
FUZZY_MATCH_THRESHOLD = 88

# Curated hard-rules groups (session 7) — each inner list is a set of
# surface forms known to refer to the same canonical entity, with the
# FIRST element in each group used as the canonical name when none of the
# variants have been registered yet. Checked case-insensitively before the
# DB-backed `find_entity_by_alias` tier (which only matches forms already
# recorded from a *previous* resolution) so that the *first-ever* mention
# of any variant in a group resolves consistently to the same canonical
# name, e.g. seeing "Anthropic Claude" before "Claude" still ends up
# registering canonical entity "Claude", not "Anthropic Claude".
KNOWN_ALIAS_GROUPS: list[list[str]] = [
    ["Claude", "Anthropic Claude", "Claude CLI", "Claude Cli", "Claude Code"],
    ["LM Studio", "LMStudio", "LM-Studio"],
    ["Graphiti", "Graphiti Core", "graphiti-core"],
    ["Neo4j", "Neo4J", "Neo 4j"],
    ["Qdrant", "QDrant"],
    ["Postgres", "PostgreSQL", "Postgres SQL"],
    ["DSPy", "DSPy AI", "dspy-ai"],
]

# surface_form.lower() -> (canonical_name, all other known variants)
_ALIAS_LOOKUP: dict[str, tuple[str, list[str]]] = {}
for _group in KNOWN_ALIAS_GROUPS:
    _canonical_name = _group[0]
    for _variant in _group:
        _ALIAS_LOOKUP[_variant.lower()] = (_canonical_name, _group)


def _find_via_known_alias_group(surface_form: str) -> tuple[str, list[str]] | None:
    """Look up a curated alias group for this surface form.

    Returns `(canonical_name, all_variants)` if the surface form belongs to
    one of `KNOWN_ALIAS_GROUPS`, else None. The caller still needs to check
    Postgres for whether `canonical_name` (or any variant) is already
    registered — this only tells us *which* canonical name to use/register,
    it doesn't do the DB lookup itself.
    """
    return _ALIAS_LOOKUP.get(surface_form.strip().lower())


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

    # Tier 0 (session 7) — curated alias groups: if this surface form (or
    # its canonical name) is already registered under any variant in its
    # group, resolve to that; the DB-backed hard-rules tier below wouldn't
    # catch this on a variant's *first* mention (e.g. seeing "Anthropic
    # Claude" when only "Claude" is registered so far, with no alias link
    # between them yet).
    known_group = _find_via_known_alias_group(surface_form)
    if known_group is not None:
        canonical_name, variants = known_group
        for variant in variants:
            existing = await db.find_entity_by_alias(variant)
            if existing is not None:
                await db.add_entity_alias(existing["canonical_id"], surface_form)
                return existing["canonical_id"]
        # None of the group's variants are registered yet — register under
        # the group's preferred canonical name (not the raw surface form),
        # so future mentions of any variant resolve here immediately.
        return await _register_new_entity(canonical_name, entity.type)

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
