"""Retrieval Service — Layer 7.

Answers "what memory is relevant to this new session?" by combining three
sources in parallel, per the tech spec's "Retrieval strategy (parallel)":

  1. Semantic search (Qdrant)   — embed the query, search the `facts`
     collection, filter status=active + score >= 0.75.
  2. Graph traversal (Graphiti) — `graphiti.search()` runs Graphiti's own
     hybrid (BM25 + cosine + BFS) search over RELATES_TO edges and returns
     the connected facts/entities. We resolve each returned edge's source
     Episodic node(s) back to our own `fact_id` (Graphiti episode `name` ==
     our `fact_id`, see graphiti_writer.py) so graph results can be merged
     with Qdrant/Postgres results by the same key.
  3. Temporal filter + status/type enrichment (PostgreSQL) — Postgres
     `facts` is the canonical source for status/confidence/session/agent;
     both other sources only return partial info (Qdrant has a full payload
     copy, but Postgres is authoritative for contradiction/status updates
     that might land there later from the Contradiction Checker).

Results are merged by `fact_id`, deduplicated, scored (semantic score *
recency weight, graph hits get a fixed centrality bonus), and assembled
into the `memory_packet` shape from the spec (Layer 7 "memory_packet
format").

Note: entities/open_tasks/recent_decisions sections use whatever facts were
already gathered from semantic+graph search, plus a small Postgres-only
top-up for open_tasks/recent_decisions if the search didn't surface any
task/decision facts — since we don't want an empty "open tasks" section
just because the query embedding didn't happen to score a task highly.

Session 7 follow-up: `recency_days` and `project_scope` (accepted on
`RetrieveRequest` per the spec's request schema but previously ignored,
see the tech-spec implementation log's long-standing "known limitation")
are now applied as post-filters — see `_apply_recency_days_filter` and
`_apply_project_scope_filter` below and their call sites in `retrieve()`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from common import db  # noqa: E402
from common.graphiti_config import get_graphiti  # noqa: E402
from common.qdrant_client import (  # noqa: E402
    FACTS_COLLECTION,
    embed_text,
    ensure_collection,
    get_qdrant,
)
from graphiti_core.nodes import EpisodicNode  # noqa: E402
from schema.retrieval import (  # noqa: E402
    MemoryPacket,
    MemoryPacketDecision,
    MemoryPacketEntity,
    MemoryPacketFact,
    MemoryPacketTask,
    RetrieveRequest,
)

logger = logging.getLogger("ingest.retrieval")

# Per spec Layer 7, strategy #1: "Filter: status = active, score >= 0.75".
# In practice, 0.75 is calibrated for OpenAI-style embedding models and is
# far too strict for `nomic-embed-text-v1.5` (verified with live queries
# against this deployment's LM Studio instance): unrelated queries score
# ~0.30-0.45 cosine similarity against this model, while genuinely relevant
# queries score ~0.6-0.95. 0.75 would filter out all but near-verbatim
# matches. Lowered to 0.55 as an empirically reasonable cutoff for this
# embedding model; kept as an env var so it can be re-tuned per model
# without a code change (e.g. if the embedding model is swapped later).
SEMANTIC_SCORE_THRESHOLD = float(os.environ.get("SEMANTIC_SCORE_THRESHOLD", "0.55"))
# Fixed bonus added to a fact's combined score for being reachable via graph
# traversal (Graphiti/Neo4j), independent of its semantic score — reflects
# "graph centrality" per the spec's ranking formula ("semantic + graph
# centrality + recency"). Chosen so a graph-only hit still ranks below a
# strong semantic hit but above a borderline one.
GRAPH_CENTRALITY_BONUS = 0.15
# Per spec Layer 7, strategy #3: recency_weight = 1.0 for last 7 days,
# decays to 0.5 at 90 days. Linear decay between those two anchor points,
# clamped to [0.5, 1.0] outside that range (older facts don't keep
# decaying further — spec only defines the two anchors).
RECENCY_FULL_WEIGHT_DAYS = 7
RECENCY_FLOOR_WEIGHT_DAYS = 90
RECENCY_FLOOR_WEIGHT = 0.5

DEFAULT_TOKEN_BUDGET = 2000  # spec default; ~4 chars/token heuristic below
_APPROX_CHARS_PER_TOKEN = 4


def _recency_weight(
    created_at: Optional[datetime], full_weight_days: int = RECENCY_FULL_WEIGHT_DAYS
) -> float:
    """Linear recency decay: 1.0 for facts within `full_weight_days`,
    decaying to `RECENCY_FLOOR_WEIGHT` by `RECENCY_FLOOR_WEIGHT_DAYS`.

    `full_weight_days` defaults to the spec's fixed anchor (7) but can be
    overridden per-request via `RetrieveRequest.recency_days` (session 7 —
    see `retrieve()`), which lets a caller ask for a stronger or weaker
    recency bias without changing the floor anchor (90 days / 0.5) that
    the spec also defines. If a caller-supplied `recency_days` is >= the
    floor anchor, the floor anchor is pushed out to match so the decay
    curve stays monotonic (no floor-before-full-weight inversion).
    """
    if created_at is None:
        return RECENCY_FLOOR_WEIGHT
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
    floor_days = max(RECENCY_FLOOR_WEIGHT_DAYS, full_weight_days + 1)
    if age_days <= full_weight_days:
        return 1.0
    if age_days >= floor_days:
        return RECENCY_FLOOR_WEIGHT
    span = floor_days - full_weight_days
    frac = (age_days - full_weight_days) / span
    return 1.0 - frac * (1.0 - RECENCY_FLOOR_WEIGHT)


async def _semantic_search(query: str, top_k: int) -> dict[str, float]:
    """Layer 5 — semantic search over the `facts` Qdrant collection.

    Returns {fact_id: raw_cosine_score} for hits scoring >= threshold.
    """
    try:
        await ensure_collection(FACTS_COLLECTION)
        vector = embed_text(query)
        client = get_qdrant()
        response = await client.query_points(
            collection_name=FACTS_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
            score_threshold=SEMANTIC_SCORE_THRESHOLD,
        )
    except Exception:
        logger.exception("semantic search failed, continuing without it")
        return {}

    scores: dict[str, float] = {}
    for point in response.points:
        payload = point.payload or {}
        fact_id = payload.get("fact_id")
        if fact_id and payload.get("status", "active") == "active":
            scores[fact_id] = max(scores.get(fact_id, 0.0), point.score)
    return scores


async def _graph_traversal(query: str, top_k: int) -> set[str]:
    """Layer 4 — graph traversal via Graphiti's hybrid search.

    Graphiti's `search()` returns `EntityEdge`s (RELATES_TO relationships),
    each carrying `episodes: list[uuid]` — the Graphiti Episodic node(s)
    that produced that edge. Our Graphiti writer (graphiti_writer.py) names
    each episode after our own `fact_id`, so resolving those episode UUIDs
    back to `EpisodicNode.name` gives us the `fact_id`s to merge with the
    other two sources.
    """
    try:
        graphiti = get_graphiti()
        edges = await graphiti.search(query, num_results=top_k)
    except Exception:
        logger.exception("graph traversal failed, continuing without it")
        return set()

    episode_uuids = {uuid for edge in edges for uuid in edge.episodes}
    if not episode_uuids:
        return set()

    try:
        episodes = await EpisodicNode.get_by_uuids(graphiti.driver, list(episode_uuids))
    except Exception:
        logger.exception("failed to resolve graph episodes to fact_ids")
        return set()

    # episode.name == fact_id, per graphiti_writer.write_memory_item()
    return {ep.name for ep in episodes}


def _combined_score(
    fact_id: str,
    semantic_scores: dict[str, float],
    graph_fact_ids: set[str],
    created_at: Optional[datetime],
    full_weight_days: int = RECENCY_FULL_WEIGHT_DAYS,
) -> float:
    base = semantic_scores.get(fact_id, 0.0)
    if fact_id in graph_fact_ids:
        base += GRAPH_CENTRALITY_BONUS
    return base * _recency_weight(created_at, full_weight_days)


async def _fact_texts_from_qdrant(fact_ids: list[str]) -> dict[str, str]:
    """Fetch fact `text` for display — Postgres doesn't store the text
    itself (only Graphiti/Qdrant do), so pull it from the Qdrant payload
    we already wrote alongside each point (see qdrant_writer.py).
    """
    if not fact_ids:
        return {}
    try:
        client = get_qdrant()
        points = await client.retrieve(
            collection_name=FACTS_COLLECTION,
            ids=_qdrant_point_ids(fact_ids),
            with_payload=True,
        )
    except Exception:
        logger.exception("failed to fetch fact texts from qdrant")
        return {}
    return {
        p.payload.get("fact_id"): p.payload.get("text", "") for p in points if p.payload
    }


def _qdrant_point_ids(fact_ids: list[str]) -> list[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, fid)) for fid in fact_ids]


async def _apply_project_scope_filter(
    scored: list[tuple[str, float, dict]], project_scope: str
) -> list[tuple[str, float, dict]]:
    """Session 7 — narrow `scored` to facts matching `project_scope`.

    Matches (case-insensitive substring) against either:
      - the fact's own text (fetched from Qdrant payload, same as the
        final assembly step needs anyway), or
      - any of its resolved entity canonical names/aliases.

    This is a best-effort topic filter, not a structured field lookup —
    the spec's Layer 6 `facts` table has no dedicated "project" column, so
    there's no exact-match alternative available without a schema change.
    """
    if not scored:
        return scored
    needle = project_scope.strip().lower()
    if not needle:
        return scored

    fact_ids = [fid for fid, _, _ in scored]
    texts = await _fact_texts_from_qdrant(fact_ids)

    all_entity_ids = {eid for _, _, row in scored for eid in (row["entity_ids"] or [])}
    entities = await db.get_entities_by_ids(list(all_entity_ids))
    entity_haystack_by_id = {
        e["canonical_id"]: " ".join(
            [e["canonical_name"] or ""] + list(e.get("aliases") or [])
        ).lower()
        for e in entities
    }

    filtered = []
    for fact_id, score, row in scored:
        haystack = texts.get(fact_id, "").lower()
        for entity_id in row["entity_ids"] or []:
            haystack += " " + (
                entity_haystack_by_id.get(entity_id) or entity_id.lower()
            )
        if needle in haystack:
            filtered.append((fact_id, score, row))
    return filtered


async def retrieve(request: RetrieveRequest) -> MemoryPacket:
    """Run the parallel retrieval strategy and assemble a `memory_packet`."""
    # Session 7: genuinely run semantic search (Qdrant) and graph
    # traversal (Graphiti/Neo4j) concurrently via `asyncio.gather` instead
    # of sequential awaits — see the tech-spec implementation log's
    # session 4/5 "known limitation" (both are independent I/O-bound calls
    # to local services, so there's no reason to serialize them).
    semantic_scores, graph_fact_ids = await asyncio.gather(
        _semantic_search(request.query, request.top_k),
        _graph_traversal(request.query, request.top_k),
    )

    candidate_ids = set(semantic_scores) | graph_fact_ids
    fact_rows = await db.get_facts_by_ids(list(candidate_ids))
    fact_by_id = {row["fact_id"]: row for row in fact_rows}

    # Session 7: `recency_days` narrows the "full weight" recency anchor
    # (spec's fixed 7-day anchor) to whatever the caller asks for, instead
    # of being silently ignored. A caller passing e.g. `recency_days: 1`
    # is asking for a *stronger* recency bias (only today's facts stay at
    # weight 1.0); `recency_days: 30` asks for a gentler one. Defaults to
    # the spec's built-in 7-day anchor when omitted.
    full_weight_days = request.recency_days or RECENCY_FULL_WEIGHT_DAYS

    # Per spec strategy #3: exclude status=outdated unless the query
    # explicitly asks for history. We don't yet have a history-query
    # detector (no such flag on RetrieveRequest per spec's example), so for
    # now this always excludes non-active facts, matching the default case.
    scored: list[tuple[str, float, dict]] = []
    for fact_id in candidate_ids:
        row = fact_by_id.get(fact_id)
        if row is None:
            # In Qdrant/graph but not (yet) in Postgres — e.g. Postgres
            # insert step hasn't run yet in a retry race. Skip rather than
            # guess status/confidence.
            continue
        if row["status"] != "active":
            continue
        if request.source_agent and row["source_agent"] != request.source_agent:
            # source_agent filter is a *display* narrowing, not exclusion
            # from shared memory (spec: memory is shared by default) — but
            # since the spec's request schema explicitly offers
            # `source_agent`, honor it as an opt-in filter when provided.
            continue
        score = _combined_score(
            fact_id,
            semantic_scores,
            graph_fact_ids,
            row["created_at"],
            full_weight_days,
        )
        scored.append((fact_id, score, row))

    scored.sort(key=lambda t: t[1], reverse=True)

    # Session 7: `project_scope` — the spec's example ("narrow to topic")
    # doesn't map to any existing column on `facts` (no "project" field),
    # so this is implemented as a case-insensitive substring match against
    # either the fact's text or any of its resolved entity names/aliases.
    # Applied *after* scoring/sorting (it's a hard filter, not a ranking
    # signal) and before the top_k cut, so a narrow scope doesn't waste the
    # budget on facts that get filtered out anyway.
    if request.project_scope:
        scored = await _apply_project_scope_filter(scored, request.project_scope)

    scored = scored[: request.top_k]

    fact_texts = await _fact_texts_from_qdrant([fid for fid, _, _ in scored])

    # Resolve entity_ids (canonical_id post-Entity-Resolver, or a raw
    # surface-form string on older pre-resolver facts) to display
    # {canonical_name, type} — see common.db.get_entities_by_ids docstring.
    all_entity_keys = {key for _, _, row in scored for key in (row["entity_ids"] or [])}
    entity_registry = {
        e["canonical_id"]: e
        for e in await db.get_entities_by_ids(list(all_entity_keys))
    }

    facts: list[MemoryPacketFact] = []
    entities_seen: dict[str, MemoryPacketEntity] = {}
    open_tasks: list[MemoryPacketTask] = []
    recent_decisions: list[MemoryPacketDecision] = []

    budget_chars = DEFAULT_TOKEN_BUDGET * _APPROX_CHARS_PER_TOKEN
    used_chars = 0

    for fact_id, score, row in scored:
        text = fact_texts.get(fact_id, "")
        entry_chars = len(text)
        if used_chars + entry_chars > budget_chars and facts:
            # Truncate per spec Layer 7 "Assembly" — most important first,
            # stop once the token budget is exhausted. Always keep at
            # least one fact even if it alone exceeds budget.
            break
        used_chars += entry_chars

        created_at = row["created_at"]
        date_str = created_at.date().isoformat() if created_at else None

        facts.append(
            MemoryPacketFact(
                text=text,
                type=row["type"],
                confidence=row["confidence"],
                source_session=row["session_id"],
                date=date_str,
                score=round(score, 4),
            )
        )

        for entity_key in row["entity_ids"] or []:
            resolved = entity_registry.get(entity_key)
            display_name = resolved["canonical_name"] if resolved else entity_key
            display_type = resolved["type"] if resolved else None
            if display_name not in entities_seen:
                entities_seen[display_name] = MemoryPacketEntity(
                    name=display_name, type=display_type, last_seen=date_str
                )
            elif date_str and (
                entities_seen[display_name].last_seen is None
                or date_str > entities_seen[display_name].last_seen
            ):
                entities_seen[display_name].last_seen = date_str

        if row["type"] == "task":
            open_tasks.append(MemoryPacketTask(text=text, status="open"))
        elif row["type"] == "decision":
            recent_decisions.append(MemoryPacketDecision(text=text, date=date_str))

    # Top-up open_tasks/recent_decisions directly from Postgres if the
    # semantic+graph search didn't surface any — see module docstring.
    if not open_tasks:
        rows = await db.get_recent_facts_by_type("task", limit=5)
        ids = [r["fact_id"] for r in rows]
        texts = await _fact_texts_from_qdrant(ids)
        for r in rows:
            text = texts.get(r["fact_id"])
            if text:
                open_tasks.append(MemoryPacketTask(text=text, status="open"))

    if not recent_decisions:
        rows = await db.get_recent_facts_by_type("decision", limit=5)
        ids = [r["fact_id"] for r in rows]
        texts = await _fact_texts_from_qdrant(ids)
        for r in rows:
            text = texts.get(r["fact_id"])
            if text:
                recent_decisions.append(
                    MemoryPacketDecision(
                        text=text,
                        date=r["created_at"].date().isoformat()
                        if r["created_at"]
                        else None,
                    )
                )

    return MemoryPacket(
        query=request.query,
        facts=facts,
        entities=list(entities_seen.values()),
        open_tasks=open_tasks,
        recent_decisions=recent_decisions,
    )
