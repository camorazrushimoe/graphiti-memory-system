"""PostgreSQL access layer — Layer 6 bookkeeping.

Shared between the ingest service (session/job registration) and the
compiler worker (job claiming, cursor tracking, fact persistence).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://memory:changeme@localhost:5432/memory_system"
)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=5)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def upsert_session(
    session_id: str, source_agent: str, status: str = "pending"
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (session_id, source_agent, ingested_at, status)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (session_id) DO UPDATE
                SET source_agent = EXCLUDED.source_agent,
                    ingested_at = EXCLUDED.ingested_at
            """,
            session_id,
            source_agent,
            datetime.now(timezone.utc),
            status,
        )


async def mark_session_status(session_id: str, status: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET status = $2 WHERE session_id = $1", session_id, status
        )


async def enqueue_compiler_job(session_id: str, force_flush: bool = False) -> None:
    """Enqueue a compiler job for this session if one isn't already queued/running.

    If `force_flush` is True (session close signal) and a job is already
    queued, upgrade that existing job to force_flush=true as well, so the
    compiler processes the buffered tail regardless of window size.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT job_id FROM compiler_jobs
            WHERE session_id = $1 AND status IN ('queued', 'running')
            """,
            session_id,
        )
        if existing is None:
            await conn.execute(
                "INSERT INTO compiler_jobs (session_id, status, force_flush) VALUES ($1, 'queued', $2)",
                session_id,
                force_flush,
            )
        elif force_flush:
            await conn.execute(
                "UPDATE compiler_jobs SET force_flush = true WHERE job_id = $1",
                existing["job_id"],
            )


async def get_session_status(session_id: str) -> Optional[str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM sessions WHERE session_id = $1", session_id
        )
        return row["status"] if row else None


async def get_last_processed_message_id(session_id: str) -> Optional[str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_processed_message_id FROM sessions WHERE session_id = $1",
            session_id,
        )
        return row["last_processed_message_id"] if row else None


async def set_last_processed_message_id(session_id: str, message_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET last_processed_message_id = $2 WHERE session_id = $1",
            session_id,
            message_id,
        )


async def bump_episode_count(session_id: str, delta: int) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET episode_count = episode_count + $2 WHERE session_id = $1",
            session_id,
            delta,
        )


# --- Compiler job queue: claim / complete / fail --------------------------


async def claim_next_job() -> Optional[dict]:
    """Atomically claim the oldest queued job (status queued -> running).

    Uses `FOR UPDATE SKIP LOCKED` so multiple compiler replicas could run
    concurrently without double-processing the same job.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT job_id, session_id, attempts, force_flush FROM compiler_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            if row is None:
                return None
            await conn.execute(
                """
                UPDATE compiler_jobs
                SET status = 'running', updated_at = now()
                WHERE job_id = $1
                """,
                row["job_id"],
            )
            return dict(row)


async def complete_job(job_id: int) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE compiler_jobs
            SET status = 'done', updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
        )


async def fail_job(job_id: int, attempts: int, error_message: str) -> None:
    """Mark a job as errored. Re-queues it (up to 3 attempts per spec),
    otherwise leaves it in 'error' status for manual inspection.
    """
    pool = await get_pool()
    next_status = "queued" if attempts < 3 else "error"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE compiler_jobs
            SET status = $2, attempts = $3, error_message = $4, updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            next_status,
            attempts,
            error_message[:2000],
        )


# --- Facts persistence (Memory Selector output) ----------------------------


async def insert_fact(
    fact_id: str,
    entity_ids: list[str],
    type_: str,
    confidence: float,
    session_id: str,
    source_agent: str,
    status: str = "active",
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO facts (fact_id, entity_ids, type, status, confidence, session_id, source_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (fact_id) DO NOTHING
            """,
            fact_id,
            entity_ids,
            type_,
            status,
            confidence,
            session_id,
            source_agent,
        )


# --- Entity registry (Step 5 — Entity Resolver) ----------------------------


async def get_entity(canonical_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM entities WHERE canonical_id = $1", canonical_id
        )
        return dict(row) if row else None


async def find_entity_by_alias(surface_form: str) -> Optional[dict]:
    """Hard-rule lookup: exact case-insensitive match against `canonical_name`
    or any known `aliases` entry. First tier of the resolver strategy
    (spec Step 5: "embeddings similarity -> fuzzy string match -> hard
    rules dict") — cheapest and most precise, tried before the fuzzier
    strategies.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM entities
            WHERE lower(canonical_name) = lower($1)
               OR EXISTS (
                   SELECT 1 FROM unnest(aliases) a WHERE lower(a) = lower($1)
               )
            LIMIT 1
            """,
            surface_form,
        )
        return dict(row) if row else None


async def get_entities_by_ids(canonical_ids: list[str]) -> list[dict]:
    """Batch-fetch canonical entity metadata (name/type) for a set of
    `canonical_id`s — used by the Retrieval endpoint (Layer 7) to resolve
    the `entity_ids` stored on `facts` rows into displayable
    `{name, type}` pairs, per the spec's `memory_packet.entities` shape.
    Post-Entity-Resolver (session 5), `facts.entity_ids` stores
    `canonical_id`s rather than raw surface-form strings, so a lookup is
    required to show a human-readable name.
    """
    if not canonical_ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT canonical_id, canonical_name, type, aliases FROM entities WHERE canonical_id = ANY($1::text[])",
            canonical_ids,
        )
        return [dict(r) for r in rows]


async def get_all_entities() -> list[dict]:
    """Full registry snapshot — used for the fuzzy-match tier (rapidfuzz),
    which needs the candidate pool of canonical names/aliases in memory.
    Registry is expected to stay small enough for this (thousands, not
    millions, of entities) for a single-user/small-team local deployment;
    revisit if it ever needs to scale beyond that.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM entities")
        return [dict(r) for r in rows]


async def insert_entity(
    canonical_id: str,
    canonical_name: str,
    type_: Optional[str],
    aliases: list[str],
) -> None:
    pool = await get_pool()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO entities (canonical_id, canonical_name, type, aliases, first_seen, last_seen)
            VALUES ($1, $2, $3, $4, $5, $5)
            ON CONFLICT (canonical_id) DO NOTHING
            """,
            canonical_id,
            canonical_name,
            type_,
            aliases,
            now,
        )


async def add_entity_alias(canonical_id: str, alias: str) -> None:
    """Append a new surface form to an existing entity's `aliases` array
    (no-op if already present) and bump `last_seen`.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE entities
            SET aliases = CASE
                    WHEN $2 = ANY(aliases) THEN aliases
                    ELSE array_append(aliases, $2)
                END,
                last_seen = $3
            WHERE canonical_id = $1
            """,
            canonical_id,
            alias,
            datetime.now(timezone.utc),
        )


# --- Fact lifecycle (Step 6 — Contradiction Checker) ------------------------


async def get_active_facts_by_entity(
    canonical_id: str,
    type_: Optional[str] = None,
    exclude_fact_id: Optional[str] = None,
) -> list[dict]:
    """Fetch active facts that mention a given canonical entity — the
    Contradiction Checker's candidate pool (spec Step 6: "existing facts
    from Graphiti (same entity scope)"; we use Postgres `entity_ids` here
    instead since it's already indexed and authoritative for `status`,
    and avoids a second round-trip to Graphiti/Neo4j for this check).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM facts
            WHERE status = 'active'
              AND $1 = ANY(entity_ids)
              AND ($2::text IS NULL OR type = $2)
              AND ($3::text IS NULL OR fact_id != $3)
            ORDER BY created_at DESC
            """,
            canonical_id,
            type_,
            exclude_fact_id,
        )
        return [dict(r) for r in rows]


async def mark_fact_outdated(fact_id: str, superseded_by: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE facts SET status = 'outdated', superseded_by = $2
            WHERE fact_id = $1
            """,
            fact_id,
            superseded_by,
        )


# --- Layer 7 (Retrieval Service) reads --------------------------------------


async def get_facts_by_ids(fact_ids: list[str]) -> list[dict]:
    """Batch-fetch canonical fact metadata (Postgres is the authoritative
    source per Layer 6) for a set of `fact_id`s discovered by the Retrieval
    endpoint's semantic search (Qdrant) and/or graph traversal (Graphiti).

    Returns rows even for `status = outdated`/`discarded` facts — the caller
    (services/ingest/retrieval.py) decides whether to filter those out, per
    spec Layer 7 strategy #3 ("Exclude status = outdated unless query
    explicitly asks for history").
    """
    if not fact_ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT fact_id, entity_ids, type, status, confidence,
                   session_id, source_agent, created_at, superseded_by
            FROM facts
            WHERE fact_id = ANY($1::text[])
            """,
            fact_ids,
        )
        return [dict(r) for r in rows]


async def get_recent_facts_by_type(type_: str, limit: int = 5) -> list[dict]:
    """Fetch the most recent active facts of a given type — used by the
    Retrieval endpoint's `open_tasks` (type='task') and `recent_decisions`
    (type='decision') memory_packet sections (Layer 7).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT fact_id, entity_ids, type, status, confidence,
                   session_id, source_agent, created_at, superseded_by
            FROM facts
            WHERE type = $1 AND status = 'active'
            ORDER BY created_at DESC
            LIMIT $2
            """,
            type_,
            limit,
        )
        return [dict(r) for r in rows]


# --- Review queue (Step 6 — Contradiction Checker "flag_for_review") -------


async def insert_review_item(
    new_fact_id: str, existing_fact_id: str, contradiction_score: float
) -> None:
    """Record a `flag_for_review` contradiction (score 0.6-0.85, per spec)
    instead of auto-applying the outdated+supersedes treatment. See
    `schema/init.sql`'s `review_queue` table docstring for the rationale.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO review_queue (new_fact_id, existing_fact_id, contradiction_score, status)
            VALUES ($1, $2, $3, 'pending')
            """,
            new_fact_id,
            existing_fact_id,
            contradiction_score,
        )


async def get_pending_reviews(limit: int = 50) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM review_queue WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]


async def resolve_review(review_id: int, status: str) -> Optional[dict]:
    """Mark a review row `approved` or `rejected`. Returns the row (before
    the caller decides whether to also apply outdated+supersedes for an
    'approved' verdict) or None if `review_id` doesn't exist.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE review_queue
            SET status = $2, resolved_at = now()
            WHERE review_id = $1 AND status = 'pending'
            RETURNING *
            """,
            review_id,
            status,
        )
        return dict(row) if row else None


# --- Worker heartbeat (dashboard support) -----------------------------------


async def touch_worker_heartbeat(worker_name: str) -> None:
    """Upsert `last_seen = now()` for a named worker. Called by the
    compiler's `worker_loop()` on every poll tick (see `schema/init.sql`'s
    `worker_heartbeat` table docstring).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO worker_heartbeat (worker_name, last_seen)
            VALUES ($1, now())
            ON CONFLICT (worker_name) DO UPDATE SET last_seen = now()
            """,
            worker_name,
        )


async def get_worker_heartbeat(worker_name: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT worker_name, last_seen FROM worker_heartbeat WHERE worker_name = $1",
            worker_name,
        )
        return dict(row) if row else None


# --- Dashboard metrics -------------------------------------------------------


async def get_dashboard_metrics() -> dict:
    """Single round-trip (a handful of cheap aggregate queries in one
    connection) gathering everything the dashboard's `/metrics` endpoint
    needs from Postgres. Kept as one function (rather than many small ones
    called separately) so the dashboard's Postgres section is always a
    consistent snapshot from roughly the same instant, and so adding a new
    metric later means touching one query list instead of hunting for
    where to slot in a new round-trip.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        sessions_total = await conn.fetchval("SELECT count(*) FROM sessions")
        sessions_24h = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE ingested_at > now() - interval '24 hours'"
        )
        facts_by_status = await conn.fetch(
            "SELECT status, count(*) AS n FROM facts GROUP BY status"
        )
        facts_by_type = await conn.fetch(
            "SELECT type, count(*) AS n FROM facts WHERE status = 'active' GROUP BY type"
        )
        entities_total = await conn.fetchval("SELECT count(*) FROM entities")
        jobs_by_status = await conn.fetch(
            "SELECT status, count(*) AS n FROM compiler_jobs GROUP BY status"
        )
        oldest_queued_age_seconds = await conn.fetchval(
            """
            SELECT EXTRACT(EPOCH FROM (now() - min(created_at)))
            FROM compiler_jobs WHERE status = 'queued'
            """
        )
        jobs_errored_1h = await conn.fetchval(
            """
            SELECT count(*) FROM compiler_jobs
            WHERE status = 'error' AND updated_at > now() - interval '1 hour'
            """
        )
        last_fact_created_at = await conn.fetchval("SELECT max(created_at) FROM facts")
        review_pending = await conn.fetchval(
            "SELECT count(*) FROM review_queue WHERE status = 'pending'"
        )
        contradictions_1h = await conn.fetchval(
            "SELECT count(*) FROM review_queue WHERE created_at > now() - interval '1 hour'"
        )
        auto_updates_1h = await conn.fetchval(
            """
            SELECT count(*) FROM facts
            WHERE status = 'outdated' AND created_at > now() - interval '1 hour'
                AND superseded_by IS NOT NULL
            """
        )
        compiler_heartbeat = await conn.fetchrow(
            "SELECT last_seen FROM worker_heartbeat WHERE worker_name = 'compiler'"
        )

    return {
        "sessions_total": sessions_total,
        "sessions_24h": sessions_24h,
        "facts_by_status": {r["status"]: r["n"] for r in facts_by_status},
        "facts_by_type": {r["type"]: r["n"] for r in facts_by_type},
        "entities_total": entities_total,
        "jobs_by_status": {r["status"]: r["n"] for r in jobs_by_status},
        "oldest_queued_job_age_seconds": oldest_queued_age_seconds,
        "jobs_errored_1h": jobs_errored_1h,
        "last_fact_created_at": last_fact_created_at,
        "review_pending": review_pending,
        "contradictions_1h": contradictions_1h,
        "auto_updates_1h": auto_updates_1h,
        "compiler_last_heartbeat": compiler_heartbeat["last_seen"]
        if compiler_heartbeat
        else None,
    }
