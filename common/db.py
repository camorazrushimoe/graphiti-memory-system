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
