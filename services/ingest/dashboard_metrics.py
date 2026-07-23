"""Dashboard support — aggregates health + throughput signals from every
store in the pipeline into one `/metrics` JSON payload for `services/ingest`
to serve (and `dashboard.html`, in the same directory, to render).

Deliberately snapshot-only (no time-series storage) per the user's stated
scope: "what's the current state of the system", not "show me trends over
time". Every check below is best-effort — a failing check reports
`{"ok": False, "error": ...}` for its own section rather than raising, so
one dead dependency (e.g. LM Studio unreachable) doesn't take down the
whole dashboard response; the dashboard's job is precisely to surface that
kind of partial failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import httpx

from common import db
from common.graphiti_config import get_graphiti
from common.qdrant_client import (
    EPISODES_COLLECTION,
    ENTITIES_COLLECTION,
    FACTS_COLLECTION,
    get_qdrant,
)

logger = logging.getLogger("ingest.dashboard_metrics")

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://192.168.0.11:1234/v1")

# How stale the compiler's heartbeat can be before the dashboard flags it
# as down. Poll interval is 2s (services/compiler/main.py's
# POLL_INTERVAL_SECONDS) — 15s is ~7 missed ticks, generous enough to
# absorb a slow LLM call mid-tick (the heartbeat write happens once per
# loop iteration, *before* claiming a job, so a single long-running job
# does delay the next heartbeat write until that job finishes).
COMPILER_HEARTBEAT_STALE_SECONDS = 15
CHECK_TIMEOUT_SECONDS = 5.0

# How stale an agent's `last_seen` can be before the dashboard's "Connected
# Agents" section flags it offline. Client agents (e.g. codemie_code) are
# far less chatty than the compiler's ~2s poll loop — a turn only lands
# every few seconds to minutes depending on how fast the user/model is
# typing — so this is deliberately much more generous than
# COMPILER_HEARTBEAT_STALE_SECONDS. Not finally tuned (per
# agent-integration.md risk #5, "can decide along the way"); 15 minutes is
# a reasonable starting point for a single interactive session.
AGENT_ACTIVITY_STALE_SECONDS = 15 * 60


def _age_seconds(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds()


async def _check_postgres() -> dict:
    started = time.monotonic()
    try:
        metrics = await asyncio.wait_for(
            db.get_dashboard_metrics(), timeout=CHECK_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard: postgres check failed")
        return {"ok": False, "error": str(exc)}

    heartbeat_age = _age_seconds(metrics["compiler_last_heartbeat"])
    compiler_alive = (
        heartbeat_age is not None and heartbeat_age <= COMPILER_HEARTBEAT_STALE_SECONDS
    )

    return {
        "ok": True,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "sessions_total": metrics["sessions_total"],
        "sessions_24h": metrics["sessions_24h"],
        "facts_by_status": metrics["facts_by_status"],
        "facts_by_type": metrics["facts_by_type"],
        "entities_total": metrics["entities_total"],
        "jobs_by_status": metrics["jobs_by_status"],
        "oldest_queued_job_age_seconds": metrics["oldest_queued_job_age_seconds"],
        "jobs_errored_1h": metrics["jobs_errored_1h"],
        "last_fact_age_seconds": _age_seconds(metrics["last_fact_created_at"]),
        "review_pending": metrics["review_pending"],
        "contradictions_1h": metrics["contradictions_1h"],
        "auto_updates_1h": metrics["auto_updates_1h"],
        "compiler_heartbeat_age_seconds": heartbeat_age,
        "compiler_alive": compiler_alive,
    }


async def _check_neo4j() -> dict:
    started = time.monotonic()
    try:
        graphiti = get_graphiti()
        records, _, _ = await asyncio.wait_for(
            graphiti.driver.execute_query(
                """
                MATCH (e:Entity) WITH count(e) AS entities
                MATCH (ep:Episodic) WITH entities, count(ep) AS episodes
                MATCH ()-[r:RELATES_TO]->() WITH entities, episodes, count(r) AS relations
                MATCH ()-[s:SUPERSEDES]->()
                RETURN entities, episodes, relations, count(s) AS supersedes
                """
            ),
            timeout=CHECK_TIMEOUT_SECONDS,
        )
        row = records[0] if records else None
        return {
            "ok": True,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "entity_nodes": row["entities"] if row else 0,
            "episodic_nodes": row["episodes"] if row else 0,
            "relates_to_edges": row["relations"] if row else 0,
            "supersedes_edges": row["supersedes"] if row else 0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard: neo4j check failed")
        return {"ok": False, "error": str(exc)}


async def _check_qdrant() -> dict:
    started = time.monotonic()
    try:
        client = get_qdrant()
        collections = {}
        for name in (FACTS_COLLECTION, ENTITIES_COLLECTION, EPISODES_COLLECTION):
            # Check existence first, and let a connection-level failure
            # here propagate to the outer try/except (which marks the
            # whole Qdrant section down) — only an existence check that
            # succeeds and returns False (collection genuinely not created
            # yet, see common/qdrant_client.ensure_collection) should
            # report 0 points instead of failing the check. Without this
            # split, a dead Qdrant (connection refused/DNS failure) was
            # silently reported as "0 points in every collection" instead
            # of "unreachable" — indistinguishable from a healthy-but-empty
            # fresh install.
            exists = await asyncio.wait_for(
                client.collection_exists(name), timeout=CHECK_TIMEOUT_SECONDS
            )
            if not exists:
                collections[name] = 0
                continue
            info = await asyncio.wait_for(
                client.get_collection(name), timeout=CHECK_TIMEOUT_SECONDS
            )
            collections[name] = info.points_count
        return {
            "ok": True,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "collections": collections,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard: qdrant check failed")
        return {"ok": False, "error": str(exc)}


async def _check_lm_studio() -> dict:
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{LM_STUDIO_URL}/models")
            resp.raise_for_status()
            data = resp.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        return {
            "ok": True,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "models_loaded": model_ids,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("dashboard: LM Studio check failed: %s", exc)
        return {"ok": False, "error": str(exc)}


async def _check_agent_activity() -> dict:
    """ "Connected Agents" dashboard section (agent-integration.md). Lists
    every agent ever seen in `agent_activity`, online/offline derived from
    `last_seen` staleness — same shape as the compiler heartbeat check,
    just per-agent instead of a single named row.
    """
    started = time.monotonic()
    try:
        rows = await asyncio.wait_for(
            db.get_all_agent_activity(), timeout=CHECK_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard: agent_activity check failed")
        return {"ok": False, "error": str(exc)}

    agents = []
    for row in rows:
        age = _age_seconds(row["last_seen"])
        agents.append(
            {
                "agent_name": row["agent_name"],
                "online": age is not None and age <= AGENT_ACTIVITY_STALE_SECONDS,
                "last_seen_age_seconds": age,
                "last_action": row["last_action"],
                "turns_written_total": row["turns_written_total"],
                "retrieve_calls_total": row["retrieve_calls_total"],
                "read_tokens_estimate_total": row["read_tokens_estimate_total"],
            }
        )

    return {
        "ok": True,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "agents": agents,
    }


async def collect_metrics() -> dict:
    """Run all checks concurrently and assemble the `/metrics` payload.

    Each check is independent and already catches its own exceptions, so
    `asyncio.gather` here never needs `return_exceptions=True` — a slow or
    dead dependency degrades its own section to `{"ok": False, ...}`
    without blocking or failing the others.
    """
    postgres, neo4j, qdrant, lm_studio, agent_activity = await asyncio.gather(
        _check_postgres(),
        _check_neo4j(),
        _check_qdrant(),
        _check_lm_studio(),
        _check_agent_activity(),
    )

    services_ok = {
        "ingest": True,  # trivially true — this code is running inside it
        "postgres": postgres["ok"],
        "neo4j": neo4j["ok"],
        "qdrant": qdrant["ok"],
        "lm_studio": lm_studio["ok"],
        # compiler has no HTTP port of its own (see docker-compose.yml) —
        # liveness is inferred from the heartbeat row it writes into
        # Postgres, not a direct check, so it depends on postgres["ok"].
        "compiler": postgres.get("compiler_alive", False) if postgres["ok"] else False,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_healthy": all(services_ok.values()),
        "services": services_ok,
        "postgres": postgres,
        "neo4j": neo4j,
        "qdrant": qdrant,
        "lm_studio": lm_studio,
        "agent_activity": agent_activity,
    }
