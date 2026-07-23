"""Ingest service — Layer 1 (hook endpoints) + writes to Layer 2 (Raw Archive)
and enqueues Layer 3 (Memory Compiler) jobs via Postgres. Also hosts Layer 7
(Retrieval Service), per the spec's services table ("ingest: Ingest endpoint
+ retrieval API (FastAPI)").

Endpoints:
  POST /ingest/turn   - streaming, one turn at a time (real-time ingestion)
  POST /ingest/close  - session close signal, flush remaining buffer
  POST /ingest        - full session dump (fallback / replay)
  POST /retrieve      - Layer 7: semantic + graph + temporal memory_packet
  GET  /healthz       - liveness check
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root, for `schema` package

from schema.retrieval import MemoryPacket, RetrieveRequest  # noqa: E402
from schema.turn import (  # noqa: E402
    IngestCloseRequest,
    IngestSessionRequest,
    IngestTurnRequest,
)

from common import archive, db  # noqa: E402
from common.graphiti_config import close_graphiti  # noqa: E402
from common.qdrant_client import close_qdrant  # noqa: E402
from services.ingest import retrieval  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

app = FastAPI(title="Graphiti Memory — Ingest Service")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/ingest/turn")
async def ingest_turn(payload: IngestTurnRequest):
    """Streaming ingestion — one POST per turn, written immediately to the
    raw archive and the session is (re)registered / enqueued for compiling.
    """
    turn_dict = payload.turn.model_dump(mode="json")

    session_doc = archive.append_turn(
        session_id=payload.session_id,
        source_agent=payload.source_agent,
        model=payload.model,
        turn=turn_dict,
    )

    await db.upsert_session(
        payload.session_id, payload.source_agent, status="processing"
    )
    await db.enqueue_compiler_job(payload.session_id)

    logger.info(
        "ingested turn %s for session %s (total turns=%d)",
        payload.turn.message_id,
        payload.session_id,
        len(session_doc["turns"]),
    )
    return {
        "status": "ok",
        "session_id": payload.session_id,
        "turn_count": len(session_doc["turns"]),
    }


@app.post("/ingest/close")
async def ingest_close(payload: IngestCloseRequest):
    """Session close signal — flush remaining buffer. Marks the session
    ready for a final compiler pass over any un-processed tail turns.
    """
    status = await db.get_session_status(payload.session_id)
    if status is None:
        raise HTTPException(
            status_code=404, detail=f"unknown session_id: {payload.session_id}"
        )

    await db.enqueue_compiler_job(payload.session_id, force_flush=True)
    await db.mark_session_status(payload.session_id, "pending_close")

    logger.info("session close requested for %s", payload.session_id)
    return {"status": "ok", "session_id": payload.session_id}


@app.post("/ingest")
async def ingest_full_session(payload: IngestSessionRequest):
    """Full session dump — fallback / replay path. Overwrites/creates the
    raw archive file for this session_id with the complete turn list.
    """
    turns = [t.model_dump(mode="json") for t in payload.turns]

    session_doc = archive.write_full_session(
        session_id=payload.session_id,
        source_agent=payload.source_agent,
        model=payload.model,
        turns=turns,
    )

    await db.upsert_session(payload.session_id, payload.source_agent, status="pending")
    await db.enqueue_compiler_job(payload.session_id)

    logger.info(
        "ingested full session %s (%d turns)",
        payload.session_id,
        len(session_doc["turns"]),
    )
    return {
        "status": "ok",
        "session_id": payload.session_id,
        "turn_count": len(session_doc["turns"]),
    }


@app.post("/retrieve", response_model=MemoryPacket)
async def retrieve_memory(payload: RetrieveRequest):
    """Layer 7 — answers "what memory is relevant to this new session?".

    Runs semantic search (Qdrant) + graph traversal (Graphiti/Neo4j) in
    parallel-ish (sequential awaits, see retrieval.py — both are I/O bound
    calls to local services so this is not a latency concern at this
    scale), merges by fact_id, ranks by combined score, and returns a
    memory_packet per the spec's Layer 7 format.
    """
    packet = await retrieval.retrieve(payload)
    logger.info(
        "retrieve: query=%r -> %d fact(s), %d entity(ies), %d task(s), %d decision(s)",
        payload.query,
        len(packet.facts),
        len(packet.entities),
        len(packet.open_tasks),
        len(packet.recent_decisions),
    )
    return packet


@app.on_event("shutdown")
async def shutdown():
    await db.close_pool()
    await close_graphiti()
    await close_qdrant()
