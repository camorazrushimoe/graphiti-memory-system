"""Memory Compiler worker — Layer 3 entry point.

Event-driven worker loop: polls `compiler_jobs` in Postgres, loads the
session's raw archive, runs the pipeline over the *new* tail of turns
(sliding window, per spec: 4-6 turns), and persists results.

Window-size gating: jobs enqueued from a normal `/ingest/turn` wait until
at least MIN_WINDOW_SIZE new turns are buffered before running the (slow)
LLM steps. Jobs enqueued from `/ingest/close` carry `force_flush=true` and
are processed immediately regardless of window size, so a session that
ends with a short tail (< MIN_WINDOW_SIZE turns) is never left unprocessed.

Pipeline steps wired so far (per MVP order in the tech spec):
  1. Normalizer      (pure Python)
  2. Episode Splitter (DSPy)
  3. Episode Classifier (DSPy)
  4. Memory Extractor (Instructor, JSON schema)
  5. Entity Resolver  (embeddings + rapidfuzz + hard-rules dict)
  6. Contradiction Checker (rules + Gemma 4B for ambiguous cases)
  7. Memory Selector  (quality gate — confidence + acknowledgment rules)
  8. Graphiti write   (Layer 4 graph core — one episode per MemoryItem)
  9. Qdrant write     (Layer 5 vector index — one point per MemoryItem)

All MVP pipeline steps (1-11 in the tech spec) are now wired end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from common import archive, db  # noqa: E402
from common.graphiti_config import close_graphiti  # noqa: E402
from common.qdrant_client import close_qdrant  # noqa: E402
from schema.memory_event import EpisodeType  # noqa: E402
from services.compiler.pipeline import normalizer  # noqa: E402
from services.compiler.pipeline.classifier import classify_episode  # noqa: E402
from services.compiler.pipeline.contradiction import check_contradiction  # noqa: E402
from services.compiler.pipeline.extractor import extract_memory_items  # noqa: E402
from services.compiler.pipeline.graphiti_writer import (  # noqa: E402
    mark_superseded as mark_superseded_graphiti,
    write_memory_item as write_to_graphiti,
)
from services.compiler.pipeline.lm_config import configure_dspy  # noqa: E402
from services.compiler.pipeline.qdrant_writer import (  # noqa: E402
    fetch_fact_texts,
    mark_superseded as mark_superseded_qdrant,
    write_memory_item as write_to_qdrant,
)
from services.compiler.pipeline.resolver import resolve_entities  # noqa: E402
from services.compiler.pipeline.selector import select_memory_items  # noqa: E402
from services.compiler.pipeline.splitter import split_into_episodes  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compiler")

POLL_INTERVAL_SECONDS = 2.0
MIN_WINDOW_SIZE = 4  # spec: process once at least this many *new* turns are buffered


def _new_tail_turns(all_turns: list, last_processed_message_id: str | None) -> list:
    """Return only the turns after `last_processed_message_id` (or all turns
    if the session hasn't been processed yet).
    """
    if last_processed_message_id is None:
        return all_turns
    for i, t in enumerate(all_turns):
        if t.message_id == last_processed_message_id:
            return all_turns[i + 1 :]
    # Cursor not found (shouldn't normally happen) — reprocess everything
    # rather than silently dropping turns.
    return all_turns


async def process_job(job: dict) -> None:
    session_id = job["session_id"]
    job_id = job["job_id"]
    attempts = job["attempts"]
    force_flush = job.get("force_flush", False)

    session_doc = archive.load_session(session_id)
    if session_doc is None:
        raise RuntimeError(f"no raw archive found for session {session_id}")

    normalized = normalizer.normalize_session_dict(session_doc)

    last_processed = await db.get_last_processed_message_id(session_id)
    window_turns = _new_tail_turns(normalized.turns, last_processed)

    if not window_turns:
        # Nothing new since last processing — nothing to do (can happen if
        # /ingest/close arrives after a job already consumed the tail).
        await db.complete_job(job_id)
        return

    if len(window_turns) < MIN_WINDOW_SIZE and not force_flush:
        # Not enough new turns yet to justify running the (slow) LLM steps.
        # Leave the job as done — /ingest/close (force_flush=true) or a
        # future turn will re-enqueue and pick these turns up again.
        logger.info(
            "session %s: only %d new turn(s) buffered (< %d), deferring",
            session_id,
            len(window_turns),
            MIN_WINDOW_SIZE,
        )
        await db.complete_job(job_id)
        return

    if force_flush:
        logger.info(
            "session %s: force_flush (session close) — processing %d new turn(s)",
            session_id,
            len(window_turns),
        )
    else:
        logger.info(
            "session %s: processing window of %d new turn(s)",
            session_id,
            len(window_turns),
        )

    episodes = split_into_episodes(normalized, window_turns)
    logger.info("session %s: split into %d episode(s)", session_id, len(episodes))

    total_items = 0
    for episode in episodes:
        classified = classify_episode(episode)
        logger.info(
            "session %s: episode %s classified as %s (confidence=%.2f)",
            session_id,
            classified.episode_id,
            classified.episode_type.value if classified.episode_type else "?",
            classified.confidence or 0.0,
        )

        if classified.episode_type == EpisodeType.META:
            continue  # small-talk / acknowledgments — skip extraction entirely

        items = extract_memory_items(classified, classified.episode_type)
        logger.info(
            "session %s: episode %s extracted %d candidate item(s)",
            session_id,
            classified.episode_id,
            len(items),
        )

        # Step 7 — Memory Selector: final quality gate (confidence
        # thresholds + pure-acknowledgment discard, see
        # pipeline/selector.py). Runs before Entity Resolver / Contradiction
        # Checker / the writers below, so low-value items never reach
        # Graphiti/Qdrant/Postgres at all.
        #
        # Order note: Graphiti write happens BEFORE the Postgres insert. If
        # Graphiti fails and the job is retried, the retry re-runs the
        # extractor and gets a *new* fact_id — writing to Postgres first
        # would leave an orphan `facts` row (no matching graph episode) on
        # every failed attempt. Writing to Graphiti first means a failed
        # attempt leaves no trace in either store, and only a successful
        # attempt is reflected in Postgres too. This still does not fully
        # solve retry duplication if extraction *succeeds* but a *later*
        # item in the same episode causes the job to fail and retry
        # (earlier items in this loop are already committed to both
        # stores on retry #2) — full idempotency needs a proper dedup key,
        # deferred to a future session.
        selected_items = select_memory_items(items)
        logger.info(
            "session %s: episode %s selector kept %d/%d item(s)",
            session_id,
            classified.episode_id,
            len(selected_items),
            len(items),
        )
        for item in selected_items:
            # Step 5 — Entity Resolver: replace each raw entity surface
            # form with its canonical_id (registering new canonical
            # entities as needed). Runs before the contradiction check
            # (which keys off canonical_id, not raw surface forms) and
            # before the writers (Qdrant/Postgres now store canonical_id
            # in entity_ids instead of the raw extracted name).
            item = item.model_copy(
                update={"entities": await resolve_entities(item.entities)}
            )

            # Step 6 — Contradiction Checker: compare against existing
            # active facts sharing a canonical entity + type. On a
            # contradiction with score > 0.6 (spec threshold), the old
            # fact is marked outdated in all three stores and linked via
            # a SUPERSEDES edge; the new fact is still written normally
            # below regardless of the checker's verdict (per spec: "Both
            # versions are kept").
            existing_fact_ids = [
                e.canonical_id for e in item.entities if e.canonical_id
            ]
            fact_texts: dict[str, str] = {}
            if existing_fact_ids:
                candidate_rows = []
                for cid in existing_fact_ids:
                    candidate_rows.extend(
                        await db.get_active_facts_by_entity(cid, type_=item.type.value)
                    )
                candidate_ids = list({r["fact_id"] for r in candidate_rows})
                if candidate_ids:
                    fact_texts = await fetch_fact_texts(candidate_ids)

            contradiction = await check_contradiction(item, fact_texts)
            if contradiction.action in ("auto_update", "flag_for_review"):
                # Spec: >0.85 auto-update, 0.6-0.85 flag for review. MVP
                # applies the same outdated+supersedes treatment to both —
                # a real "review queue" surface for flag_for_review is not
                # implemented yet, tracked as follow-up work.
                old_fact_id = contradiction.contradicts_fact_id
                fact_id = f"fact_{uuid.uuid4().hex[:12]}"
                try:
                    await write_to_graphiti(
                        item=item,
                        fact_id=fact_id,
                        session_id=session_id,
                        source_agent=normalized.source_agent,
                    )
                    await mark_superseded_graphiti(old_fact_id, fact_id)
                except Exception:
                    logger.exception(
                        "session %s: graphiti write/supersede failed for fact %s",
                        session_id,
                        fact_id,
                    )
                    raise
                try:
                    await write_to_qdrant(
                        item=item,
                        fact_id=fact_id,
                        session_id=session_id,
                        source_agent=normalized.source_agent,
                    )
                    await mark_superseded_qdrant(old_fact_id)
                except Exception:
                    logger.exception(
                        "session %s: qdrant write/supersede failed for fact %s",
                        session_id,
                        fact_id,
                    )
                    raise
                await db.insert_fact(
                    fact_id=fact_id,
                    entity_ids=[e.canonical_id or e.name for e in item.entities],
                    type_=item.type.value,
                    confidence=item.confidence,
                    session_id=session_id,
                    source_agent=normalized.source_agent,
                )
                await db.mark_fact_outdated(old_fact_id, fact_id)
                logger.info(
                    "session %s: fact %s supersedes %s (contradiction score=%.2f, action=%s)",
                    session_id,
                    fact_id,
                    old_fact_id,
                    contradiction.contradiction_score,
                    contradiction.action,
                )
                total_items += 1
                continue

            fact_id = f"fact_{uuid.uuid4().hex[:12]}"
            try:
                await write_to_graphiti(
                    item=item,
                    fact_id=fact_id,
                    session_id=session_id,
                    source_agent=normalized.source_agent,
                )
            except Exception:
                logger.exception(
                    "session %s: graphiti write failed for fact %s",
                    session_id,
                    fact_id,
                )
                raise
            try:
                await write_to_qdrant(
                    item=item,
                    fact_id=fact_id,
                    session_id=session_id,
                    source_agent=normalized.source_agent,
                )
            except Exception:
                # Qdrant failure after a successful Graphiti write leaves the
                # graph and the vector index out of sync for this fact_id.
                # Surfacing the error (job retry) re-runs extraction and
                # produces a *new* fact_id, so the Graphiti episode from
                # this attempt becomes an orphan (no Qdrant/Postgres entry).
                # Same known limitation as the Graphiti-write ordering
                # comment above — proper idempotency needs a stable dedup
                # key, deferred to a future session.
                logger.exception(
                    "session %s: qdrant write failed for fact %s",
                    session_id,
                    fact_id,
                )
                raise
            await db.insert_fact(
                fact_id=fact_id,
                entity_ids=[e.canonical_id or e.name for e in item.entities],
                type_=item.type.value,
                confidence=item.confidence,
                session_id=session_id,
                source_agent=normalized.source_agent,
            )
            total_items += 1

    if episodes:
        await db.bump_episode_count(session_id, len(episodes))
        await db.set_last_processed_message_id(session_id, window_turns[-1].message_id)

    if force_flush:
        await db.mark_session_status(session_id, "done")

    logger.info(
        "session %s: job %d done — %d fact(s) persisted",
        session_id,
        job_id,
        total_items,
    )
    await db.complete_job(job_id)


async def worker_loop() -> None:
    configure_dspy()
    logger.info("compiler worker started, polling every %.1fs", POLL_INTERVAL_SECONDS)

    while True:
        job = await db.claim_next_job()
        if job is None:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        try:
            await process_job(job)
        except Exception as exc:  # noqa: BLE001 — worker must never crash the loop
            logger.exception("job %s failed", job["job_id"])
            await db.fail_job(job["job_id"], job["attempts"] + 1, str(exc))


async def main() -> None:
    try:
        await worker_loop()
    finally:
        await close_graphiti()
        await close_qdrant()
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
