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
  3b. Qdrant `episodes` write (Layer 5 — one point per episode, session 7)
  4. Memory Extractor (Instructor, JSON schema)
  5. Entity Resolver  (embeddings + rapidfuzz + hard-rules dict + curated
     alias groups, session 7)
  6. Contradiction Checker (rules + Gemma 4B for ambiguous cases) — splits
     into `auto_update` (score > 0.85, applied immediately) vs.
     `flag_for_review` (score 0.6-0.85, queued in `review_queue` instead of
     being auto-applied, session 7)
  7. Memory Selector  (quality gate — confidence + acknowledgment rules)
  8. Graphiti write   (Layer 4 graph core — one episode per MemoryItem,
     idempotent by deterministic fact_id, session 7)
  9. Qdrant write     (Layer 5 vector index — one point per MemoryItem)

All MVP pipeline steps (1-11 in the tech spec) are now wired end-to-end.
`fact_id` (see `_deterministic_fact_id` below) is a stable dedup key
derived from `(episode_id, item_index)` rather than a fresh random UUID
per attempt, closing the retry-duplication gap tracked since session 3.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from common import archive, db  # noqa: E402
from common.graphiti_config import close_graphiti  # noqa: E402
from common.qdrant_client import close_qdrant  # noqa: E402
from schema.memory_event import EpisodeType  # noqa: E402
from services.compiler.pipeline import normalizer  # noqa: E402
from services.compiler.pipeline.classifier import classify_episode  # noqa: E402
from services.compiler.pipeline.contradiction import check_contradiction  # noqa: E402
from services.compiler.pipeline.episode_writer import write_episode  # noqa: E402
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


def _deterministic_fact_id(episode_id: str, item_index: int) -> str:
    """Session 7 — stable dedup key for a `MemoryItem`, replacing the old
    `f"fact_{uuid.uuid4().hex[:12]}"` (a fresh random id on every attempt).

    Derived from `(episode_id, item_index)` — the episode a MemoryItem
    came from, plus its position in the Selector's output list for that
    episode — both of which are stable across retries of the *same* job
    (the extractor is deterministic enough in practice for this session's
    scope: same episode text -> same episode_id already, and the Selector
    only filters, never reorders, so index within the kept list is stable
    too). This means a retried job that re-reaches the same item recomputes
    the *same* fact_id, so:
      - `graphiti_writer.write_memory_item` skips re-adding the episode
        (see its docstring "Idempotency" note)
      - `qdrant_writer.write_memory_item` upserts the same point id (no-op
        duplicate)
      - `db.insert_fact` is `ON CONFLICT (fact_id) DO NOTHING` (no-op
        duplicate row)
    closing the retry-duplication gap called out in the tech-spec
    implementation log since session 3.

    Known residual limitation: if the extractor's output for the *same*
    episode text genuinely varies between attempts (LLM non-determinism —
    e.g. splits one sentence into two items on retry instead of one), the
    index-based key can drift and this degrades back to the old duplicate
    behavior for that item only. Not fully solved without content-hashing
    the item text itself, which was considered but rejected for now since
    it would also treat two textually-identical-but-legitimately-distinct
    memories (e.g. the same decision restated in two different episodes)
    as collisions — deterministic-per-attempt is judged good enough for
    the common case (transient network/LM Studio timeout mid-job).
    """
    digest = hashlib.sha256(f"{episode_id}:{item_index}".encode()).hexdigest()
    return f"fact_{digest[:24]}"


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

        # Layer 5 `episodes` collection (session 7) — embed and index this
        # episode's text regardless of what extraction below finds, so a
        # future retrieval pass can surface episodes by topic even when no
        # individual item made it past the Memory Selector. Non-fatal: a
        # failure here shouldn't block fact extraction/writes, which are
        # the pipeline's primary output.
        try:
            await write_episode(classified)
        except Exception:
            logger.exception(
                "session %s: episode %s qdrant episodes-collection write failed (non-fatal)",
                session_id,
                classified.episode_id,
            )

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
        # extractor and (per the docstring on `_deterministic_fact_id`,
        # session 7) recomputes the *same* fact_id for the same
        # `(episode_id, item_index)` — writing to Postgres first would
        # otherwise leave an orphan `facts` row (no matching graph episode)
        # on every failed attempt. Writing to Graphiti first means a failed
        # attempt leaves no trace in either store, and only a successful
        # attempt is reflected in Postgres too. Combined with the
        # idempotent dedup key, a retry that re-reaches an item already
        # committed on a prior attempt within the same job now safely
        # no-ops in all three stores instead of duplicating it (closes the
        # gap called out here since session 3 — see
        # `_deterministic_fact_id`'s docstring for the one residual edge
        # case that isn't fully covered: non-deterministic extractor
        # output across attempts).
        selected_items = select_memory_items(items)
        logger.info(
            "session %s: episode %s selector kept %d/%d item(s)",
            session_id,
            classified.episode_id,
            len(selected_items),
            len(items),
        )
        for item_index, item in enumerate(selected_items):
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
            if contradiction.action == "auto_update":
                # Spec: contradiction score > 0.85 -> automatic update. Old
                # fact is marked outdated in all three stores and linked
                # via a SUPERSEDES edge; the new fact is still written
                # normally below regardless (per spec: "Both versions are
                # kept").
                old_fact_id = contradiction.contradicts_fact_id
                fact_id = _deterministic_fact_id(classified.episode_id, item_index)
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

            if contradiction.action == "flag_for_review":
                # Session 7: spec's "0.6-0.85 flags for review" is now a
                # real review queue (schema/init.sql `review_queue` table)
                # instead of being treated identically to auto_update (see
                # tech-spec implementation log, session 6 known
                # limitation). The old fact is left `active`/unlinked; the
                # new fact is written normally below (per spec: "both
                # versions kept") and a pending review row records the
                # pair for a human (or future auto-resolution pass) to
                # adjudicate via `db.resolve_review`.
                old_fact_id = contradiction.contradicts_fact_id
                fact_id = _deterministic_fact_id(classified.episode_id, item_index)
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
                await db.insert_review_item(
                    new_fact_id=fact_id,
                    existing_fact_id=old_fact_id,
                    contradiction_score=contradiction.contradiction_score,
                )
                logger.info(
                    "session %s: fact %s flagged for review against %s (contradiction score=%.2f)",
                    session_id,
                    fact_id,
                    old_fact_id,
                    contradiction.contradiction_score,
                )
                total_items += 1
                continue

            fact_id = _deterministic_fact_id(classified.episode_id, item_index)
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
                # graph and the vector index out of sync for this fact_id
                # *until the retry*. Since `fact_id` is now deterministic
                # (session 7 — see `_deterministic_fact_id`), the retry
                # recomputes the same fact_id: the Graphiti write above
                # becomes a no-op (episode already exists) and this Qdrant
                # write is attempted again for real, closing the gap that
                # used to leave the Graphiti episode permanently orphaned.
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
