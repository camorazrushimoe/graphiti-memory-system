"""Step 6 (MVP) · Graphiti write — Layer 4 (Graph Core).

Sends selected `MemoryItem` objects to Graphiti as episodes. Graphiti (via
its own LLM-driven extraction pass over the episode text) creates/updates
entity nodes and relationship edges in Neo4j, and embeds them using the
configured embedder (LM Studio / nomic-embed-text-v1.5).

Design notes
------------
- We do NOT pass a custom `group_id` to `add_episode`: graphiti-core's Neo4j
  driver treats a non-default `group_id` as a *separate database name*
  (see `Graphiti.add_episode` -> `self.driver.clone(database=group_id)`).
  The spec requires memory to be shared across all agents by default, so
  everything stays on the default group/database; `source_agent` is instead
  recorded in `source_description` for provenance (see the spec's open
  question: "it would be good if in memory we will have agent name").
- One episode per `MemoryItem` (not one per compiler `Episode`): this keeps
  each Graphiti episode small and focused, which the underlying extraction
  prompt handles more reliably than a whole multi-item episode blob, and
  keeps a clean 1:1 mapping to our own `fact_id` for later cross-referencing
  (Entity Resolver / Contradiction Checker will need this).
- Non-blocking per spec (Layer 3 "Real-time processing model"): the caller
  (compiler main loop) awaits this, but the ingest HTTP endpoint itself is
  never blocked — only the async worker loop is, which is by design.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core.nodes import EpisodeType as GraphitiEpisodeType

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common.graphiti_config import get_graphiti  # noqa: E402
from schema.memory_event import MemoryItem  # noqa: E402

logger = logging.getLogger("compiler.graphiti_writer")


async def write_memory_item(
    item: MemoryItem,
    fact_id: str,
    session_id: str,
    source_agent: str,
) -> None:
    """Write a single MemoryItem to Graphiti as one episode.

    Raises on failure — the caller decides whether a Graphiti write error
    should fail the whole compiler job (current behavior: yes, so retries
    per spec's "failed episodes are retried up to 3 times" apply here too).
    """
    graphiti = get_graphiti()
    await graphiti.add_episode(
        name=fact_id,
        episode_body=item.text,
        source_description=f"{source_agent} session {session_id}",
        reference_time=datetime.now(timezone.utc),
        source=GraphitiEpisodeType.text,
        group_id=None,
    )
    logger.info(
        "graphiti: wrote episode for fact %s (session=%s, type=%s)",
        fact_id,
        session_id,
        item.type.value,
    )


async def mark_superseded(old_fact_id: str, new_fact_id: str) -> None:
    """Step 6 (Contradiction Checker) support — create the
    `(MemoryItem)-[:SUPERSEDES]->(MemoryItem)` link the spec's graph model
    calls for (Layer 4 "Graph model"), between the two Graphiti Episodic
    nodes named after each fact_id (see `write_memory_item` above — the
    episode `name` is always the fact_id).

    Both episodes are kept (per spec: "Both versions are kept, the temporal
    graph supports this natively") — this only adds the supersedes edge,
    it does not delete or modify the old episode's content.
    """
    graphiti = get_graphiti()
    await graphiti.driver.execute_query(
        """
        MATCH (old:Episodic {name: $old_name}), (new:Episodic {name: $new_name})
        MERGE (new)-[:SUPERSEDES]->(old)
        """,
        old_name=old_fact_id,
        new_name=new_fact_id,
    )
    logger.info(
        "graphiti: marked fact %s as superseded by %s (SUPERSEDES edge)",
        old_fact_id,
        new_fact_id,
    )
