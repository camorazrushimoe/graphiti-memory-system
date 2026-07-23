"""Step 7 (MVP, session 7 addition) · Qdrant `episodes` collection write —
Layer 5 (Vector Index).

The tech spec's Layer 5 defines three Qdrant collections: `entities`,
`episodes`, `facts`. Only `facts` (session 3) and `entities` (session 5,
via the Entity Resolver) were wired before this module — `episodes` was
deferred each time (see tech-spec implementation log, sessions 3/6 "known
limitation": "`episodes` Qdrant-collection — still not implemented").

This module embeds each compiler `Episode`'s full text (no separate
summarization pass — the Episode Splitter (Step 2) already groups turns
into one coherent topic per spec, and that raw grouped text is what's
available and short enough to embed directly) and upserts it into the
`episodes` collection, one point per Episode.

Written once per Episode, right after classification (see
`services/compiler/main.py`) — independent of whether the episode ends up
producing zero, one, or many `MemoryItem`s, since the point is to let a
future retrieval pass find *episodes* by topic even when no individual
fact/item scored high enough to be extracted (e.g. skipped `meta`
episodes, or an episode whose items were all discarded by the Memory
Selector). `services/ingest/retrieval.py` does not yet query this
collection (deferred — see tech-spec implementation log's follow-up list),
this module only covers the write side.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import PointStruct

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common.qdrant_client import (  # noqa: E402
    EPISODES_COLLECTION,
    embed_text,
    ensure_collection,
    get_qdrant,
    point_id_for,
)
from schema.memory_event import Episode  # noqa: E402

logger = logging.getLogger("compiler.episode_writer")


async def write_episode(episode: Episode) -> None:
    """Embed an Episode's full text and upsert it into the `episodes`
    collection. Idempotent by construction: `point_id_for(episode.episode_id)`
    is deterministic, so re-processing the same episode (e.g. a job retry)
    simply overwrites the same point rather than creating a duplicate.
    """
    await ensure_collection(EPISODES_COLLECTION)
    vector = embed_text(episode.text)

    client = get_qdrant()
    await client.upsert(
        collection_name=EPISODES_COLLECTION,
        points=[
            PointStruct(
                id=point_id_for(episode.episode_id),
                vector=vector,
                payload={
                    "episode_id": episode.episode_id,
                    "session_id": episode.session_id,
                    "source_agent": episode.source_agent,
                    "topic_hint": episode.topic_hint,
                    "episode_type": episode.episode_type.value
                    if episode.episode_type
                    else None,
                    "confidence": episode.confidence,
                    "message_ids": episode.message_ids,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    logger.info(
        "qdrant: upserted episode %s into '%s' collection (session=%s, type=%s)",
        episode.episode_id,
        EPISODES_COLLECTION,
        episode.session_id,
        episode.episode_type.value if episode.episode_type else "?",
    )
