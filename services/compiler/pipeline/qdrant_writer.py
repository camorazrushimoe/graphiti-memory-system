"""Step 7 (MVP) · Qdrant write — Layer 5 (Vector Index).

Embeds each written `MemoryItem`'s text (via the same LM Studio
nomic-embed-text-v1.5 endpoint used elsewhere) and upserts it into the
`facts` collection, alongside the metadata payload the spec calls for
(`entity_id`, `session_id`, `timestamp`, `type`, `status`).

Per spec (Layer 5): three collections are planned — `entities`, `episodes`,
`facts`. Only `facts` is wired for this MVP step, since that's the unit the
compiler main loop already produces one-per-`MemoryItem`; `entities` and
`episodes` collections are deferred to the Entity Resolver step (which is
the first place we have a stable canonical entity to embed) and to a
possible later episode-summary embedding pass.

The Qdrant client / embedding helper itself now lives in
`common/qdrant_client.py` (moved there in session 4) so the ingest
service's Retrieval endpoint (`/retrieve`) can reuse the exact same client
setup for semantic search reads instead of duplicating it.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import PointStruct

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common.qdrant_client import (  # noqa: E402
    FACTS_COLLECTION,
    close_qdrant,  # noqa: F401 (re-exported for main.py's shutdown hook)
    embed_text,
    ensure_collection,
    get_qdrant,
)
from schema.memory_event import MemoryItem  # noqa: E402

logger = logging.getLogger("compiler.qdrant_writer")


async def write_memory_item(
    item: MemoryItem,
    fact_id: str,
    session_id: str,
    source_agent: str,
    status: str = "active",
) -> None:
    """Embed a MemoryItem's text and upsert it into the `facts` collection.

    Raises on failure so the caller's existing retry path applies (same
    pattern as graphiti_writer.write_memory_item).
    """
    await ensure_collection(FACTS_COLLECTION)
    vector = embed_text(item.text)

    client = get_qdrant()
    await client.upsert(
        collection_name=FACTS_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, fact_id)),
                vector=vector,
                payload={
                    "fact_id": fact_id,
                    "text": item.text,
                    "entity_ids": [e.name for e in item.entities],
                    "session_id": session_id,
                    "source_agent": source_agent,
                    "type": item.type.value,
                    "status": status,
                    "confidence": item.confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    logger.info(
        "qdrant: upserted fact %s into '%s' collection (session=%s)",
        fact_id,
        FACTS_COLLECTION,
        session_id,
    )
