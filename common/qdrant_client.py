"""Qdrant client + LM Studio embedding helper — Layer 5 (Vector Index).

Shared between the compiler's `qdrant_writer.py` (writes) and the ingest
service's `retrieval.py` (Layer 7 semantic search reads). Moved here from
`services/compiler/pipeline/qdrant_writer.py` (session 4) to avoid the
retrieval endpoint duplicating the same LM Studio embedding client / Qdrant
collection setup.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from openai import OpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger("common.qdrant")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://192.168.0.11:1234/v1")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
)

# nomic-embed-text-v1.5 via LM Studio returns 768-dim vectors (verified with
# a live call) — NOT the 384 the spec's table originally stated. Kept
# consistent with EMBEDDING_DIM in common/graphiti_config.py.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

FACTS_COLLECTION = "facts"
# Layer 5 per spec defines three collections (entities/episodes/facts).
# `entities` was deferred in session 3 ("no stable canonical entity to embed
# yet") — now used by the Entity Resolver (session 5,
# services/compiler/pipeline/resolver.py) as its similarity-search index:
# one point per canonical entity, payload {canonical_id, canonical_name,
# type}. `episodes` was deferred alongside it — now implemented in session
# 7 (services/compiler/pipeline/episode_writer.py): one point per compiler
# Episode (not per MemoryItem), payload {episode_id, session_id,
# topic_hint, episode_type, message_ids, timestamp}, embedding the
# episode's full text as its "summary" (no separate summarization pass
# exists yet — the episode text itself, per spec Step 2, already covers
# one coherent topic and is short enough to embed directly).
ENTITIES_COLLECTION = "entities"
EPISODES_COLLECTION = "episodes"

_qdrant: AsyncQdrantClient | None = None
_embed_client: OpenAI | None = None
_ensured_collections: set[str] = set()


def get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _embed_client


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(url=QDRANT_URL)
    return _qdrant


async def ensure_collection(name: str) -> None:
    if name in _ensured_collections:
        return
    client = get_qdrant()
    if not await client.collection_exists(name):
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("qdrant: created collection %s (dim=%d)", name, EMBEDDING_DIM)
    _ensured_collections.add(name)


def embed_text(text: str) -> list[float]:
    """Synchronous embedding call (OpenAI SDK's sync client) — this is the
    same LM Studio endpoint used by Instructor/DSPy elsewhere. Kept
    synchronous+blocking like the rest of the pipeline's LLM calls; callers
    running in an async context should wrap this with
    `asyncio.to_thread`/`run_in_executor` if it becomes a latency concern
    (not done yet — retrieval requests are infrequent and this mirrors the
    existing compiler pattern).
    """
    client = get_embed_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def point_id_for(key: str) -> str:
    """Deterministic Qdrant point UUID for a given natural key (fact_id or
    canonical entity_id) — lets writers upsert idempotently by re-deriving
    the same point id from the key instead of tracking a separate mapping.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


async def close_qdrant() -> None:
    global _qdrant
    if _qdrant is not None:
        await _qdrant.close()
        _qdrant = None
