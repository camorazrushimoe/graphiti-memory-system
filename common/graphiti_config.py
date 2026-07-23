"""Graphiti client configuration — points at Neo4j + the local LM Studio
OpenAI-compatible endpoint for both the LLM (extraction/summaries) and the
embedder (node/edge embeddings).

Shared, lazily-constructed singleton so a process only opens one Neo4j
driver connection for its lifetime. Moved here from
`services/compiler/pipeline/graphiti_config.py` (session 4) so the
`ingest` service's Retrieval endpoint (`/retrieve`, Layer 7 — graph
traversal via `graphiti.search()`) can reuse the exact same client
configuration as the compiler's Graphiti writer, instead of duplicating
the LM Studio / cross-encoder workarounds in two places.
"""

from __future__ import annotations

import os
from typing import Any
from pydantic import BaseModel

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://192.168.0.11:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-e4b-it")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
)

# nomic-embed-text-v1.5 (via LM Studio) returns 768-dim vectors — verified
# with a live call. graphiti-core's EmbedderConfig defaults to 1024 (OpenAI's
# text-embedding-3-small dimension), which is wrong for our model and would
# silently truncate/break similarity search, so it must be set explicitly.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "changeme")

_graphiti: Graphiti | None = None


class LMStudioOpenAIGenericClient(OpenAIGenericClient):
    """Custom client that overrides _build_response_format to be fully compatible with LM Studio.

    If response_model is None, we return {"type": "text"} instead of {"type": "json_object"},
    as LM Studio's llama.cpp backend rejects json_object requests with 400.
    """

    def _build_response_format(
        self, response_model: type[BaseModel] | None
    ) -> dict[str, Any]:
        if response_model is None:
            return {"type": "text"}
        return super()._build_response_format(response_model)


def get_graphiti() -> Graphiti:
    """Lazily create and cache the Graphiti client.

    Both the LLM client and the embedder point at the same LM Studio
    OpenAI-compatible endpoint used elsewhere in the pipeline.
    """
    global _graphiti
    if _graphiti is None:
        llm_client = LMStudioOpenAIGenericClient(
            config=LLMConfig(
                api_key="lm-studio",  # LM Studio ignores the key, field is required
                model=LLM_MODEL,
                small_model=LLM_MODEL,  # single-model pipeline, per spec
                base_url=LM_STUDIO_URL,
            ),
            # LM Studio's OpenAI-compatible server (llama.cpp backend) only
            # accepts response_format.type of 'json_schema' or 'text' — a
            # bare 'json_object' request is rejected with a 400 (verified
            # live: "'response_format.type' must be 'json_schema' or
            # 'text'"). Use native json_schema, matching the mode already
            # used for Instructor (see instructor_config.py).
            structured_output_mode="json_schema",
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="lm-studio",
                embedding_model=EMBEDDING_MODEL,
                base_url=LM_STUDIO_URL,
                embedding_dim=EMBEDDING_DIM,
            )
        )
        # graphiti-core defaults to OpenAIRerankerClient(), which builds an
        # AsyncOpenAI() with NO base_url/api_key override and therefore
        # requires OPENAI_API_KEY — it reaches out to api.openai.com by
        # default, not LM Studio. We're single-LLM-only per spec, so point
        # the reranker at the same local endpoint explicitly. This matters
        # for `/retrieve` too: Graphiti's hybrid search uses the
        # cross-encoder to rerank edges unless a different reranker config
        # is passed.
        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                api_key="lm-studio",
                model=LLM_MODEL,
                base_url=LM_STUDIO_URL,
            )
        )
        _graphiti = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
    return _graphiti


async def close_graphiti() -> None:
    global _graphiti
    if _graphiti is not None:
        await _graphiti.close()
        _graphiti = None
