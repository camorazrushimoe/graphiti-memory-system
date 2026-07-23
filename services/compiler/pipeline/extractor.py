"""Step 4 · Memory Extractor — LLM + Instructor (Layer 3).

Extracts entities, relations, claims, tasks, and open questions from a
classified Episode. Enforces the MemoryItem JSON schema via Instructor.

Rule (per spec): extract only what is explicitly stated — never infer
unless marked `inferred: true` (i.e. `explicit: False`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from schema.memory_event import Entity, Episode, EpisodeType, MemoryItem, Relation  # noqa: E402
from services.compiler.pipeline.instructor_config import (
    LLM_MODEL,
    get_instructor_client,
)  # noqa: E402


class ExtractedItem(BaseModel):
    """Instructor response model for a single extracted memory item.

    Mirrors `schema.memory_event.MemoryItem` but keeps `type` implicit
    (the caller already knows the episode's classified type) and omits
    `source_message_ids`, which we fill in ourselves from the episode.

    Field-level `description`s matter a lot here: Gemma 4B (via Instructor/
    JSON-schema mode) follows per-field schema descriptions far more
    reliably than prose instructions in the prompt, so entities/relations
    guidance lives on the fields themselves, not just in the system prompt.
    """

    text: str = Field(
        description="The extracted fact/decision/task/etc, stated concisely."
    )
    entities: list[Entity] = Field(
        description=(
            "Every named tool, person, project, file, or concept mentioned "
            "in this item's text, each with a short `type` label. Return "
            "an empty list only if the text truly contains none."
        ),
    )
    relations: list[Relation] = Field(
        description=(
            "Subject-predicate-object triples describing how the entities "
            "above relate to each other or to the action described. Return "
            "an empty list only if the text truly contains none."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    explicit: bool = Field(
        default=True,
        description="True if stated directly in the text, False if inferred rather than stated.",
    )


class ExtractionResult(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


_EXTRACTION_INSTRUCTIONS = (
    "Extract memory items from the episode text below. The episode has "
    "already been classified as type '{episode_type}'. Extract ONLY what "
    "is explicitly stated in the text — do not add outside knowledge or "
    "speculation. If you must infer something not directly stated, mark "
    "that item's `explicit` field as false. Assign a confidence score "
    "(0.0-1.0) to each item reflecting how clearly and unambiguously it "
    "was stated.\n\n"
    "For EVERY item, also populate its `entities` and `relations` fields:\n"
    "- `entities`: every named tool, person, project, file, or concept "
    "mentioned in that item's text, with a short `type` (e.g. tool, "
    "person, project, file, concept).\n"
    "- `relations`: subject-predicate-object triples describing how those "
    "entities relate to each other or to the action described (e.g. "
    "subject='Graphiti', predicate='used_as', object='memory layer').\n"
    "Leave `entities`/`relations` empty only if the item text truly "
    "contains no identifiable entities or relations.\n\n"
    "It is fine to return zero items if nothing memory-worthy is present."
)


def extract_memory_items(
    episode: Episode, episode_type: Optional[EpisodeType] = None
) -> list[MemoryItem]:
    """Run the Memory Extractor over a single episode.

    Returns a list of `MemoryItem` objects, each linked back to the
    episode's source `message_id`s for provenance.
    """
    etype = episode_type or episode.episode_type or EpisodeType.FACT
    client = get_instructor_client()

    prompt = (
        _EXTRACTION_INSTRUCTIONS.format(episode_type=etype.value)
        + "\n\nEPISODE TEXT:\n"
        + episode.text
    )

    try:
        result = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=ExtractionResult,
            max_retries=2,
        )
    except Exception:
        return []

    memory_items = []
    for item in result.items:
        memory_items.append(
            MemoryItem(
                text=item.text,
                type=etype,
                entities=item.entities,
                relations=item.relations,
                confidence=item.confidence,
                source_message_ids=list(episode.message_ids),
                explicit=item.explicit,
            )
        )
    return memory_items
