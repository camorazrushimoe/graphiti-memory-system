"""Shared memory schemas — Episode, MemoryItem, MemoryEvent, and pipeline
intermediate types used across the compiler pipeline steps."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EpisodeType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    TASK = "task"
    PREFERENCE = "preference"
    QUESTION = "question"
    IDEA = "idea"
    CONSTRAINT = "constraint"
    ENTITY_UPDATE = "entity_update"
    META = "meta"


class EpisodeBoundary(BaseModel):
    """Output of the Episode Splitter — marks where one episode starts/ends."""

    start_message_id: str
    end_message_id: str
    topic_hint: str


class Episode(BaseModel):
    """Grouped turns representing one coherent topic or task."""

    episode_id: str
    session_id: str
    source_agent: str
    topic_hint: str
    message_ids: list[str]
    text: str  # concatenated turn content for this episode
    episode_type: Optional[EpisodeType] = None
    confidence: Optional[float] = None


class Entity(BaseModel):
    name: str
    canonical_id: Optional[str] = None
    type: Optional[str] = None


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str


class MemoryItem(BaseModel):
    """Output of the Memory Extractor — one candidate memory item."""

    text: str
    type: EpisodeType
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_message_ids: list[str] = Field(default_factory=list)
    explicit: bool = True  # True = stated directly, False = inferred


class ContradictionResult(BaseModel):
    """Output of the Contradiction Checker."""

    contradicts_fact_id: Optional[str] = None
    contradiction_score: float = 0.0
    action: str = "none"  # none | auto_update | flag_for_review


class MemoryEvent(BaseModel):
    """Final structured event written to Graphiti + Qdrant + Postgres."""

    type: EpisodeType
    session_id: str
    turn_ids: list[str]
    text: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_agent: str
    explicit: bool = True
    status: str = "active"  # active | outdated | discarded
    created_at: datetime = Field(default_factory=datetime.utcnow)
    fact_id: Optional[str] = None
    superseded_by: Optional[str] = None
