"""Retrieval API schemas — Layer 7.

Request/response models for `POST /retrieve` (see tech spec, Layer 7 —
Retrieval Service). Kept in `schema/` alongside turn.py/memory_event.py so
both a future dedicated `retrieval` service and the current `ingest`
service (which hosts `/retrieve` per the spec's services table) can share
the same contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Payload for POST /retrieve."""

    query: str
    source_agent: Optional[str] = None
    top_k: int = 10
    recency_days: Optional[int] = None
    project_scope: Optional[str] = None


class MemoryPacketFact(BaseModel):
    text: str
    type: str
    confidence: float
    source_session: Optional[str] = None
    date: Optional[str] = None
    score: Optional[float] = None


class MemoryPacketEntity(BaseModel):
    name: str
    type: Optional[str] = None
    last_seen: Optional[str] = None


class MemoryPacketTask(BaseModel):
    text: str
    status: str = "open"


class MemoryPacketDecision(BaseModel):
    text: str
    date: Optional[str] = None


class MemoryPacket(BaseModel):
    """Response of POST /retrieve — see tech spec Layer 7 memory_packet format."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    query: str
    facts: list[MemoryPacketFact] = Field(default_factory=list)
    entities: list[MemoryPacketEntity] = Field(default_factory=list)
    open_tasks: list[MemoryPacketTask] = Field(default_factory=list)
    recent_decisions: list[MemoryPacketDecision] = Field(default_factory=list)
