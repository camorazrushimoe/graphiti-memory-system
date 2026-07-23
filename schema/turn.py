"""Shared Turn schema — used by ingest and compiler services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result: Optional[str] = None


class RawTurn(BaseModel):
    """Turn as received from an agent hook, before normalization."""

    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime
    message_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Turn(BaseModel):
    """Normalized turn — unified fields, ready for the compiler pipeline."""

    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime
    message_id: str
    session_id: str
    source_agent: str
    named_entities: list[str] = Field(default_factory=list)  # spaCy pre-tag


class IngestTurnRequest(BaseModel):
    """Payload for POST /ingest/turn"""

    session_id: str
    source_agent: str
    model: str
    turn: RawTurn


class IngestSessionRequest(BaseModel):
    """Payload for POST /ingest (full session dump / replay)"""

    session_id: str
    source_agent: str
    model: str
    turns: list[RawTurn]


class IngestCloseRequest(BaseModel):
    """Payload for POST /ingest/close"""

    session_id: str


class NormalizedSession(BaseModel):
    session_id: str
    source_agent: str
    model: str
    turns: list[Turn]
