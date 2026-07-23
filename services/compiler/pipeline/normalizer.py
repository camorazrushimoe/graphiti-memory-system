"""Step 1 · Normalizer — Layer 3 pipeline entry point.

Pure Python, no LLM calls — fully unit-testable.

Responsibilities:
  - Flattens all turns into the standard `Turn` schema (schema/turn.py)
  - Tags named entities (dates, file paths, URLs, code blocks) using spaCy
    rules (falls back to lightweight regex rules if the spaCy model isn't
    installed, so this module never hard-fails in a minimal environment)
  - Strips tool call binary content, keeps text representations
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3])
)  # repo root, for `schema` package

from schema.turn import NormalizedSession, RawTurn, Turn  # noqa: E402

# --- Regex fallbacks (used if spaCy / en_core_web_sm is unavailable) -------
_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(r"(?:[\w./~-]+/)+[\w.-]+\.[A-Za-z0-9]{1,10}")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

_NLP = None
_SPACY_AVAILABLE = False


def _load_spacy():
    """Lazily load spaCy's en_core_web_sm model, if installed."""
    global _NLP, _SPACY_AVAILABLE
    if _NLP is not None or _SPACY_AVAILABLE:
        return
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except Exception:
        _NLP = None
        _SPACY_AVAILABLE = False


def extract_named_entities(text: str) -> list[str]:
    """Tag named entities in `text`.

    Uses spaCy NER (PERSON, ORG, GPE, DATE, PRODUCT, ...) when available,
    always supplemented by regex-based rules for URLs, file paths, explicit
    ISO dates, and fenced code blocks (spaCy doesn't reliably catch these).
    """
    entities: list[str] = []

    _load_spacy()
    if _SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text)
        entities.extend(ent.text for ent in doc.ents)

    entities.extend(_URL_RE.findall(text))
    entities.extend(_FILE_PATH_RE.findall(text))
    entities.extend(_DATE_RE.findall(text))
    if _CODE_BLOCK_RE.search(text):
        entities.append("[code_block]")

    # de-dupe while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def _strip_tool_call_binary(turn: RawTurn) -> str:
    """Return the turn's content with tool call binary payloads stripped,
    keeping only text representations (tool name + text result, if any).
    """
    content = turn.content
    if turn.tool_calls:
        tool_summaries = []
        for tc in turn.tool_calls:
            result_text = tc.result if isinstance(tc.result, str) else None
            summary = f"[tool_call:{tc.name}]"
            if result_text:
                summary += f" -> {result_text}"
            tool_summaries.append(summary)
        if tool_summaries:
            content = (
                content + "\n" + "\n".join(tool_summaries)
                if content
                else "\n".join(tool_summaries)
            )
    return content


def normalize_turn(raw_turn: RawTurn, session_id: str, source_agent: str) -> Turn:
    content = _strip_tool_call_binary(raw_turn)
    return Turn(
        role=raw_turn.role,
        content=content,
        timestamp=raw_turn.timestamp,
        message_id=raw_turn.message_id,
        session_id=session_id,
        source_agent=source_agent,
        named_entities=extract_named_entities(content),
    )


def normalize_session(
    session_id: str, source_agent: str, model: str, raw_turns: Iterable[RawTurn]
) -> NormalizedSession:
    turns = [normalize_turn(rt, session_id, source_agent) for rt in raw_turns]
    return NormalizedSession(
        session_id=session_id, source_agent=source_agent, model=model, turns=turns
    )


def normalize_session_dict(session_doc: dict) -> NormalizedSession:
    """Convenience entry point used by the compiler worker — takes the raw
    session dict as loaded from the Raw Archive JSON file.
    """
    raw_turns = [RawTurn.model_validate(t) for t in session_doc["turns"]]
    return normalize_session(
        session_id=session_doc["session_id"],
        source_agent=session_doc["source_agent"],
        model=session_doc.get("model", ""),
        raw_turns=raw_turns,
    )
