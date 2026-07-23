"""Unit tests for the Normalizer — pure Python, no LLM needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schema.turn import RawTurn, ToolCall
from services.compiler.pipeline import normalizer


def test_extract_named_entities_url_and_date():
    text = "Check https://example.com/doc on 2026-07-22 for details."
    entities = normalizer.extract_named_entities(text)
    assert "https://example.com/doc" in entities
    assert "2026-07-22" in entities


def test_extract_named_entities_file_path():
    text = "Edit services/compiler/pipeline/normalizer.py to fix the bug."
    entities = normalizer.extract_named_entities(text)
    assert any("normalizer.py" in e for e in entities)


def test_extract_named_entities_code_block():
    text = "Here is the fix:\n```python\nprint('hi')\n```"
    entities = normalizer.extract_named_entities(text)
    assert "[code_block]" in entities


def test_normalize_turn_strips_tool_call_binary_keeps_text():
    raw = RawTurn(
        role="assistant",
        content="Running the tests now.",
        timestamp="2026-07-22T10:00:00Z",
        message_id="a1",
        tool_calls=[
            ToolCall(name="run_tests", arguments={"path": "."}, result="5 passed")
        ],
    )
    turn = normalizer.normalize_turn(
        raw, session_id="sess_1", source_agent="claude_cli"
    )
    assert "Running the tests now." in turn.content
    assert "[tool_call:run_tests]" in turn.content
    assert "5 passed" in turn.content
    assert turn.session_id == "sess_1"
    assert turn.source_agent == "claude_cli"


def test_normalize_session_dict_roundtrip():
    session_doc = {
        "session_id": "sess_1",
        "source_agent": "claude_cli",
        "model": "claude-sonnet-4-6",
        "turns": [
            {
                "role": "user",
                "content": "Let's use Graphiti.",
                "timestamp": "2026-07-22T10:00:00Z",
                "message_id": "u1",
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "Sounds good.",
                "timestamp": "2026-07-22T10:00:05Z",
                "message_id": "a1",
                "tool_calls": [],
            },
        ],
    }
    normalized = normalizer.normalize_session_dict(session_doc)
    assert normalized.session_id == "sess_1"
    assert len(normalized.turns) == 2
    assert normalized.turns[0].message_id == "u1"
    assert normalized.turns[1].role == "assistant"


def test_normalize_turn_truncates_long_tool_call_result():
    raw = RawTurn(
        role="assistant",
        content="Reading file",
        timestamp="2026-07-22T10:00:00Z",
        message_id="a1",
        tool_calls=[
            ToolCall(name="read_file", arguments={"path": "."}, result="A" * 1500)
        ],
    )
    turn = normalizer.normalize_turn(
        raw, session_id="sess_1", source_agent="claude_cli"
    )
    assert "[tool_call:read_file]" in turn.content
    assert "truncated" in turn.content
    assert len(turn.content) < 1500
