"""Raw Archive writer — Layer 2.

Immutable-ish store of all raw session JSON. Each session is a single JSON
file located at data/raw/YYYY/MM/DD/<session_id>.json (date = first turn's
date). Turns are appended to that file as they arrive.

Writes are atomic: write to a temporary file then os.replace() into place,
so a crash mid-write never corrupts the archive.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RAW_ARCHIVE_PATH = Path(os.environ.get("RAW_ARCHIVE_PATH", "./data/raw"))


def _session_dir_for_date(dt: datetime) -> Path:
    return RAW_ARCHIVE_PATH / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"


def _find_existing_session_file(session_id: str) -> Optional[Path]:
    """Search the archive for an existing file for this session_id.

    Sessions are date-partitioned by the day they were first seen, so once
    created the file always lives under that original date directory.
    """
    if not RAW_ARCHIVE_PATH.exists():
        return None
    for path in RAW_ARCHIVE_PATH.glob(f"*/*/*/{session_id}.json"):
        return path
    return None


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def load_session(session_id: str) -> Optional[dict]:
    path = _find_existing_session_file(session_id)
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def append_turn(session_id: str, source_agent: str, model: str, turn: dict) -> dict:
    """Append a single turn to the session's raw archive file.

    Creates the file (and date partition) on first turn. Returns the full
    session document after the append.
    """
    existing_path = _find_existing_session_file(session_id)

    if existing_path is not None:
        with open(existing_path) as f:
            session_doc = json.load(f)
        session_doc["turns"].append(turn)
        _atomic_write_json(existing_path, session_doc)
        return session_doc

    # New session — partition by today's date (UTC)
    now = datetime.now(timezone.utc)
    session_doc = {
        "session_id": session_id,
        "source_agent": source_agent,
        "model": model,
        "turns": [turn],
    }
    path = _session_dir_for_date(now) / f"{session_id}.json"
    _atomic_write_json(path, session_doc)
    return session_doc


def write_full_session(
    session_id: str, source_agent: str, model: str, turns: list[dict]
) -> dict:
    """Write (or overwrite) a full session dump — used for the replay/fallback endpoint."""
    existing_path = _find_existing_session_file(session_id)
    now = datetime.now(timezone.utc)
    path = existing_path or (_session_dir_for_date(now) / f"{session_id}.json")

    session_doc = {
        "session_id": session_id,
        "source_agent": source_agent,
        "model": model,
        "turns": turns,
    }
    _atomic_write_json(path, session_doc)
    return session_doc


def list_unprocessed_sessions(processed_ids: set[str]) -> list[str]:
    """Scan the raw archive for session files not present in `processed_ids`."""
    if not RAW_ARCHIVE_PATH.exists():
        return []
    found = []
    for path in RAW_ARCHIVE_PATH.glob("*/*/*/*.json"):
        session_id = path.stem
        if session_id not in processed_ids:
            found.append(session_id)
    return found
