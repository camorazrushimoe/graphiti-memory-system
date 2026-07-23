"""Instructor client configuration — points at the local LM Studio
OpenAI-compatible endpoint. Used by the Memory Extractor (Step 4), which
needs enforced JSON schema output rather than DSPy's free-text signatures.
"""

from __future__ import annotations

import os

import instructor
from openai import OpenAI

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://192.168.0.11:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-e4b-it")

_client = None


def get_instructor_client():
    """Lazily create and cache the Instructor-wrapped OpenAI client.

    LM Studio's OpenAI-compatible server requires `response_format.type`
    to be either 'json_schema' or 'text' — Instructor's default JSON mode
    (tool-calling based) is rejected, so we use JSON_SCHEMA mode.
    """
    global _client
    if _client is None:
        _client = instructor.from_openai(
            OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio"),
            mode=instructor.Mode.JSON_SCHEMA,
        )
    return _client
