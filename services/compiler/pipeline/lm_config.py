"""DSPy LM configuration — points at the local LM Studio OpenAI-compatible
endpoint. Shared by all DSPy modules in the compiler pipeline.
"""

from __future__ import annotations

import os

import dspy

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://192.168.0.11:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-e4b-it")

_configured = False


def configure_dspy(max_tokens: int = 2048, timeout: float = 300.0) -> dspy.LM:
    """Configure DSPy's global LM to use Gemma 4B via LM Studio.

    Idempotent — safe to call multiple times (e.g. once per module import).
    """
    global _configured
    lm = dspy.LM(
        model=f"openai/{LLM_MODEL}",
        api_base=LM_STUDIO_URL,
        api_key="lm-studio",  # LM Studio doesn't check the key, but the field is required
        max_tokens=max_tokens,
        timeout=timeout,
    )
    dspy.configure(lm=lm)
    _configured = True
    return lm
