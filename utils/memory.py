"""Supplied memory compaction. Do not modify.

Deterministic message compaction: once a conversation exceeds keep_recent
messages, the older prefix is collapsed into one summary message produced
by the injected summarize_fn (an LLM call in production, a deterministic
stub in tests), and the most recent keep_recent messages are kept verbatim.
"""
from __future__ import annotations

from typing import Any, Callable


def compact_history(
    messages: list[dict[str, Any]],
    keep_recent: int,
    summarize_fn: Callable[[list[dict[str, Any]]], str],
) -> list[dict[str, Any]]:
    if len(messages) <= keep_recent:
        return list(messages)

    older = messages[:-keep_recent] if keep_recent > 0 else messages
    recent = messages[-keep_recent:] if keep_recent > 0 else []
    summary_text = summarize_fn(older)
    summary_message = {
        "role": "system",
        "content": f"Summary of earlier conversation: {summary_text}",
    }
    return [summary_message, *recent]
