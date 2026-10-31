"""Supplied graph/session state. Do not modify."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    notebook_id: str
    messages: list[dict[str, Any]]
    plan: list[str]
    retrieved: list[dict[str, Any]]
    draft_answer: str
    citations: list[str]
    final_answer: str
