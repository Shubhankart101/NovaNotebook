"""Supplied session persistence. Do not modify.

Simple JSON-file-backed storage, one file per notebook_id, so a notebook's
state survives a process restart.
"""
from __future__ import annotations

import json
import os
from typing import Any


def save_session(storage_dir: str, notebook_id: str, state: dict[str, Any]) -> None:
    os.makedirs(storage_dir, exist_ok=True)
    path = os.path.join(storage_dir, f"{notebook_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def load_session(storage_dir: str, notebook_id: str) -> dict[str, Any] | None:
    path = os.path.join(storage_dir, f"{notebook_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
