"""Supplied local retriever. Do not modify.

A deterministic, dependency-free stand-in for the course's FastEmbed+FAISS
retriever: bag-of-words vectors and cosine similarity, so retrieval behaves
identically in the app and in tests with no model download or network call.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _vector(text: str) -> Counter:
    return Counter(_tokenize(text))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """notes: [{"note_id": str, "text": str}, ...]."""
    return [
        {"note_id": note["note_id"], "text": note["text"], "vector": _vector(note["text"])}
        for note in notes
    ]


def retrieve_top_k(index: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
    """Returns up to k entries as [{"note_id", "text", "score"}, ...],
    highest score first, ties broken by note_id. Entries with zero overlap
    with the query are excluded."""
    query_vector = _vector(query)
    scored = [
        {
            "note_id": entry["note_id"],
            "text": entry["text"],
            "score": _cosine(query_vector, entry["vector"]),
        }
        for entry in index
    ]
    scored.sort(key=lambda item: (-item["score"], item["note_id"]))
    return [item for item in scored[:k] if item["score"] > 0]
