"""Phase 8.3 — Personal Vocabulary.

A small caregiver-managed list of names, medications, and preferred terms.
Surfaced to the LLM as a vocabulary boost so candidate generation prefers
these over phonetically-close generic words.
"""
from __future__ import annotations

import sqlite3

from .db import (
    add_personal_vocabulary,
    delete_personal_vocabulary,
    list_personal_vocabulary,
)

ALLOWED_KINDS = ("name", "medication", "place", "preferred", "other")


def vocabulary_prompt_block(conn: sqlite3.Connection, max_terms: int = 32) -> str:
    rows = list_personal_vocabulary(conn)
    if not rows:
        return ""
    rows = rows[:max_terms]
    by_kind: dict[str, list[str]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(
            f"{r['term']}" + (f" (boost {r['boost']:.1f})" if r["boost"] != 1.5 else "")
        )
    parts = ["Patient personal vocabulary (boost in candidate generation):"]
    for kind in ("name", "medication", "place", "preferred", "other"):
        terms = by_kind.get(kind)
        if not terms:
            continue
        parts.append(f"  {kind}: {', '.join(terms)}")
    return "\n".join(parts)


def upsert_term(
    conn: sqlite3.Connection,
    kind: str,
    term: str,
    notes: str | None = None,
    boost: float = 1.5,
) -> dict:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"kind must be one of {ALLOWED_KINDS}; got {kind!r}")
    if not term.strip():
        raise ValueError("term cannot be empty")
    add_personal_vocabulary(conn, kind, term.strip(), notes, boost)
    rows = list_personal_vocabulary(conn, kind)
    return next(r for r in rows if r["term"] == term.strip())


def remove_term(conn: sqlite3.Connection, vocab_id: int) -> bool:
    return delete_personal_vocabulary(conn, vocab_id)
