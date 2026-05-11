"""Deterministic per-clip cal/eval split for TORGO evaluations.

Calibration utterances and evaluation utterances must come from disjoint
sets — otherwise eval metrics measure how well the model fits its own
training data, not how well it generalizes. We assign each clip a stable
hash bucket so the split is reproducible across runs and across speakers
without storing any per-clip flag in the database.

The default 20% eval / 80% cal mirrors what Phonemer's calibration phase
actually consumes (one calibration session, then live use).
"""
from __future__ import annotations

import hashlib

DEFAULT_SEED = 42
DEFAULT_EVAL_FRAC = 0.2

Split = str  # one of "cal", "eval"


def split_for_clip(
    clip_id: int,
    speaker_id: str,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Return ``"eval"`` for the held-out fraction, otherwise ``"cal"``.

    Hash is stable across machines and Python releases (sha256 of a
    tagged string) so the split survives db wipes and re-ingests.
    """
    key = f"{seed}|{speaker_id}|{clip_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / (1 << 64)
    return "eval" if bucket < eval_frac else "cal"


def filter_by_split(
    clips: list[dict],
    split: str,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Keep only clips matching ``split`` (``"cal"``, ``"eval"``, or ``"all"``)."""
    if split == "all":
        return list(clips)
    if split not in ("cal", "eval"):
        raise ValueError(f"split must be cal/eval/all, got {split!r}")
    return [
        c
        for c in clips
        if split_for_clip(int(c["id"]), c["speaker_id"], eval_frac, seed) == split
    ]
