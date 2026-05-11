"""Fatigue tracking — per-utterance error density bucketed into phases.

Spec:
  - fresh:   error_density < 0.15
  - mild:    0.15 <= ed < 0.30
  - moderate:0.30 <= ed < 0.50
  - severe:  ed >= 0.50

We compute error density as the fraction of frames whose top phoneme
disagrees with the intended sequence (during calibration we have ground
truth) or with a rolling baseline (during live sessions).

Snapshots are written every minute via ``snapshot()``.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from .db import write_fatigue_snapshot


_BUCKETS = [
    ("fresh", 0.15),
    ("mild", 0.30),
    ("moderate", 0.50),
    ("severe", float("inf")),
]


def phase_for(error_density: float) -> str:
    for label, ceiling in _BUCKETS:
        if error_density < ceiling:
            return label
    return "severe"


@dataclass
class FatigueTracker:
    session_id: int
    started_at: float = field(default_factory=time.time)
    _per_minute_errors: list[int] = field(default_factory=list)
    _per_minute_total: list[int] = field(default_factory=list)
    _last_snapshot_minute: int = -1

    def observe(self, error: bool) -> None:
        minute = int((time.time() - self.started_at) // 60)
        while len(self._per_minute_errors) <= minute:
            self._per_minute_errors.append(0)
            self._per_minute_total.append(0)
        self._per_minute_errors[minute] += 1 if error else 0
        self._per_minute_total[minute] += 1

    def snapshot(self, conn: sqlite3.Connection) -> dict | None:
        minute = int((time.time() - self.started_at) // 60)
        if minute == self._last_snapshot_minute:
            return None
        self._last_snapshot_minute = minute
        if minute >= len(self._per_minute_total):
            return None
        total = self._per_minute_total[minute]
        if total == 0:
            return None
        ed = self._per_minute_errors[minute] / total
        ph = phase_for(ed)
        write_fatigue_snapshot(conn, self.session_id, minute, ed, ph)
        return {"minute": minute, "error_density": ed, "phase": ph}
