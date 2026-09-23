"""Merge per-speaker rank_demo_clips.py JSON outputs into the single
``demo_ranking.json`` consumed by the TORGO drawer.

The output schema is keyed by speaker id (uppercase) and lists one
entry per ranked clip with the fields the client needs to drive its
sort + per-row badge:

    {
      "M01": [
        {
          "id": 16760,
          "intended": "The islands are sparsely populated",
          "adapted_text": "the islands are sparsely populated",
          "adapted_wer": 0.0,
          "adapted_flagged": false,
          "vanilla_wer": 1.0
        },
        ...
      ],
      "F03": [...],
      "FC02": [...]
    }

Usage (run from server/):
    .venv/bin/python -m scripts.merge_demo_ranking \
        --in M01=/tmp/rank_M01.json \
        --in F03=/tmp/rank_F03.json \
        --in FC02=/tmp/rank_FC02.json \
        --out ../client/public/demo_ranking.json

Designed to be re-runnable: missing inputs are skipped with a warning so
a partial demo ranking (e.g. only M01 finished) still produces a useful
file the drawer can sort against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLIENT_FIELDS = (
    "id",
    "intended",
    "adapted_text",
    "adapted_wer",
    "adapted_flagged",
    "vanilla_wer",
)


def _project(entry: dict) -> dict | None:
    """Drop fields the client doesn't need and skip entries that errored.

    The rank script keeps unrelated keys (segments, flag_reason, etc.)
    that don't drive the UI; the client schema is intentionally narrow
    so the JSON downloaded by every drawer-open stays small.
    """
    if entry.get("vanilla_err") or entry.get("adapted_err"):
        return None
    out = {k: entry.get(k) for k in CLIENT_FIELDS}
    # Cast booleans / numbers defensively in case the rank script ever
    # changes shape.
    out["adapted_flagged"] = bool(entry.get("adapted_flagged"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        action="append",
        dest="inputs",
        required=True,
        help="speaker=path/to/rank_<speaker>.json (repeatable)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output path for the merged demo_ranking.json",
    )
    args = parser.parse_args()

    merged: dict[str, list[dict]] = {}
    for spec in args.inputs:
        if "=" not in spec:
            print(f"  ✗ malformed --in {spec!r}; expected speaker=path", file=sys.stderr)
            return 1
        speaker, path_str = spec.split("=", 1)
        speaker = speaker.upper().strip()
        path = Path(path_str)
        if not path.exists():
            print(f"  · skip {speaker}: {path} not found", file=sys.stderr)
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  ✗ {speaker}: failed to parse {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"  ✗ {speaker}: {path} is not a list", file=sys.stderr)
            continue
        entries = [p for p in (_project(d) for d in data) if p is not None]
        # Sort within each speaker for human-readable diff: best (lowest
        # adapted_wer, unflagged first) at the top.
        entries.sort(
            key=lambda e: (bool(e["adapted_flagged"]), float(e["adapted_wer"]))
        )
        merged[speaker] = entries
        print(
            f"  ✓ {speaker}: {len(entries)} clip(s) "
            f"(min WER {min((e['adapted_wer'] for e in entries), default=float('nan')):.2f}, "
            f"flagged {sum(1 for e in entries if e['adapted_flagged'])})",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nwrote {out_path} ({sum(len(v) for v in merged.values())} clips total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
