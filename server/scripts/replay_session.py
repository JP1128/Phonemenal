"""Re-stream a recorded session's PCM into /ws/audio for backend regression.

Recordings produced by the debug page with "include audio" enabled contain
base64-encoded int16 PCM chunks. This script reconstructs the original byte
stream, sends it to a running server, and prints a structured diff between
the original recorded events and the freshly-produced events.

Usage:
    cd server
    python -m scripts.replay_session phonemer-session-12-2026-05-05.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--uri", default="ws://localhost:8000/ws/audio")
    parser.add_argument("--realtime", action="store_true",
                        help="Pace PCM replay at the original wall-clock cadence")
    args = parser.parse_args()

    rec = json.loads(args.recording.read_text())
    if rec.get("version") != 1:
        raise SystemExit(f"Unsupported recording version: {rec.get('version')}")
    pcm = rec.get("pcm")
    if not pcm:
        raise SystemExit(
            "Recording has no captured PCM (was 'include audio' enabled?)."
        )

    expected_events = [e for e in rec["events"]]
    new_events: list[dict] = []

    async with websockets.connect(args.uri) as ws:
        async def receiver() -> None:
            async for msg in ws:
                new_events.append(json.loads(msg))

        recv_task = asyncio.create_task(receiver())
        prev_t = 0.0
        try:
            for chunk in pcm:
                if args.realtime:
                    delay = max(0.0, (chunk["t_ms"] - prev_t) / 1000.0)
                    if delay:
                        await asyncio.sleep(delay)
                    prev_t = chunk["t_ms"]
                await ws.send(base64.b64decode(chunk["data_b64"]))
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(2.0)
        finally:
            recv_task.cancel()

    print(f"replay produced {len(new_events)} events; recording had {len(expected_events)}")
    diff = _summarize_diff(expected_events, new_events)
    print(diff)


def _summarize_diff(expected: list, actual: list) -> str:
    def kinds(events: list) -> dict[str, int]:
        out: dict[str, int] = {}
        for ev in events:
            payload = ev["payload"] if "payload" in ev else ev
            out[payload["type"]] = out.get(payload["type"], 0) + 1
        return out

    e = kinds(expected)
    a = kinds(actual)
    keys = sorted(set(e) | set(a))
    rows = [f"  {k:>22}: expected={e.get(k, 0):>4}  actual={a.get(k, 0):>4}"
            for k in keys]
    return "\n".join(rows)


if __name__ == "__main__":
    asyncio.run(main())
