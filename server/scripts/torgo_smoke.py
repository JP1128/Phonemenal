"""Pick N random TORGO clips, stream them through /ws/audio, sanity-check output.

Usage:
    cd server
    uvicorn app.main:app --port 8000 &              # in another shell
    python -m scripts.torgo_smoke --n 5

Asserts:
  * server emits at least one `phoneme_window` per patient-classified utterance.
  * top phoneme probability is non-degenerate (> 0.05) — otherwise the model
    is producing flat distributions (most often a sample-rate mismatch).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402


CHUNK_SAMPLES = 1600  # 100 ms
PACING_SECONDS = 0.05


async def run_clip(uri: str, wav_path: str) -> tuple[int, float]:
    """Stream one clip; return (frames_received, max_top_prob)."""
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        # Resample with librosa to match the model.
        import librosa

        audio = librosa.resample(
            audio.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000
        )
        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    audio = np.ascontiguousarray(audio)

    frames_received = 0
    max_top_prob = 0.0

    async with websockets.connect(uri) as ws:
        async def receiver() -> None:
            nonlocal frames_received, max_top_prob
            async for message in ws:
                payload = json.loads(message)
                if payload.get("type") == "phoneme_window":
                    for frame in payload["frames"]:
                        frames_received += 1
                        top = frame["phonemes"][0]["prob"]
                        max_top_prob = max(max_top_prob, top)

        recv_task = asyncio.create_task(receiver())
        try:
            for i in range(0, len(audio), CHUNK_SAMPLES):
                chunk = audio[i : i + CHUNK_SAMPLES].tobytes()
                await ws.send(chunk)
                await asyncio.sleep(PACING_SECONDS)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(2.0)
        finally:
            recv_task.cancel()

    return frames_received, max_top_prob


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--uri", default="ws://localhost:8000/ws/audio")
    parser.add_argument("--condition", default=None,
                        help="filter clips by condition: control / dysarthria")
    parser.add_argument("--db", default=settings.db_path)
    args = parser.parse_args()

    conn = connect(args.db)
    clips = list_torgo_clips(conn, condition=args.condition, limit=10000)
    if not clips:
        raise SystemExit(
            "torgo_manifest is empty. Run scripts/torgo_loader.py first."
        )

    sample = random.sample(clips, min(args.n, len(clips)))
    failures: list[str] = []
    for clip in sample:
        frames, top = await run_clip(args.uri, clip["wav_path"])
        ok = frames > 0 and top > 0.05
        status = "OK" if ok else "FAIL"
        text = (clip["intended_text"] or "")[:40]
        print(
            f"[{status}] {clip['speaker_id']:>5}  frames={frames:4d}  "
            f"top_p={top:.3f}  {text!r}"
        )
        if not ok:
            failures.append(clip["wav_path"])

    if failures:
        print(f"\n{len(failures)}/{len(sample)} clips produced degenerate output.")
        sys.exit(1)
    print(f"\nAll {len(sample)} clips passed.")


if __name__ == "__main__":
    asyncio.run(main())
