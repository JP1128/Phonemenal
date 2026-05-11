"""Stream a 16 kHz mono WAV file through /ws/audio and print pipeline output.

Usage:
    cd server
    python -m scripts.test_stream <path/to/file.wav>
"""
from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import soundfile as sf
import websockets


CHUNK_SAMPLES = 1600  # 100 ms of audio per chunk
PACING_SECONDS = 0.05


async def main(path: str) -> None:
    audio, sr = sf.read(path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        raise SystemExit(f"Expected 16 kHz audio, got {sr} Hz")
    audio = np.ascontiguousarray(audio)

    uri = "ws://localhost:8000/ws/audio"
    async with websockets.connect(uri) as ws:
        async def receiver() -> None:
            async for message in ws:
                payload = json.loads(message)
                _print_message(payload)

        recv_task = asyncio.create_task(receiver())
        try:
            for i in range(0, len(audio), CHUNK_SAMPLES):
                chunk = audio[i : i + CHUNK_SAMPLES].tobytes()
                await ws.send(chunk)
                await asyncio.sleep(PACING_SECONDS)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(2.0)  # let server flush + reply
        finally:
            recv_task.cancel()


def _print_message(payload: dict) -> None:
    kind = payload.get("type")
    if kind == "utterance":
        marker = "[PATIENT]" if payload["is_patient"] else f"[{payload['speaker']}]"
        sim = payload.get("speaker_similarity") or 0.0
        print(
            f"\n{marker}  utt={payload['utterance_id']}  "
            f"{payload['start_ms']}-{payload['end_ms']} ms  sim={sim:.2f}"
        )
    elif kind == "phoneme_window":
        for frame in payload["frames"]:
            top = frame["phonemes"][0]
            print(
                f"  t={frame['timestamp_ms']:6d}ms  "
                f"{top['symbol']:>4}  p={top['prob']:.3f}"
            )
    elif kind == "features":
        f = payload["features"]
        print(
            f"  features  pitch={f['pitch_hz_mean']:.0f}±{f['pitch_hz_std']:.0f}Hz  "
            f"jitter={f['jitter']:.3f}  shimmer={f['shimmer']:.3f}  "
            f"pauses={f['pause_count']}/{f['pause_total_ms']}ms  "
            f"VOT={f['vot_proxy_ms']:.0f}ms"
        )
    elif kind == "utterance_transcript":
        print(f"  transcript: {payload.get('transcript') or '(empty)'}")
    else:
        print(f"  ?? {payload}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.test_stream <wav>")
    asyncio.run(main(sys.argv[1]))
