"""Score every eligible TIMIT clip for a speaker against the LIVE server
(both /demo/whisper and /ws/audio?force_patient=1) so the ranking
reflects what a judge will see on the /how-it-works demo slide —
including the safety-gate flag, which the older find_demo_clips.py
predates.

Two buckets are surfaced:

  * "accuracy wins"  — Phonemer's adapted text has materially lower
                        WER than vanilla AND the safety gate did not
                        flag. The judge sees better transcription.
  * "safety catches" — Phonemer flagged the clip (alignment doesn't
                        support the words) while vanilla confidently
                        printed something. The judge sees honest
                        uncertainty instead of hallucination.

Usage:
    cd server
    .venv/bin/python -m scripts.rank_demo_clips --speaker M01
    .venv/bin/python -m scripts.rank_demo_clips --speaker M01 --split eval
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
import websockets

SERVER_HTTP = "http://localhost:8000"
SERVER_WS = "ws://localhost:8000"


def normalize_words(text: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for ch in text.lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.append(cur)
                cur = ""
    if cur:
        out.append(cur)
    return out


def wer(reference: str, hypothesis: str) -> float:
    r = normalize_words(reference)
    h = normalize_words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    rows = len(r) + 1
    cols = len(h) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(1, len(r))


async def stream_clip(audio_bytes: bytes, adapter_speaker: str) -> dict[str, Any]:
    """Stream a clip's PCM through /ws/audio and collect every translation
    event. Forces the patient path and pins the per-speaker adapter so
    the server doesn't fall back to its default ``patient`` symlink
    (which is one specific speaker's adapter and produces nonsense for
    every other speaker's audio). WAV PCM starts after the 44-byte
    header.
    """
    url = (
        f"{SERVER_WS}/ws/audio?force_patient=1"
        f"&adapter_speaker={adapter_speaker}"
    )
    translations: list[dict] = []
    error: str | None = None
    pcm = audio_bytes[44:]
    try:
        async with websockets.connect(url, max_size=None) as ws:
            CHUNK = 3200  # 1600 int16 samples per chunk
            for i in range(0, len(pcm), CHUNK):
                await ws.send(pcm[i : i + CHUNK])
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "stop"}))
            # whisper-medium.en (M01's adapter) on CPU takes 20-30 s per
            # utterance; whisper-small.en (F03/FC02) takes 3-5 s. Use 60 s
            # so neither speaker times out before producing a translation,
            # and reset on each event so multi-utterance clips finish.
            deadline = time.monotonic() + 60.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "translation":
                    translations.append(msg)
                    deadline = time.monotonic() + 10.0
                elif t == "translation_error":
                    error = msg.get("message") or "translation_error"
                    break
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    text = " ".join((t.get("primary") or "") for t in translations).strip()
    flagged = any(t.get("flagged") for t in translations)
    return {
        "text": text,
        "flagged": flagged,
        "flag_reason": next(
            (t.get("flag_reason") for t in translations if t.get("flagged")),
            None,
        ),
        "segments": len(translations),
        "error": error,
    }


def fetch_vanilla(clip_id: int) -> tuple[str | None, str | None]:
    try:
        r = requests.post(f"{SERVER_HTTP}/demo/whisper/{clip_id}", timeout=60)
        r.raise_for_status()
        return r.json().get("transcript", ""), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def fetch_audio(clip_id: int) -> bytes:
    r = requests.get(f"{SERVER_HTTP}/torgo/audio/{clip_id}", timeout=30)
    r.raise_for_status()
    return r.content


async def evaluate(speaker: str, split: str | None, min_words: int) -> list[dict]:
    clips = requests.get(
        f"{SERVER_HTTP}/torgo/clips?limit=20000", timeout=30
    ).json()["clips"]
    eligible = [
        c
        for c in clips
        if c.get("speaker_id") == speaker
        and c.get("prompt_set") == "timit"
        and len((c.get("intended_text") or "").split()) >= min_words
    ]
    if split:
        eligible = [c for c in eligible if c.get("split") == split]
    eligible.sort(key=lambda c: c["id"])

    print(
        f"Evaluating {len(eligible)} clip(s) for speaker={speaker} "
        f"split={split or 'any'} (TIMIT, ≥{min_words} words)",
        file=sys.stderr,
    )

    results: list[dict] = []
    for i, c in enumerate(eligible, 1):
        intended = c["intended_text"]
        try:
            audio = fetch_audio(c["id"])
        except Exception as e:
            print(
                f"  [{i}/{len(eligible)}] {c['id']}: audio fetch failed: {e}",
                file=sys.stderr,
            )
            continue
        vanilla_text, vanilla_err = fetch_vanilla(c["id"])
        adapted = await stream_clip(audio, speaker)
        vw = wer(intended, vanilla_text or "")
        aw = wer(intended, adapted["text"] or "")
        results.append(
            {
                "id": c["id"],
                "split": c.get("split"),
                "duration_ms": c.get("duration_ms"),
                "intended": intended,
                "vanilla_text": vanilla_text,
                "vanilla_err": vanilla_err,
                "adapted_text": adapted["text"],
                "adapted_flagged": adapted["flagged"],
                "adapted_flag_reason": adapted["flag_reason"],
                "adapted_segments": adapted["segments"],
                "adapted_err": adapted["error"],
                "vanilla_wer": vw,
                "adapted_wer": aw,
            }
        )
        flag = "F" if adapted["flagged"] else " "
        err = "E" if (vanilla_err or adapted["error"]) else " "
        print(
            f"  [{i:>3}/{len(eligible)}] id={c['id']:>5}  dur={c.get('duration_ms', 0) / 1000:>4.1f}s"
            f"  van={vw:.2f}  ad={aw:.2f}  {flag}{err}  '{intended[:55]}'",
            file=sys.stderr,
        )
    return results


def bucket(results: list[dict]) -> dict[str, list[dict]]:
    accuracy_wins, safety_catches, no_change, losses = [], [], [], []
    for r in results:
        if r["vanilla_err"] or r["adapted_err"]:
            continue
        delta = r["vanilla_wer"] - r["adapted_wer"]
        if r["adapted_flagged"]:
            safety_catches.append({**r, "delta": delta})
        elif delta > 0.05 and r["vanilla_wer"] > 0.1:
            accuracy_wins.append({**r, "delta": delta})
        elif abs(delta) <= 0.05:
            no_change.append({**r, "delta": delta})
        else:
            losses.append({**r, "delta": delta})
    accuracy_wins.sort(key=lambda r: -r["delta"])
    safety_catches.sort(key=lambda r: -r["vanilla_wer"])
    no_change.sort(key=lambda r: r["adapted_wer"])
    losses.sort(key=lambda r: r["delta"])
    return {
        "accuracy_wins": accuracy_wins,
        "safety_catches": safety_catches,
        "no_change": no_change,
        "losses": losses,
    }


def pp(r: dict) -> str:
    flag = "  ⚑ FLAGGED" if r["adapted_flagged"] else ""
    return (
        f"  clip {r['id']}  ({r.get('split')}, {r.get('duration_ms', 0) / 1000:.1f}s)\n"
        f"    intended : {r['intended']}\n"
        f"    vanilla  : {r['vanilla_text']!r}  WER {r['vanilla_wer'] * 100:.0f}%\n"
        f"    phonemer : {r['adapted_text']!r}  WER {r['adapted_wer'] * 100:.0f}%{flag}\n"
        f"    delta    : {r['delta'] * 100:+.0f}pp"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="M01")
    parser.add_argument("--split", default="eval", help="cal / eval / any")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    split = args.split if args.split != "any" else None
    results = asyncio.run(evaluate(args.speaker, split, args.min_words))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))

    b = bucket(results)
    print()
    print("=" * 78)
    print(f"  Demo-clip recommendations for {args.speaker}")
    print("=" * 78)
    print()
    print(f"ACCURACY WINS ({len(b['accuracy_wins'])})")
    print("  Phonemer transcribes better than vanilla, gate clean.")
    print()
    for r in b["accuracy_wins"][:5]:
        print(pp(r))
        print()
    print(f"SAFETY CATCHES ({len(b['safety_catches'])})")
    print("  Phonemer flagged the clip; vanilla didn't admit uncertainty.")
    print()
    for r in b["safety_catches"][:5]:
        print(pp(r))
        print()
    print(f"Summary on {len(results)} clips: "
          f"{len(b['accuracy_wins'])} accuracy wins, "
          f"{len(b['safety_catches'])} safety catches, "
          f"{len(b['no_change'])} ties, {len(b['losses'])} regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
