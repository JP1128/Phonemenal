"""Pull the TORGO mirror that ships paired transcriptions from Hugging Face.

The Kaggle archive layout we ingested earlier is audio-only — every clip in
``torgo_manifest`` has ``intended_text=NULL``, which silently breaks
calibration, forced alignment, and any WER/PER evaluation. ``abnerh/TORGO-database``
on Hugging Face mirrors the same audio with the prompt text attached
(``transcription`` field) and is the simplest way to fix that.

For each sample, we materialize the wav to ``data/torgo/hf/<speaker_id>/<filename>``
so the rest of the pipeline (which expects a real path on disk via
``torgo_manifest.wav_path``) keeps working unchanged. Re-runs are idempotent —
the upsert is keyed on ``wav_path``, and we skip writing the wav when it
already exists.

Usage:
    cd server
    python -m scripts.torgo_hf_loader --limit 20         # preview 20 samples
    python -m scripts.torgo_hf_loader                    # full ingest (~16k)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, upsert_torgo_clip  # noqa: E402
from scripts.torgo_loader import infer_prompt_set  # noqa: E402

logger = logging.getLogger("torgo_hf_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


HF_DATASET = "abnerh/TORGO-database"
TARGET_SAMPLE_RATE = 16000

# Filenames look like ``FC01_1_arrayMic_0066.wav`` — speaker / session / mic / clip.
FILENAME_RE = re.compile(
    r"^(?P<speaker>(?:F|FC|M|MC)\d{2})"
    r"_(?P<session>\d+)"
    r"_(?P<mic>arrayMic|headMic)"
    r"_(?P<clip>\d+)\.wav$"
)


def _condition_from_status(speech_status: str | None) -> str:
    if (speech_status or "").lower() == "healthy":
        return "control"
    return "dysarthria"


def _parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    return {
        "speaker_id": m.group("speaker"),
        "session": f"S{int(m.group('session')):02d}",
        "mic": m.group("mic"),
        "clip_id": m.group("clip"),
    }


def _materialize_wav(audio_field: dict, target: Path) -> tuple[int, int]:
    """Persist the dataset's audio bytes to ``target``; return (sample_rate, n_samples).

    The dataset column was cast with ``Audio(decode=False)``, so each sample
    is ``{"path": str, "bytes": bytes | None}``. When ``bytes`` is populated
    we write it verbatim; when only ``path`` is set we copy the cached file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = audio_field.get("bytes")
    if raw is None:
        src = audio_field.get("path")
        if not src:
            raise ValueError("audio field has neither bytes nor path")
        raw = Path(src).read_bytes()
    target.write_bytes(raw)
    info = sf.info(str(target))
    return int(info.samplerate), int(info.frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest abnerh/TORGO-database from HF.")
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--out-dir",
        default="data/torgo/hf",
        help="Where to materialize the wav files on disk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many samples (0 = all). Useful for previewing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write wavs or DB rows; just log parsed metadata.",
    )
    parser.add_argument(
        "--prefer-mic",
        choices=("arrayMic", "headMic", "any"),
        default="arrayMic",
        help="Mic preference. 'arrayMic' (default) skips head-mic duplicates.",
    )
    args = parser.parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. pip install datasets, "
            "or pip install -r requirements.txt."
        ) from exc

    logger.info("Loading %s split=train (this can take a few min on first run)…", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")
    # Skip torchcodec/numpy decoding — the column already contains wav bytes
    # we can persist verbatim. Faster and removes the heavy decoder dep.
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info("Loaded %d samples", len(ds))

    out_dir = Path(args.out_dir).resolve()
    conn = None if args.dry_run else connect(args.db)

    inserted = 0
    skipped_filename = 0
    skipped_mic = 0
    skipped_short_text = 0
    for sample in ds:
        if args.limit and inserted >= args.limit:
            break

        audio = sample.get("audio") or {}
        path = audio.get("path") or ""
        name = Path(path).name
        meta = _parse_filename(name)
        if meta is None:
            skipped_filename += 1
            continue
        if args.prefer_mic != "any" and meta["mic"] != args.prefer_mic:
            skipped_mic += 1
            continue

        transcription = (sample.get("transcription") or "").strip()
        if not transcription:
            skipped_short_text += 1
            continue

        condition = _condition_from_status(sample.get("speech_status"))
        prompt_set = infer_prompt_set(transcription)
        speaker_id = meta["speaker_id"]
        wav_target = out_dir / speaker_id / name

        if args.dry_run:
            logger.info(
                "[dry-run] %-32s spk=%s sess=%s mic=%s set=%s cond=%s text=%r",
                name,
                speaker_id,
                meta["session"],
                meta["mic"],
                prompt_set,
                condition,
                transcription[:60],
            )
            inserted += 1
            continue

        if wav_target.exists():
            info = sf.info(str(wav_target))
            sr, n_frames = int(info.samplerate), int(info.frames)
        else:
            sr, n_frames = _materialize_wav(audio, wav_target)
        duration_ms = int(1000 * n_frames / sr) if sr else 0

        upsert_torgo_clip(
            conn,
            speaker_id=speaker_id,
            condition=condition,
            intelligibility=meta["session"],
            prompt_set=prompt_set,
            intended_text=transcription,
            wav_path=str(wav_target),
            duration_ms=duration_ms,
            sample_rate=sr,
        )
        inserted += 1
        if inserted % 500 == 0:
            logger.info(
                "ingested %d (skipped: filename=%d mic=%d empty_text=%d)",
                inserted,
                skipped_filename,
                skipped_mic,
                skipped_short_text,
            )

    logger.info(
        "done: %d ingested (skipped: filename=%d mic=%d empty_text=%d)",
        inserted,
        skipped_filename,
        skipped_mic,
        skipped_short_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
