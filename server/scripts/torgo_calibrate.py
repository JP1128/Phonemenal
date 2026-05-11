"""Build distortion profiles from TORGO speakers; sanity-check the pipeline.

For each speaker in the manifest, treats their (intended_text + audio) clips
as a calibration corpus, runs the same alignment + profile pipeline that the
real calibration endpoint uses, and prints summary statistics.

A working pipeline should show:
  * Control speakers: confusion matrix concentrated on the diagonal
    (P(produced=intended) >= 0.5 for most phonemes).
  * Dysarthria speakers: noticeable off-diagonal mass and higher
    syllable_deletion_rate.

Usage:
    cd server
    python -m scripts.torgo_calibrate --speakers F01,FC01 --max-clips 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.alignment import alignment_score, forced_align, text_to_phonemes  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.distortion import build_profile  # noqa: E402
from app.phonemes import PhonemeExtractor  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402


def _load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", required=True, help="Comma-separated TORGO speaker ids")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        default=settings.min_alignment_score,
        help="Drop clips whose mean forced-alignment log-prob is below this.",
    )
    parser.add_argument(
        "--skip-non-word",
        action="store_true",
        default=True,
        help="Skip clips whose prompt_set is 'non-word' (DDK-style stimuli).",
    )
    parser.add_argument(
        "--split",
        choices=("cal", "eval", "all"),
        default="cal",
        help="Which split to calibrate on. Default: cal (the 80%% the eval "
        "harness never sees).",
    )
    args = parser.parse_args()

    extractor = PhonemeExtractor(settings.phoneme_model_name, device=settings.device)
    vocab = list(extractor.processor.tokenizer.get_vocab().keys())
    token_to_id = extractor.processor.tokenizer.get_vocab()
    blank_id = extractor.processor.tokenizer.pad_token_id or 0

    conn = connect(args.db)
    speakers = [s.strip() for s in args.speakers.split(",")]

    for speaker_id in speakers:
        # Pull all clips, apply the split, then truncate — so --max-clips
        # selects the first N clips of the requested split rather than
        # filtering after a possibly-truncated SQL LIMIT.
        clips = list_torgo_clips(conn, speaker_id=speaker_id, limit=1_000_000)
        clips = filter_by_split(clips, args.split)
        if args.max_clips:
            clips = clips[: args.max_clips]
        if not clips:
            print(f"[skip] {speaker_id}: no clips in manifest")
            continue
        print(f"\n=== {speaker_id} ({len(clips)} clips, split={args.split}) ===")

        aligned_utterances = []
        rejected_low_score = 0
        rejected_non_word = 0
        for clip in clips:
            text = clip.get("intended_text") or ""
            if not text:
                continue
            if args.skip_non_word and clip.get("prompt_set") == "non-word":
                rejected_non_word += 1
                continue
            audio = _load_audio(clip["wav_path"])
            if audio is None:
                continue
            probs = extractor.extract(audio, 16000)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            intended = text_to_phonemes(text)
            words_ph = [text_to_phonemes(w) for w in text.split()]
            aligned = forced_align(log_probs, intended, token_to_id, blank_id)
            score = alignment_score(aligned)
            if score < args.min_alignment_score:
                rejected_low_score += 1
                continue
            top = extractor.top_k_per_frame(probs, settings.top_k_phonemes)
            aligned_utterances.append(
                {
                    "intended_tokens": intended,
                    "intended_words_phonemes": words_ph,
                    "aligned": aligned,
                    "produced": top,
                }
            )

        if rejected_low_score or rejected_non_word:
            print(
                f"  rejected: {rejected_low_score} low-alignment, "
                f"{rejected_non_word} non-word"
            )

        if not aligned_utterances:
            print(f"  (no alignable clips)")
            continue

        profile = build_profile(aligned_utterances, vocab_symbols=vocab)
        diag, off_diag = _diag_mass(profile)
        print(f"  utterances aligned       : {profile.n_calibration_utterances}")
        print(f"  diagonal probability mass: {diag:.3f}")
        print(f"  off-diagonal mass        : {off_diag:.3f}")
        print(f"  syllable deletion rate   : {profile.syllable_deletion_rate:.3f}")
        top_confusions = _top_confusions(profile, k=5)
        if top_confusions:
            print("  top confusions (intended → produced, count):")
            for intended, produced, count in top_confusions:
                print(f"    {intended!r:>5} -> {produced!r:<5}  {count}")
    return 0


def _diag_mass(profile) -> tuple[float, float]:
    rows = profile.confusion_overall.rows
    counts = profile.confusion_overall.counts
    diag = 0
    total = 0
    for i, row in enumerate(counts):
        diag += row[i]
        total += sum(row)
    if total == 0:
        return 0.0, 0.0
    return diag / total, 1.0 - diag / total


def _top_confusions(profile, k: int = 5):
    rows = profile.confusion_overall.rows
    cols = profile.confusion_overall.cols
    out = []
    for i, row in enumerate(profile.confusion_overall.counts):
        for j, count in enumerate(row):
            if i == j or count == 0:
                continue
            out.append((rows[i], cols[j], count))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:k]


if __name__ == "__main__":
    raise SystemExit(main())
