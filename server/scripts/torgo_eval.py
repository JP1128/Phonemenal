"""Offline evaluation harness for the Phonemer pipeline against TORGO.

Computes the metrics that answer "is the model listening, and how much
does our calibrated pipeline beat naive ASR?":

  * PER (phoneme error rate): greedy CTC decode of wav2vec2 logits vs.
    espeak phonemization of the intended text. Controls should land
    around 0.05-0.20; dysarthria patients vary widely by intelligibility.
  * Alignment score: mean per-token log-prob from forced alignment.
    Correlates with "did the model hear what was said?" — clips below
    -3.5 are usually silence / throat-clear / mismatched transcript.
  * Whisper WER: faster-whisper transcription vs. intended text. The
    "naive ASR baseline" Phonemer needs to beat. Skipped by default
    (slow); pass --with-whisper to include.

Outputs a per-speaker × per-condition summary, plus an optional verbose
per-clip line. The non-word DDK clips (prompt_set='non-word') are
excluded by default since they aren't lexical content.

Usage:
    cd server
    python -m scripts.torgo_eval --speakers FC01,F01 --limit 50
    python -m scripts.torgo_eval --condition control --with-whisper --limit 100
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.alignment import alignment_score, forced_align, text_to_phonemes  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.phonemes import PhonemeExtractor  # noqa: E402
from app.vocab import normalize_phonemes  # noqa: E402


# CTC special tokens we never want to decode as actual phonemes.
_CTC_SPECIALS = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}


@dataclass
class ClipResult:
    speaker_id: str
    condition: str
    intended_text: str
    duration_ms: int
    per: float
    n_phonemes: int
    align_score: float
    whisper_text: str | None = None
    whisper_wer: float | None = None


@dataclass
class SpeakerStats:
    n: int = 0
    per_values: list[float] = field(default_factory=list)
    align_values: list[float] = field(default_factory=list)
    whisper_wers: list[float] = field(default_factory=list)


# ============================================================ metrics


def levenshtein(a: list[str], b: list[str]) -> int:
    """Standard token-level edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def greedy_ctc_decode(probs: np.ndarray, id_to_token: dict[int, str]) -> list[str]:
    """Argmax per frame, collapse runs, drop blanks/specials. Returns phonemes.

    The output is run through the same ``normalize_phonemes`` step the
    intended-text side uses, so length marks (``iː`` vs ``i``) and other
    surface differences don't artificially inflate PER.
    """
    ids = probs.argmax(axis=-1).tolist()
    raw: list[str] = []
    last = -1
    for i in ids:
        if i == last:
            continue
        last = i
        sym = id_to_token.get(int(i), "")
        if not sym or sym in _CTC_SPECIALS:
            continue
        raw.append(sym)
    return normalize_phonemes(raw)


def normalize_words(text: str) -> list[str]:
    """Lowercase + strip non-alphanumerics for WER comparison."""
    keep = []
    for w in text.lower().split():
        cleaned = "".join(c for c in w if c.isalnum() or c == "'")
        if cleaned:
            keep.append(cleaned)
    return keep


# ============================================================ pipeline


def _load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != settings.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
    return audio


def evaluate_clip(
    extractor: PhonemeExtractor,
    transcriber,
    clip: dict,
    id_to_token: dict[int, str],
    token_to_id: dict[str, int],
    blank_id: int,
) -> ClipResult | None:
    text = (clip.get("intended_text") or "").strip()
    if not text:
        return None
    audio = _load_audio(clip["wav_path"])
    if audio is None or audio.size == 0:
        return None

    probs = extractor.extract(audio, settings.sample_rate)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))

    intended_phonemes = text_to_phonemes(text)
    decoded = greedy_ctc_decode(probs, id_to_token)
    per_dist = levenshtein(intended_phonemes, decoded)
    per = per_dist / max(len(intended_phonemes), 1)

    aligned = forced_align(log_probs, intended_phonemes, token_to_id, blank_id)
    score = alignment_score(aligned)

    whisper_text = None
    whisper_wer = None
    if transcriber is not None:
        whisper_text = transcriber.transcribe(audio).strip()
        ref_words = normalize_words(text)
        hyp_words = normalize_words(whisper_text)
        if ref_words:
            whisper_wer = levenshtein(ref_words, hyp_words) / len(ref_words)

    return ClipResult(
        speaker_id=clip["speaker_id"],
        condition=clip.get("condition") or "?",
        intended_text=text,
        duration_ms=int(clip.get("duration_ms") or 0),
        per=per,
        n_phonemes=len(intended_phonemes),
        align_score=score,
        whisper_text=whisper_text,
        whisper_wer=whisper_wer,
    )


# ============================================================ reporting


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


def _print_per_clip(rows: Iterable[ClipResult], with_whisper: bool) -> None:
    if with_whisper:
        header = (
            f"{'speaker':<7} {'cond':<10} {'dur_s':>5} {'PER':>5} "
            f"{'align':>6} {'WERw':>5}  intended | whisper"
        )
    else:
        header = (
            f"{'speaker':<7} {'cond':<10} {'dur_s':>5} {'PER':>5} "
            f"{'align':>6}  intended"
        )
    print(header)
    print("-" * len(header))
    for r in rows:
        score_str = "  -inf" if r.align_score == float("-inf") else f"{r.align_score:>6.2f}"
        if with_whisper:
            wer_str = "    -" if r.whisper_wer is None else f"{r.whisper_wer:>5.2f}"
            tail = f"{r.intended_text[:40]!r} | {(r.whisper_text or '')[:40]!r}"
            print(
                f"{r.speaker_id:<7} {r.condition:<10} {r.duration_ms/1000:>5.2f} "
                f"{r.per:>5.2f} {score_str} {wer_str}  {tail}"
            )
        else:
            print(
                f"{r.speaker_id:<7} {r.condition:<10} {r.duration_ms/1000:>5.2f} "
                f"{r.per:>5.2f} {score_str}  {r.intended_text[:60]!r}"
            )


def _print_summary(rows: list[ClipResult], with_whisper: bool) -> None:
    by_key: dict[tuple[str, str], SpeakerStats] = {}
    for r in rows:
        key = (r.speaker_id, r.condition)
        bucket = by_key.setdefault(key, SpeakerStats())
        bucket.n += 1
        bucket.per_values.append(r.per)
        if r.align_score != float("-inf"):
            bucket.align_values.append(r.align_score)
        if r.whisper_wer is not None:
            bucket.whisper_wers.append(r.whisper_wer)

    print()
    print("== summary ==")
    if with_whisper:
        header = (
            f"{'speaker':<7} {'cond':<10} {'n':>4} "
            f"{'PER_mean':>9} {'PER_p50':>8} {'align':>7}  {'WERw_mean':>9}"
        )
    else:
        header = (
            f"{'speaker':<7} {'cond':<10} {'n':>4} "
            f"{'PER_mean':>9} {'PER_p50':>8} {'align':>7}"
        )
    print(header)
    print("-" * len(header))
    for (speaker, cond), s in sorted(by_key.items()):
        per_mean = float(np.mean(s.per_values)) if s.per_values else float("nan")
        per_med = _percentile(s.per_values, 50)
        align = float(np.mean(s.align_values)) if s.align_values else float("nan")
        if with_whisper:
            wer_mean = (
                float(np.mean(s.whisper_wers)) if s.whisper_wers else float("nan")
            )
            print(
                f"{speaker:<7} {cond:<10} {s.n:>4} "
                f"{per_mean:>9.3f} {per_med:>8.3f} {align:>7.2f}  {wer_mean:>9.3f}"
            )
        else:
            print(
                f"{speaker:<7} {cond:<10} {s.n:>4} "
                f"{per_mean:>9.3f} {per_med:>8.3f} {align:>7.2f}"
            )


# ============================================================ entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phonemer against TORGO.")
    parser.add_argument(
        "--speakers",
        default=None,
        help="Comma-separated speaker ids (e.g. FC01,F01). Default: all.",
    )
    parser.add_argument(
        "--condition",
        choices=("control", "dysarthria"),
        default=None,
        help="Filter to one condition.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max clips per speaker (deterministic by id order). 0 = all.",
    )
    parser.add_argument(
        "--with-whisper",
        action="store_true",
        help="Also run faster-whisper as a baseline. Slower (~0.5s per clip on CPU).",
    )
    parser.add_argument(
        "--include-non-word",
        action="store_true",
        help="Don't skip prompt_set='non-word' clips (DDK stimuli).",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Print per-clip metrics in addition to the summary.",
    )
    parser.add_argument("--db", default=settings.db_path)
    args = parser.parse_args()

    extractor = PhonemeExtractor(settings.phoneme_model_name, device=settings.device)
    token_to_id = extractor.processor.tokenizer.get_vocab()
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    blank_id = extractor.processor.tokenizer.pad_token_id or 0

    transcriber = None
    if args.with_whisper:
        from app.stt import Transcriber

        transcriber = Transcriber(
            model_size=settings.stt_model_size,
            compute_type=settings.stt_compute_type,
            device=settings.device,
        )

    conn = connect(args.db)
    speakers: list[str | None]
    if args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        speakers = [None]

    all_rows: list[ClipResult] = []
    started = time.monotonic()
    for spk in speakers:
        clips = list_torgo_clips(
            conn,
            speaker_id=spk,
            condition=args.condition,
            limit=args.limit if args.limit else 1_000_000,
        )
        if not args.include_non_word:
            clips = [c for c in clips if c.get("prompt_set") != "non-word"]
        if not clips:
            print(f"[skip] {spk or 'all'}: no clips after filters")
            continue

        per_clip_rows: list[ClipResult] = []
        for clip in clips:
            r = evaluate_clip(
                extractor, transcriber, clip, id_to_token, token_to_id, blank_id
            )
            if r is not None:
                per_clip_rows.append(r)

        if args.per_clip:
            _print_per_clip(per_clip_rows, args.with_whisper)
            print()
        all_rows.extend(per_clip_rows)

    if not all_rows:
        print("no clips evaluated")
        return 1

    _print_summary(all_rows, args.with_whisper)
    elapsed = time.monotonic() - started
    print(f"\n{len(all_rows)} clips in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
