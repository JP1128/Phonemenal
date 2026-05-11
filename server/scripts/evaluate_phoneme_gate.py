"""Validate the phoneme-alignment gate against Whisper's failure modes.

The probe (probe_phoneme_signal.py) showed phoneme alignment can't
reliably correct Whisper, but it CAN catch Whisper's worst failure
mode: hallucinated long transcriptions for short utterances.

This harness measures whether the gate (alignment score and/or
phoneme density) correctly flags those failures while leaving
Whisper's normal output alone. Concretely, for each held-out clip:

  * compute Whisper's WER vs. the intended text
  * run the gate
  * categorize:
      - true positive  : Whisper WER > 1.0 (over-produced)  AND flagged
      - false negative : Whisper WER > 1.0  AND not flagged
      - true negative  : Whisper WER ≤ 0.5 (mostly right)   AND not flagged
      - false positive : Whisper WER ≤ 0.5                  AND flagged
      - "ambiguous"    : 0.5 < WER ≤ 1.0 — neither clear hallucination
                         nor clean output; reported separately

A useful gate has high TP rate (catches the bad ones) and low FP
rate (doesn't flag clean speech). The medical safety story stands or
falls on these numbers.

Usage:
    cd server
    python -m scripts.evaluate_phoneme_gate --speakers F03,M01 --limit 30
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.phonemes import PhonemeExtractor  # noqa: E402
from app.stt import GatedTranscriber, Transcriber  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402
from scripts.torgo_eval import levenshtein, normalize_words  # noqa: E402


@dataclass
class GateRow:
    speaker_id: str
    intended: str
    whisper: str
    whisper_wer: float
    align: float
    density: float
    flagged: bool
    reason: str | None


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


def _categorize(wer: float, flagged: bool) -> str:
    if wer > 1.0:
        return "TP" if flagged else "FN"
    if wer <= 0.5:
        return "FP" if flagged else "TN"
    return "AMB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", default="F03,M01")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--split", choices=("eval", "cal", "all"), default="eval")
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--min-align",
        type=float,
        default=-1.0,
        help="Flag if mean forced-align log-prob falls below this. "
        "Empirically tuned on F03, M01, FC02 eval splits: -1.0 catches "
        "~58%% of Whisper hallucinations with 0%% false-positive rate "
        "on clean control speech.",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=25.0,
        help="Flag if Whisper's text implies more than N phonemes per "
        "second of audio. Normal speech runs ~10-15.",
    )
    parser.add_argument(
        "--include-non-word",
        action="store_true",
        help="Include DDK / non-word prompts.",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Print every clip's row in addition to the summary.",
    )
    args = parser.parse_args()

    extractor = PhonemeExtractor(settings.phoneme_model_name, device=settings.device)
    token_to_id = extractor.processor.tokenizer.get_vocab()
    blank_id = extractor.processor.tokenizer.pad_token_id or 0

    transcriber = Transcriber(
        model_size=settings.stt_model_size,
        compute_type=settings.stt_compute_type,
        device=settings.device,
    )
    gate = GatedTranscriber(
        transcriber=transcriber,
        extractor=extractor,
        token_to_id=token_to_id,
        blank_id=blank_id,
        sample_rate=settings.sample_rate,
        min_alignment_score=args.min_align,
        max_phoneme_density=args.max_density,
    )

    conn = connect(args.db)
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    rows: list[GateRow] = []
    started = time.monotonic()

    for spk in speakers:
        clips = list_torgo_clips(conn, speaker_id=spk, limit=1_000_000)
        if not args.include_non_word:
            clips = [c for c in clips if c.get("prompt_set") != "non-word"]
        clips = filter_by_split(clips, args.split)
        if args.limit:
            clips = clips[: args.limit]
        if not clips:
            print(f"[skip] {spk}: no clips after filters")
            continue

        for clip in clips:
            text = (clip.get("intended_text") or "").strip()
            if not text:
                continue
            audio = _load_audio(clip["wav_path"])
            if audio is None or audio.size == 0:
                continue

            gated = gate.transcribe_with_gate(audio)
            ref_words = normalize_words(text)
            hyp_words = normalize_words(gated.text)
            wer = levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)

            rows.append(
                GateRow(
                    speaker_id=spk,
                    intended=text,
                    whisper=gated.text,
                    whisper_wer=wer,
                    align=gated.align_score,
                    density=gated.phoneme_density,
                    flagged=gated.flagged,
                    reason=gated.reason,
                )
            )

    if not rows:
        print("no clips evaluated")
        return 1

    if args.per_clip:
        print(
            f"{'spk':<5} {'WER':>5} {'align':>6} {'dens':>5} "
            f"{'cat':>4} {'flag':>5}  intended | whisper"
        )
        print("-" * 80)
        for r in rows:
            cat = _categorize(r.whisper_wer, r.flagged)
            flag_marker = "✗" if r.flagged else " "
            print(
                f"{r.speaker_id:<5} {r.whisper_wer:>5.2f} {r.align:>6.2f} "
                f"{r.density:>5.1f} {cat:>4} {flag_marker:>5}  "
                f"{r.intended[:30]!r} | {r.whisper[:30]!r}"
            )

    # Confusion-matrix summary.
    print()
    print("== gate effectiveness ==")
    print(
        f"{'speaker':<7} {'n':>4} {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4} "
        f"{'AMB':>4}  {'TP_rate':>8} {'FP_rate':>8}"
    )
    print("-" * 60)
    by_spk: dict[str, list[GateRow]] = {}
    for r in rows:
        by_spk.setdefault(r.speaker_id, []).append(r)

    for spk, spk_rows in sorted(by_spk.items()):
        cats = [_categorize(r.whisper_wer, r.flagged) for r in spk_rows]
        n_tp = cats.count("TP")
        n_fn = cats.count("FN")
        n_tn = cats.count("TN")
        n_fp = cats.count("FP")
        n_amb = cats.count("AMB")
        n = len(spk_rows)
        tp_rate = n_tp / max(n_tp + n_fn, 1)
        fp_rate = n_fp / max(n_tn + n_fp, 1)
        print(
            f"{spk:<7} {n:>4} {n_tp:>4} {n_fn:>4} {n_tn:>4} {n_fp:>4} "
            f"{n_amb:>4}  {tp_rate:>8.2%} {fp_rate:>8.2%}"
        )

    # Aggregate.
    cats = [_categorize(r.whisper_wer, r.flagged) for r in rows]
    n_tp, n_fn = cats.count("TP"), cats.count("FN")
    n_tn, n_fp = cats.count("TN"), cats.count("FP")
    n_amb = cats.count("AMB")
    tp_rate = n_tp / max(n_tp + n_fn, 1)
    fp_rate = n_fp / max(n_tn + n_fp, 1)
    print(
        f"{'ALL':<7} {len(rows):>4} {n_tp:>4} {n_fn:>4} {n_tn:>4} {n_fp:>4} "
        f"{n_amb:>4}  {tp_rate:>8.2%} {fp_rate:>8.2%}"
    )

    elapsed = time.monotonic() - started
    print(f"\n{len(rows)} clips in {elapsed:.1f}s "
          f"(thresholds: align<{args.min_align}, density>{args.max_density})")
    print()
    print("interpretation:")
    print("  TP_rate = catches Whisper hallucinations (WER>1.0). High is good.")
    print("  FP_rate = flags Whisper output Whisper got mostly right. Low is good.")
    print("  AMB    = Whisper near-misses (0.5<WER≤1.0). Reported separately —")
    print("           neither clear hallucination nor clear win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
