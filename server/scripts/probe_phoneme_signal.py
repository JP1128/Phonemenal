"""Does the phoneme distribution add signal Whisper missed?

This is the architecture-direction probe. The current Phonemer pipeline
loses to Whisper on every condition; before rebuilding, we want to
know whether the wav2vec2 phoneme distribution contains information
that *would* let a smarter pipeline beat Whisper.

For each held-out clip we compute two alignment scores against the
same audio:

  align(intended)  — forced alignment of the ground-truth transcript
  align(whisper)   — forced alignment of Whisper's top-1 transcript

If the phoneme distribution contains useful signal, then on clips
where Whisper was wrong, ``align(intended) > align(whisper)`` more
often than chance — meaning a re-ranker that scored hypotheses by
phoneme alignment would have fixed Whisper's mistakes.

We also bin by Whisper's WER to see whether the signal scales with
how badly Whisper missed.

Usage:
    cd server
    python -m scripts.probe_phoneme_signal --speakers F03 --limit 30
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
from app.alignment import alignment_score, forced_align, text_to_phonemes  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.phonemes import PhonemeExtractor  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402
from scripts.torgo_eval import levenshtein, normalize_words  # noqa: E402


@dataclass
class ProbeRow:
    speaker_id: str
    intended: str
    whisper: str
    whisper_wer: float
    align_intended: float
    align_whisper: float
    delta: float          # align_intended - align_whisper, > 0 means phoneme prefers truth
    n_phonemes: int


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


def _align_score_for_text(
    text: str,
    log_probs: np.ndarray,
    token_to_id: dict[str, int],
    blank_id: int,
) -> tuple[float, int]:
    """Phonemize ``text`` and return (mean alignment log-prob, phoneme count)."""
    phonemes = text_to_phonemes(text)
    if not phonemes:
        return float("-inf"), 0
    aligned = forced_align(log_probs, phonemes, token_to_id, blank_id)
    return alignment_score(aligned), len(phonemes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", default="F03,M01")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--split",
        choices=("eval", "cal", "all"),
        default="eval",
        help="Which split to probe. Default: held-out eval set.",
    )
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--include-non-word",
        action="store_true",
        help="Include DDK / non-word prompts.",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Print every clip's intended/whisper/delta in addition to the summary.",
    )
    args = parser.parse_args()

    extractor = PhonemeExtractor(settings.phoneme_model_name, device=settings.device)
    token_to_id = extractor.processor.tokenizer.get_vocab()
    blank_id = extractor.processor.tokenizer.pad_token_id or 0

    from app.stt import Transcriber

    transcriber = Transcriber(
        model_size=settings.stt_model_size,
        compute_type=settings.stt_compute_type,
        device=settings.device,
    )

    conn = connect(args.db)
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    rows: list[ProbeRow] = []
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

            # Phoneme distributions used by both alignments.
            probs = extractor.extract(audio, settings.sample_rate)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))

            whisper_text = transcriber.transcribe(audio).strip()

            align_int, n_ph = _align_score_for_text(
                text, log_probs, token_to_id, blank_id
            )
            align_whp, _ = _align_score_for_text(
                whisper_text or text, log_probs, token_to_id, blank_id
            )

            ref_words = normalize_words(text)
            hyp_words = normalize_words(whisper_text)
            wer = levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)

            rows.append(
                ProbeRow(
                    speaker_id=spk,
                    intended=text,
                    whisper=whisper_text,
                    whisper_wer=wer,
                    align_intended=align_int,
                    align_whisper=align_whp,
                    delta=align_int - align_whp,
                    n_phonemes=n_ph,
                )
            )

    if not rows:
        print("no clips evaluated")
        return 1

    if args.per_clip:
        print(
            f"{'spk':<5} {'WER':>5} {'a(int)':>7} {'a(whp)':>7} {'Δ':>7}  intended | whisper"
        )
        print("-" * 72)
        for r in rows:
            print(
                f"{r.speaker_id:<5} {r.whisper_wer:>5.2f} "
                f"{r.align_intended:>7.2f} {r.align_whisper:>7.2f} "
                f"{r.delta:>+7.2f}  {r.intended[:30]!r} | {r.whisper[:30]!r}"
            )

    print()
    print("== summary ==")
    print(f"{'speaker':<7} {'n':>4} {'WER_mean':>9} {'n_wrong':>8} "
          f"{'Δ>0':>5} {'Δ>0|wrong':>11} {'Δ_mean':>8}")
    print("-" * 65)
    by_spk: dict[str, list[ProbeRow]] = {}
    for r in rows:
        by_spk.setdefault(r.speaker_id, []).append(r)

    for spk, spk_rows in sorted(by_spk.items()):
        n = len(spk_rows)
        wer_mean = float(np.mean([r.whisper_wer for r in spk_rows]))
        wrong = [r for r in spk_rows if r.whisper_wer > 0.0]
        delta_pos = sum(1 for r in spk_rows if r.delta > 0)
        delta_pos_wrong = sum(1 for r in wrong if r.delta > 0)
        delta_mean = float(np.mean([r.delta for r in spk_rows]))
        print(
            f"{spk:<7} {n:>4} {wer_mean:>9.3f} {len(wrong):>8} "
            f"{delta_pos:>5} {delta_pos_wrong:>3}/{len(wrong):<7} {delta_mean:>+8.2f}"
        )

    elapsed = time.monotonic() - started
    print(f"\n{len(rows)} clips in {elapsed:.1f}s")

    print()
    print("interpretation:")
    print("  Δ > 0 means the phoneme distribution prefers the intended")
    print("  text over Whisper's transcription. If 'Δ>0|wrong' is much")
    print("  greater than 50% of n_wrong, a phoneme-aware re-ranker")
    print("  could fix Whisper's mistakes — i.e. the hybrid architecture")
    print("  has signal and is worth building.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
