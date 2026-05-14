"""Compute per-patient threshold calibration from a trained adapter.

Runs the full gated-transcribe pipeline against the patient's calibration
split, collects per-clip and per-word distributions of the gate's signals
(alignment score, voiced-duration phoneme density, per-word phoneme
confidence, overall confidence), and writes percentile-based thresholds
to ``{adapter_dir}/calibration.json``.

The live runtime's ``GatedTranscriber`` loads this file when an adapter
is active and uses patient-specific thresholds instead of the global
defaults. Without it the gate uses ``settings.gate_*`` for everyone,
which means severely-dysarthric speakers (whose normal align scores
sit around -1.5) get over-flagged while controls get under-flagged.

Usage:
    cd server
    .venv/bin/python -m scripts.compute_calibration \\
      --speaker-id M01 --adapter-dir data/adapters/M01_small_hicap
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.phonemes import PhonemeExtractor  # noqa: E402
from app.stt import AdaptedTranscriber, GatedTranscriber  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402

TARGET_SR = 16000


def _load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def compute_calibration(
    speaker_id: str,
    adapter_dir: Path,
    base_model: str | None = None,
    max_clips: int = 50,
    out_path: Path | None = None,
) -> dict:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.exists():
        raise SystemExit(f"adapter dir missing: {adapter_dir}")
    if base_model is None:
        marker = adapter_dir / "BASE_MODEL.txt"
        base_model = (
            marker.read_text().strip() if marker.exists() else "openai/whisper-small.en"
        )

    print(f"Loading adapter {adapter_dir} on base {base_model}", flush=True)
    transcriber = AdaptedTranscriber(str(adapter_dir), base_model=base_model)
    extractor = PhonemeExtractor(model_name=settings.phoneme_model_name)
    gate = GatedTranscriber(
        transcriber=transcriber,
        extractor=extractor,
        token_to_id=extractor.processor.tokenizer.get_vocab(),
        blank_id=extractor.processor.tokenizer.pad_token_id or 0,
        sample_rate=TARGET_SR,
        n_best=max(1, int(settings.gate_n_best)),
    )

    conn = connect(settings.db_path)
    all_clips = list_torgo_clips(conn, speaker_id=speaker_id, limit=1_000_000)
    cal_clips = filter_by_split(all_clips, "cal")
    cal_clips = [c for c in cal_clips if (c.get("intended_text") or "").strip()]
    cal_clips = cal_clips[:max_clips]
    if not cal_clips:
        raise SystemExit(f"no cal-split clips for {speaker_id}")
    print(f"calibration set: {len(cal_clips)} clips for {speaker_id}", flush=True)

    align_scores: list[float] = []
    densities: list[float] = []
    overall_confs: list[float] = []
    per_word_confs: list[float] = []
    for i, clip in enumerate(cal_clips, 1):
        try:
            audio = _load_audio(clip["wav_path"])
        except Exception as e:
            print(f"  [{i:2d}] audio load failed: {e}", flush=True)
            continue
        result = gate.transcribe_with_gate(audio)
        if not np.isfinite(result.align_score):
            continue
        align_scores.append(float(result.align_score))
        densities.append(float(result.phoneme_density))
        overall_confs.append(
            float(max(0.0, min(1.0, np.exp(result.align_score))))
        )
        for s in result.per_word_scores:
            if np.isfinite(s):
                per_word_confs.append(float(max(0.0, min(1.0, np.exp(s)))))
        print(
            f"  [{i:2d}/{len(cal_clips)}]  "
            f"align={result.align_score:+.3f}  density={result.phoneme_density:.2f}  "
            f"text={result.text[:50]!r}",
            flush=True,
        )

    if not align_scores:
        raise SystemExit("no finite gate signals — calibration failed")

    def pct(arr, ps):
        return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

    out = {
        "schema_version": 1,
        "speaker_id": speaker_id,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "n_clips_used": len(align_scores),
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "stats": {
            "align_score": pct(align_scores, [10, 25, 50, 75, 90]),
            "phoneme_density_voiced": pct(densities, [10, 25, 50, 75, 90]),
            "overall_conf": pct(overall_confs, [10, 25, 50, 75, 90]),
            "per_word_conf": pct(per_word_confs, [10, 25, 50, 75, 90]),
        },
        # Derived thresholds for runtime. Rationale:
        # - min_alignment_score: p10 of cal align_scores — anything below
        #   the patient's bottom 10% is unusual *for them*.
        # - min_phoneme_density: p10 of voiced-duration densities — same.
        # - ui_low_confidence_threshold: p10 of overall_conf — below this
        #   the listener UI promotes to flagged.
        # - shaded_word_red_below / yellow_below: p10 / p25 of per-word
        #   confidence on the patient's *correctly-transcribed-during-
        #   training* audio. The UI uses these to draw red / yellow on
        #   the right side of the patient's natural distribution, not a
        #   global "30%" that fits no one well.
        "thresholds": {
            "min_alignment_score": float(np.percentile(align_scores, 10)),
            "min_phoneme_density": float(np.percentile(densities, 10)),
            "ui_low_confidence_threshold": float(np.percentile(overall_confs, 10)),
            "shaded_word_red_below": float(np.percentile(per_word_confs, 10)),
            "shaded_word_yellow_below": float(np.percentile(per_word_confs, 25)),
        },
    }

    out_path = out_path or (adapter_dir / "calibration.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print("Derived per-patient thresholds:")
    for k, v in out["thresholds"].items():
        print(f"  {k:32s} {v:+.4f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    compute_calibration(
        speaker_id=args.speaker_id,
        adapter_dir=Path(args.adapter_dir),
        base_model=args.base_model,
        max_clips=args.max_clips,
        out_path=Path(args.out_json) if args.out_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
