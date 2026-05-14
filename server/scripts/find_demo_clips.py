"""Rank TIMIT clips for the /how-it-works presentation slides.

The Phase-1 cards on that page need clips where:
  * vanilla Whisper produces a *visibly* wrong transcript (high WER)
  * the per-patient adapter produces a *visibly* better one (low WER)
  * the contrast reads cleanly to a non-technical audience

This script iterates the TIMIT-prompt eval-split clips for each demo
speaker (FC02, F03, M01), runs both transcribers on every clip, and
ranks them by a "demonstration score" that rewards (a) a big absolute
WER drop, (b) keeping the adapted output recognizable (≤50% WER), and
(c) a vanilla output that's plausibly nonsensical so the audience can
see the model going off the rails.

Vanilla uses faster-whisper at the same base (``base.en``) the live
``/demo/whisper`` endpoint serves — that's what the page actually
shows. Adapted uses HF whisper-small.en + the speaker's LoRA, matching
the live websocket path. Outputs both are byte-for-byte what the page
will display.

Usage:
    cd server
    .venv/bin/python -m scripts.find_demo_clips
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402
from scripts.torgo_eval import levenshtein  # noqa: E402

TARGET_SR = 16000

# Match the page's vanilla + adapted setup so rankings predict what the
# audience will actually see.
VANILLA_MODEL = "base.en"      # faster-whisper, same as /demo/whisper
ADAPTED_BASE = "openai/whisper-small.en"

ADAPTERS = {
    "FC02": "data/adapters/FC02_small_hicap",
    "F03": "data/adapters/F03_small_hicap",
    "M01": "data/adapters/M01_small_hicap",
}


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def normalize_words(text: str) -> list[str]:
    """Whisper's official EnglishTextNormalizer, with stray-punctuation tokens
    filtered. The normalizer leaves a literal ``.`` for ``...`` ellipses,
    which would otherwise count as a real word edit and inflate WER.
    """
    import re
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    n = EnglishTextNormalizer({})
    return [
        w for w in n(text or "").strip().split()
        # Drop tokens that are nothing but punctuation (e.g. "." from "...").
        if w and re.search(r"[a-z0-9]", w)
    ]


def wer(ref: str, hyp: str) -> float:
    r = normalize_words(ref)
    h = normalize_words(hyp)
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


class VanillaTranscriber:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(VANILLA_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


class AdaptedTranscriber:
    def __init__(self, adapter_dir: str, device: str = "cpu"):
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(ADAPTED_BASE)
        base = WhisperForConditionalGeneration.from_pretrained(ADAPTED_BASE)
        base.config.forced_decoder_ids = None
        base.config.suppress_tokens = []
        self.model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        feats = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)
        out = self.model.generate(feats, max_new_tokens=128, num_beams=1, do_sample=False)
        return self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def demo_score(vanilla_wer: float, adapted_wer: float, vanilla_text: str, adapted_text: str) -> float:
    """Higher = better demo material.

    Reward big improvement, penalize adapter that's still terrible (so the
    "after" looks like a win), prefer vanilla outputs that are clearly off
    semantically (different word counts) over near-miss substitutions.
    """
    improvement = vanilla_wer - adapted_wer
    base = improvement * 2.0          # main signal
    base -= max(0.0, adapted_wer - 0.5) * 1.5  # penalize adapter still bad
    base += min(0.3, vanilla_wer - 0.5) * 0.5  # bonus for very bad vanilla
    # Bonus when vanilla is visibly nonsensical (longer or much shorter than ref).
    nv = len(normalize_words(vanilla_text))
    na = len(normalize_words(adapted_text))
    if nv > 0 and na > 0:
        # A vanilla output 50% longer or 50% shorter than adapted reads as
        # "the model went off the rails" — good for a slide.
        ratio = nv / max(1, na)
        if ratio > 1.4 or ratio < 0.6:
            base += 0.2
    return base


def main() -> int:
    conn = connect(settings.db_path)
    print("loading vanilla (faster-whisper base.en)…", flush=True)
    vanilla = VanillaTranscriber()

    for speaker in ["FC02", "F03", "M01"]:
        print(f"\n{'=' * 78}\n{speaker}: loading adapter from {ADAPTERS[speaker]}\n{'=' * 78}", flush=True)
        try:
            adapted = AdaptedTranscriber(ADAPTERS[speaker])
        except Exception as e:
            print(f"  (no adapter: {e}); skipping")
            continue

        clips = [
            c for c in filter_by_split(
                list_torgo_clips(conn, speaker_id=speaker, limit=1_000_000), "eval"
            )
            if c.get("prompt_set") == "timit"
            and (c.get("intended_text") or "").strip()
        ]
        # Limit to 30 clips per speaker for runtime — these eval splits are
        # ~25-30 TIMIT clips each, so we typically run them all.
        clips = clips[:30]

        rows = []
        for i, clip in enumerate(clips, 1):
            audio = load_audio(clip["wav_path"])
            v = vanilla.transcribe(audio)
            a = adapted.transcribe(audio)
            ref = clip["intended_text"]
            vw = wer(ref, v)
            aw = wer(ref, a)
            score = demo_score(vw, aw, v, a)
            rows.append({
                "clip_id": clip["id"],
                "intended": ref,
                "vanilla": v,
                "adapted": a,
                "vanilla_wer": vw,
                "adapted_wer": aw,
                "score": score,
            })
            print(
                f"  [{i:2d}/{len(clips)}] "
                f"V={vw:.2f} A={aw:.2f} score={score:+.2f}  "
                f"intended={ref[:50]!r}",
                flush=True,
            )

        rows.sort(key=lambda r: r["score"], reverse=True)
        print(f"\n  TOP 5 demo candidates for {speaker}:")
        for j, r in enumerate(rows[:5], 1):
            print(f"\n  #{j}  clip={r['clip_id']}  score={r['score']:+.2f}  "
                  f"vanilla WER {r['vanilla_wer']*100:.0f}% → adapted {r['adapted_wer']*100:.0f}%")
            print(f"      intended: {r['intended']!r}")
            print(f"      vanilla : {r['vanilla']!r}")
            print(f"      adapted : {r['adapted']!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
