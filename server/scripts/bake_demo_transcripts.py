"""Bake real vanilla + adapted transcripts for the /how-it-works DemoSlide.

The DemoSlide on the presentation page (client/app/how-it-works/page.tsx)
uses a `DEMO_EXAMPLES` constant of hand-curated examples per patient so
the page can fake a 1s transcription delay instead of waiting on the
real pipeline (which takes 10-30s per clip). This script:

  1. Resolves the 6 demo examples to actual TORGO clip ids by matching
     ``intended_text``.
  2. Runs vanilla faster-whisper ``base.en`` on each clip (same model
     the live ``/demo/whisper`` endpoint and the production pipeline
     serve).
  3. Runs the per-patient LoRA adapter (whisper-small.en + PEFT) on
     each clip (same path the live websocket pipeline serves).
  4. Computes per-clip WER for both.
  5. Emits a JSON-shaped block (and a TypeScript-shaped block) ready to
     paste over the existing ``DEMO_EXAMPLES`` constant in page.tsx so
     the demo cards show byte-for-byte what the real models would
     output.

Usage:
    cd server
    .venv/bin/python -m scripts.bake_demo_transcripts

The `flagged` boolean still needs to be set by hand — the gate's
decision depends on phoneme alignment which this script doesn't run.
A reasonable starting heuristic: flag adapted outputs whose WER > 0.5.
The script will suggest this in the printed output.

To add or change examples, edit ``DEMO_PHRASES`` at the top of this
file. Use the *exact* intended_text from the TORGO DB (case-sensitive,
no trailing period — match what's in ``torgo.intended_text``).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from scripts.torgo_eval import levenshtein  # noqa: E402

TARGET_SR = 16000

# Same models as the live pipeline so this script's output matches what
# the audience would see if they waited on the real run.
VANILLA_MODEL = "base.en"            # faster-whisper
ADAPTED_BASE = "openai/whisper-small.en"

ADAPTERS = {
    "FC02": "data/adapters/FC02_small_hicap",
    "F03": "data/adapters/F03_small_hicap",
    "M01": "data/adapters/M01_small_hicap",
}

# The 6 demo examples the DemoSlide displays — keep in sync with
# ``DEMO_EXAMPLES`` in client/app/how-it-works/page.tsx. Match the
# ``intended_text`` column verbatim (TORGO stores it without trailing
# punctuation).
DEMO_PHRASES: dict[str, list[str]] = {
    "FC02": [
        "We gathered shells on the beach",
        "I looked up and noticed two old men",
    ],
    "F03": [
        "The fish swam across the pond",
        "A roll of wire lay near the wall",
    ],
    "M01": [
        "The islands are sparsely populated",
        "Just one side got wet",
        "Both figures would go higher in later years",
        "If you are losing water replace it immediately",
        "Night after night they received annoying phone calls",
        "I just try to do my best",
    ],
}


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa  # type: ignore[import-untyped]
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def normalize_words(text: str) -> list[str]:
    """Whisper's official EnglishTextNormalizer with stray-punctuation
    tokens dropped — matches the WER used in scripts/find_demo_clips.py
    and on the page so reported numbers agree.
    """
    from transformers.models.whisper.english_normalizer import (
        EnglishTextNormalizer,
    )
    n = EnglishTextNormalizer({})
    return [
        w for w in n(text or "").strip().split()
        if w and re.search(r"[a-z0-9]", w)
    ]


def wer(ref: str, hyp: str) -> float:
    r = normalize_words(ref)
    h = normalize_words(hyp)
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


class VanillaTranscriber:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel
        self.model = WhisperModel(VANILLA_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False
        )
        return " ".join(s.text.strip() for s in segments).strip()


class AdaptedTranscriber:
    def __init__(self, adapter_dir: str, device: str = "cpu") -> None:
        from peft import PeftModel
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
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
        out = self.model.generate(
            feats, max_new_tokens=128, num_beams=1, do_sample=False
        )
        return self.processor.tokenizer.batch_decode(
            out, skip_special_tokens=True
        )[0].strip()


def resolve_clip(conn, speaker: str, phrase: str) -> dict | None:
    """Find the TORGO clip whose intended_text matches ``phrase`` for
    this speaker. Returns the clip row dict or None. We prefer
    prompt_set='timit' since the demo phrases are TIMIT sentences."""
    target = phrase.strip().lower()
    candidates = list_torgo_clips(conn, speaker_id=speaker, limit=1_000_000)
    exact = [
        c for c in candidates
        if (c.get("intended_text") or "").strip().lower() == target
    ]
    if exact:
        # Prefer the TIMIT version if there are dupes.
        timit = [c for c in exact if c.get("prompt_set") == "timit"]
        return (timit or exact)[0]
    return None


def main() -> int:
    conn = connect(settings.db_path)

    # Pass 1 — resolve every phrase to a clip BEFORE we load models, so
    # we fail fast if a phrase is misspelled.
    targets: list[tuple[str, str, dict]] = []
    for speaker, phrases in DEMO_PHRASES.items():
        for phrase in phrases:
            clip = resolve_clip(conn, speaker, phrase)
            if not clip:
                print(
                    f"  ✗ {speaker}: no clip matches {phrase!r} — fix DEMO_PHRASES",
                    flush=True,
                )
                return 1
            targets.append((speaker, phrase, clip))
            print(
                f"  ✓ {speaker} clip={clip['id']} prompt_set={clip.get('prompt_set')}: {phrase!r}",
                flush=True,
            )

    # Pass 2 — load vanilla once, run on every target.
    print("\nloading vanilla (faster-whisper base.en)…", flush=True)
    vanilla = VanillaTranscriber()
    vanilla_outputs: dict[int, str] = {}
    for _, _, clip in targets:
        audio = load_audio(clip["wav_path"])
        vanilla_outputs[clip["id"]] = vanilla.transcribe(audio)
        print(f"  vanilla {clip['id']:>6}: {vanilla_outputs[clip['id']]!r}", flush=True)

    # Pass 3 — load each speaker's adapter exactly once, run on its targets.
    adapted_outputs: dict[int, str] = {}
    by_speaker: dict[str, list[tuple[str, dict]]] = {}
    for speaker, phrase, clip in targets:
        by_speaker.setdefault(speaker, []).append((phrase, clip))

    for speaker, clip_list in by_speaker.items():
        adapter_dir = ADAPTERS.get(speaker)
        if not adapter_dir:
            print(f"  skip adapted for {speaker}: no adapter configured")
            continue
        print(f"\nloading adapter for {speaker} from {adapter_dir}…", flush=True)
        try:
            adapted = AdaptedTranscriber(adapter_dir)
        except Exception as e:
            print(f"  failed to load adapter for {speaker}: {e}")
            continue
        for _, clip in clip_list:
            audio = load_audio(clip["wav_path"])
            adapted_outputs[clip["id"]] = adapted.transcribe(audio)
            print(
                f"  adapted {clip['id']:>6}: {adapted_outputs[clip['id']]!r}",
                flush=True,
            )

    # Pass 4 — assemble the output blob and emit copy-pasteable TS.
    out_by_speaker: dict[str, list[dict]] = {}
    print("\n" + "=" * 78)
    print("Paste this over `DEMO_EXAMPLES` in client/app/how-it-works/page.tsx")
    print("=" * 78)
    print("const DEMO_EXAMPLES: Record<string, DemoExample[]> = {")
    for speaker, clip_list in by_speaker.items():
        entries: list[dict] = []
        print(f"  {speaker}: [")
        for phrase, clip in clip_list:
            v = vanilla_outputs.get(clip["id"], "")
            a = adapted_outputs.get(clip["id"], "")
            vw = wer(phrase, v)
            aw = wer(phrase, a)
            # Heuristic flag: gate would flag an adapter output whose
            # WER is still high (≥0.5). The real gate decision uses
            # phoneme alignment, not WER, but this is a sensible
            # baseline; tighten the value by hand if needed.
            flagged = aw >= 0.5
            entries.append(
                {
                    "clip_id": clip["id"],
                    "intended": phrase,
                    "vanilla": v,
                    "adapted": a,
                    "vanilla_wer": vw,
                    "adapted_wer": aw,
                    "flagged_heuristic": flagged,
                }
            )
            print("    {")
            print(f"      intended: {json.dumps(phrase)},")
            print(f"      vanilla: {json.dumps(v)},")
            print(f"      adapted: {json.dumps(a)},")
            print(f"      flagged: {str(flagged).lower()},")
            print("    },")
        print("  ],")
        out_by_speaker[speaker] = entries
    print("};")

    # Also dump a JSON for diffing / record-keeping.
    out_path = Path("/tmp/demo_transcripts.json")
    out_path.write_text(json.dumps(out_by_speaker, indent=2))
    print(f"\nwrote {out_path}")

    print("\nWER summary (vanilla → adapted, lower is better):")
    for speaker, entries in out_by_speaker.items():
        for e in entries:
            arrow = "→" if e["adapted_wer"] <= e["vanilla_wer"] else "↑"
            print(
                f"  {speaker:>4} clip={e['clip_id']:>6}: "
                f"vanilla {e['vanilla_wer']*100:5.1f}%  {arrow}  "
                f"adapted {e['adapted_wer']*100:5.1f}%  "
                f"flagged={e['flagged_heuristic']}  "
                f"({e['intended'][:50]!r})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
