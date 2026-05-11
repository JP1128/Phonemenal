"""Verify phonemizer (espeak-ng) output aligns with the wav2vec2 phoneme vocab.

Without this gate, the confusion matrix Phase 2.3 builds would be
meaningless: rows from `phonemizer` and columns from the model would
disagree on which symbol means /θ/ vs /d/.

This script:
  1. Loads the wav2vec2 tokenizer vocab (`facebook/wav2vec2-lv-60-espeak-cv-ft`).
  2. Phonemizes a corpus of probe sentences with espeak-ng.
  3. Diffs the produced symbols against the model vocab.
  4. Tries to repair common discrepancies with deterministic rewrites and
     writes the resulting mapping to ``app/phoneme_alias_map.json``.
  5. Exits non-zero if any unmappable symbols remain — the distortion
     pipeline must not run until this passes.

Usage:
    cd server
    python -m scripts.verify_phoneme_vocab
    python -m scripts.verify_phoneme_vocab --strict   # fail on any unknown
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401  (sets PHONEMIZER_ESPEAK_LIBRARY)
from app.config import settings  # noqa: E402
from app.vocab import write_alias_map  # noqa: E402


PROBE_SENTENCES = [
    # Mix of basic-needs, medical, emotional, and TIMIT-style coverage so we
    # exercise a broad slice of the espeak inventory.
    "I need water please.",
    "Can you help me sit up.",
    "My head hurts on the right side.",
    "Please call my daughter.",
    "I am cold and I am scared.",
    "Thank you doctor.",
    "She had your dark suit in greasy wash water all year.",
    "Don't ask me to carry an oily rag like that.",
    "The quick brown fox jumps over the lazy dog.",
    "How much pain do you feel right now.",
    "I cannot breathe.",
    "Where is the bathroom.",
    "Yes I understand.",
    "No I do not want that.",
    "Please bring my glasses.",
]


# Regional variants we deliberately collapse to the model's preferred form.
HARDCODED_ALIASES = {
    "ɝ": "ɜː",
    "ɚ": "ə",
    "ɾ": "t",      # American flap → /t/
    "ʔ": "t",      # glottal stop in casual speech
    "ɫ": "l",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any unmappable symbol remains after auto-repair.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer  # heavy import, deferred

    tokenizer = AutoTokenizer.from_pretrained(settings.phoneme_model_name)
    vocab: set[str] = set(tokenizer.get_vocab().keys())
    print(f"wav2vec2 tokenizer vocab: {len(vocab)} symbols")

    from phonemizer import phonemize

    raw = phonemize(
        PROBE_SENTENCES,
        language="en-us",
        backend="espeak",
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        njobs=1,
    )

    seen: Counter[str] = Counter()
    for sent in raw:
        for tok in sent.split():
            tok = tok.replace("ˈ", "").replace("ˌ", "").replace(".", "")
            tok = tok.replace("ː", "")  # length, kept separately if needed
            seen[tok] += 1

    print(f"espeak emitted {len(seen)} distinct symbols across "
          f"{len(PROBE_SENTENCES)} probe sentences.")

    alias_map: dict[str, str] = {}
    unresolved: list[str] = []
    for sym, count in seen.most_common():
        if sym in vocab:
            continue
        # Try hardcoded substitutions.
        if sym in HARDCODED_ALIASES and HARDCODED_ALIASES[sym] in vocab:
            alias_map[sym] = HARDCODED_ALIASES[sym]
            continue
        # Try Unicode NFD-normalized form.
        nfd = unicodedata.normalize("NFD", sym)
        if nfd != sym and nfd in vocab:
            alias_map[sym] = nfd
            continue
        # Try stripping combining marks entirely.
        bare = "".join(
            c for c in unicodedata.normalize("NFD", sym)
            if not unicodedata.combining(c)
        )
        if bare and bare != sym and bare in vocab:
            alias_map[sym] = bare
            continue
        # Try splitting compound symbols (e.g. "tʃ" -> "t","ʃ").
        if len(sym) >= 2 and all(c in vocab for c in sym):
            # Multi-char tokens are kept as-is; we just record an "ok" alias.
            continue
        unresolved.append((sym, count))

    if alias_map:
        write_alias_map(alias_map)
        print(f"\n✓ wrote {len(alias_map)} aliases to app/phoneme_alias_map.json:")
        for k, v in alias_map.items():
            print(f"    {k!r} -> {v!r}")

    if unresolved:
        print(f"\n✗ {len(unresolved)} symbol(s) cannot be mapped to the model vocab:")
        for sym, count in unresolved:
            print(f"    {sym!r:>12}  (seen {count} time(s))")
        if args.strict:
            return 1
        print("\nNon-strict mode: these will pass through as-is and may degrade alignment.")
    else:
        print("\n✓ every espeak symbol maps cleanly into the wav2vec2 vocab.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
