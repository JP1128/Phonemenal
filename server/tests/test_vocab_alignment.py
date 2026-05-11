"""Phase 2.2 alignment gate.

This test must pass before any distortion-model code is allowed to run.
It mirrors scripts/verify_phoneme_vocab.py but as a pytest-runnable check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.config import settings
from app.vocab import normalize_phonemes


PROBE_SENTENCES = [
    "I need water please.",
    "Can you help me sit up.",
    "My head hurts on the right side.",
    "Thank you doctor.",
    "She had your dark suit in greasy wash water all year.",
]


@pytest.fixture(scope="module")
def model_vocab() -> set[str]:
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(settings.phoneme_model_name)
    return set(tok.get_vocab().keys())


@pytest.fixture(scope="module")
def phonemized() -> list[str]:
    pytest.importorskip("phonemizer")
    from phonemizer import phonemize

    return phonemize(
        PROBE_SENTENCES,
        language="en-us",
        backend="espeak",
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        njobs=1,
    )


def test_every_phonemizer_symbol_maps_into_model_vocab(model_vocab, phonemized):
    unresolved: set[str] = set()
    for sent in phonemized:
        for tok in normalize_phonemes(sent):
            if tok not in model_vocab and not all(c in model_vocab for c in tok):
                unresolved.add(tok)
    assert not unresolved, (
        f"Phonemizer emitted {len(unresolved)} symbols missing from the "
        f"wav2vec2 vocab: {sorted(unresolved)!r}. Run "
        "scripts/verify_phoneme_vocab.py to (re)build phoneme_alias_map.json."
    )
