"""Phonemer server package.

Side effects (run at import time, before any heavy module imports):

  1. Locate the espeak-ng dynamic library on macOS so the ``phonemizer``
     backend (used by the wav2vec2 phoneme tokenizer at load time) can
     dlopen it. Homebrew installs to a non-standard prefix and the system
     loader doesn't search there by default.
  2. Load ``.env`` from the project root if present so ``ANTHROPIC_API_KEY``
     and other settings are available before ``app.config`` is read.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_espeak_library() -> None:
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    if sys.platform != "darwin":
        return  # Linux + Windows install to default loader paths

    candidates = [
        "/opt/homebrew/lib/libespeak-ng.dylib",          # Apple Silicon brew
        "/opt/homebrew/lib/libespeak-ng.1.dylib",
        "/usr/local/lib/libespeak-ng.dylib",             # Intel brew
        "/usr/local/lib/libespeak-ng.1.dylib",
    ]
    for path in candidates:
        if Path(path).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = path
            return


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_ensure_espeak_library()
_load_dotenv()
