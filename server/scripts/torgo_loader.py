"""Walk a TORGO download and populate the torgo_manifest table.

Two layouts are supported:

  1. Official tarball layout (with prompts):
        server/data/torgo/F/F01/Session1/wav_arrayMic/0001.wav
        server/data/torgo/F/F01/Session1/prompts/0001.txt

  2. Flat Kaggle audio-only layout (no prompts):
        server/data/torgo/archive/F_Con/wav_arrayMic_FC01S01/wav_arrayMic_FC01S01_0001.wav
        server/data/torgo/archive/F_Dys/wav_arrayMic_F01/wav_arrayMic_F01_0001.wav

Where the layout omits prompts, ``intended_text`` is left NULL — those clips
can still be streamed through the live pipeline (Phase 9 demo) but cannot be
used for calibration or for ground-truth WER evaluation.

Usage:
    cd server
    python -m scripts.torgo_loader --root data/torgo
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, upsert_torgo_clip  # noqa: E402

logger = logging.getLogger("torgo_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


CONTROL_PREFIXES = ("FC", "MC")  # control speakers in TORGO

# Speaker id appearing as a path component (`F01`, `FC02`, …).
SPEAKER_ID_RE = re.compile(r"^(?:F|FC|M|MC)\d{2}$")
# Speaker id embedded in a flat-layout directory name like
# ``wav_arrayMic_FC01S01`` or ``wav_headMic_M03``.
FLAT_DIR_RE = re.compile(r"wav_(?:arrayMic|headMic)_((?:F|FC|M|MC)\d{2})(?:S\d{2})?$")
# Group dirs in the Kaggle archive layout.
GROUP_DIRS_KAGGLE = {
    "F_Con": "control",
    "F_Dys": "dysarthria",
    "M_Con": "control",
    "M_Dys": "dysarthria",
}


def classify_speaker(speaker_id: str) -> str:
    return "control" if speaker_id.upper().startswith(CONTROL_PREFIXES) else "dysarthria"


_DDK_CV_RE = re.compile(r"^([bcdfghjklmnpqrstvwxz][aeiouy])\1{1,}$", re.IGNORECASE)
_DDK_LITERALS = {"pataka", "patakata", "tapataka", "kalaka", "kataka"}


def is_non_word(prompt_text: str) -> bool:
    """Heuristic detector for DDK / non-word phonetic stimuli.

    These are diadochokinetic tasks where the speaker repeats a syllable
    (``papapa``, ``kakaka``, ``pataka``) or a single phoneme (``aaa``, ``iyiyiy``).
    They aren't speech in the linguistic sense, so calibrating a confusion
    matrix on them produces noise — they map to no real word the LLM can
    recover. We err toward false positives: dropping a few real one-syllable
    stimuli is cheaper than poisoning the profile.
    """
    t = (prompt_text or "").strip().lower()
    if not t:
        return False
    compact = "".join(t.split())
    if not compact.isalpha():
        return False
    # Single repeated character — "aaaa", "iiii"
    if len(compact) >= 3 and len(set(compact)) == 1:
        return True
    # CV bigram repeated — "papa", "papapa", "kakaka"
    if _DDK_CV_RE.match(compact):
        return True
    # Hand-listed DDK targets that don't fit a simple regex.
    if compact in _DDK_LITERALS:
        return True
    return False


def infer_prompt_set(prompt_text: str) -> str:
    p = (prompt_text or "").strip()
    if not p:
        return "unknown"
    if p.lower().startswith("[") or "describe" in p.lower():
        return "image"
    if is_non_word(p):
        return "non-word"
    word_count = len(re.findall(r"\w+", p))
    if word_count <= 2:
        return "intelligibility"
    return "timit"


def read_prompt(prompt_path: Path) -> str:
    try:
        text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
    except FileNotFoundError:
        return ""
    text = re.sub(r"^\[\w+\]\s*", "", text)
    return text


def find_prompt_for_wav(wav_path: Path) -> Path | None:
    """Tarball layout: walk up to find a sibling/cousin ``prompts/<stem>.txt``."""
    stem = wav_path.stem
    # In the flat Kaggle layout the stem is `wav_arrayMic_FC01S01_0001`; strip
    # the prefix so we still find any matching prompt by the trailing index.
    short = stem.split("_")[-1]
    for parent in (wav_path.parent, *wav_path.parents):
        for name in (stem, short):
            for sub in ("prompts", "prompts_text"):
                candidate = parent / sub / f"{name}.txt"
                if candidate.exists():
                    return candidate
        if parent.name.lower().startswith("session"):
            break
    return None


def speaker_from_path(wav_path: Path, root: Path) -> str | None:
    """Recognize speaker id from any path style.

    Looks for an explicit ``F01`` / ``FC02`` style component first; otherwise
    falls back to extracting it from a flat ``wav_<mic>_<id>S0N`` directory.
    """
    rel = wav_path.relative_to(root)
    for part in rel.parts:
        if SPEAKER_ID_RE.match(part):
            return part
    for part in rel.parts:
        m = FLAT_DIR_RE.match(part)
        if m:
            return m.group(1)
    return None


def condition_from_path(wav_path: Path, root: Path, speaker_id: str) -> str:
    """Prefer the explicit Kaggle group dir (`F_Con` / `F_Dys`) when present."""
    rel = wav_path.relative_to(root)
    for part in rel.parts:
        if part in GROUP_DIRS_KAGGLE:
            return GROUP_DIRS_KAGGLE[part]
    return classify_speaker(speaker_id)


def session_from_path(wav_path: Path) -> str | None:
    """Extract a session label like ``S01`` if present."""
    for part in wav_path.parts:
        m = re.search(r"S(\d{2})$", part)
        if m:
            return f"S{m.group(1)}"
    return None


def iter_clips(root: Path):
    """Yield every wav we want to ingest, preferring array-mic over head-mic."""
    # We dedupe array vs head mic by per-clip key once both directories share
    # a parent. That works for both layouts.
    seen_keys: set[str] = set()
    array_paths: list[Path] = []
    head_paths: list[Path] = []
    for wav in sorted(root.rglob("*.wav")):
        parent_name = wav.parent.name.lower()
        # Tarball layout uses `wav_arrayMic` and `wav_headMic` as siblings.
        if parent_name == "wav_headmic":
            head_paths.append(wav)
            continue
        if parent_name == "wav_arraymic":
            array_paths.append(wav)
            seen_keys.add(_dedup_key(wav))
            continue
        # Flat layout: wav_arrayMic_<spkr>S0N or wav_headMic_<spkr>S0N.
        if parent_name.startswith("wav_headmic_"):
            head_paths.append(wav)
            continue
        if parent_name.startswith("wav_arraymic_"):
            array_paths.append(wav)
            seen_keys.add(_dedup_key(wav))
            continue
        # Anything else: take it as-is.
        yield wav

    for wav in array_paths:
        yield wav
    for wav in head_paths:
        if _dedup_key(wav) not in seen_keys:
            yield wav


def _dedup_key(wav_path: Path) -> str:
    """Key used to recognize that the same clip exists for both mics."""
    parent = wav_path.parent.name.lower()
    norm = parent.replace("wav_arraymic", "wav_xmic").replace("wav_headmic", "wav_xmic")
    return f"{norm}/{wav_path.stem.replace('arrayMic', 'xMic').replace('headMic', 'xMic')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TORGO into torgo_manifest.")
    parser.add_argument("--root", default="data/torgo", help="TORGO dataset root")
    parser.add_argument("--db", default=settings.db_path, help="SQLite path")
    parser.add_argument("--limit", type=int, default=0, help="Max clips (0 = all)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(
            f"TORGO root not found: {root}\n"
            "Download a TORGO mirror and extract it there."
        )

    conn = connect(args.db)
    inserted = 0
    skipped = 0
    no_prompt = 0
    for i, wav_path in enumerate(iter_clips(root)):
        if args.limit and inserted >= args.limit:
            break
        speaker_id = speaker_from_path(wav_path, root)
        if speaker_id is None:
            skipped += 1
            continue

        prompt_path = find_prompt_for_wav(wav_path)
        intended = read_prompt(prompt_path) if prompt_path else ""
        if not intended:
            no_prompt += 1
        prompt_set = infer_prompt_set(intended) if intended else "unknown"
        condition = condition_from_path(wav_path, root, speaker_id)
        session = session_from_path(wav_path)

        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            logger.warning("skipping %s: %s", wav_path, e)
            skipped += 1
            continue

        duration_ms = int(info.duration * 1000)
        upsert_torgo_clip(
            conn,
            speaker_id=speaker_id,
            condition=condition,
            intelligibility=session,         # repurposed for session label
            prompt_set=prompt_set,
            intended_text=intended or None,
            wav_path=str(wav_path),
            duration_ms=duration_ms,
            sample_rate=int(info.samplerate),
        )
        inserted += 1
        if inserted % 500 == 0:
            logger.info("loaded %d clips (skipped %d, no_prompt %d)",
                        inserted, skipped, no_prompt)

    logger.info(
        "done: %d clips inserted, %d skipped, %d without prompts",
        inserted, skipped, no_prompt,
    )


if __name__ == "__main__":
    main()
