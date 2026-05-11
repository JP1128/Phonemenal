"""Speaker identification via SpeechBrain ECAPA-TDNN embeddings."""
from __future__ import annotations

import io
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class SpeakerMatch:
    label: str           # "patient" or "speaker_<n>"
    role: str | None     # "patient" / "medical_professional" / "family_member" / None
    similarity: float    # cosine similarity to the chosen centroid
    is_new: bool         # True when a new in-session speaker was registered
    rejected: bool = False  # True when the audio likely isn't human speech


class SpeakerIdentifier:
    """Embeds short audio clips, classifies vs. an in-DB speaker registry.

    The patient embedding is loaded once from `speakers` (label="patient").
    Other speakers are auto-labeled `speaker_A`, `speaker_B`, ... as they
    appear; their centroids are updated in place via running mean.
    """

    def __init__(
        self,
        model_source: str,
        savedir: str,
        device: str,
        match_threshold: float,
        reject_threshold: float = 0.15,
    ) -> None:
        from speechbrain.inference.speaker import EncoderClassifier  # heavy import

        Path(savedir).mkdir(parents=True, exist_ok=True)
        self._model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": device},
        )
        self._device = device
        self._threshold = match_threshold
        self._reject_threshold = reject_threshold
        self._lock = threading.Lock()  # speechbrain forward isn't thread-safe

    # ------------------------------------------------------------- inference

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(192, dtype=np.float32)
        with self._lock, torch.inference_mode():
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            wav = wav.to(self._device)
            emb = self._model.encode_batch(wav).squeeze().detach().cpu().numpy()
        return emb.astype(np.float32, copy=False)

    # ------------------------------------------------------------- registry

    def classify(
        self,
        conn: sqlite3.Connection,
        session_id: int,
        embedding: np.ndarray,
    ) -> SpeakerMatch:
        registered = list(_load_session_speakers(conn, session_id))
        # Always try the global "patient" embedding regardless of session.
        patient = _load_patient(conn)
        if patient is not None:
            registered.insert(0, patient)

        if not registered:
            # Fresh session, no enrolled patient — we can't reject without a
            # reference to compare against, so treat this as a new speaker.
            new_label = "speaker_A"
            _insert_speaker(conn, session_id, new_label, "unknown", embedding)
            return SpeakerMatch(new_label, None, 0.0, is_new=True)

        sims = [(_cosine(embedding, e), label, role) for (label, role, e) in registered]
        sims.sort(key=lambda t: t[0], reverse=True)
        best_sim, best_label, best_role = sims[0]

        # Hard floor: cosine sim to *every* enrolled voice below threshold →
        # reject as non-speech. Real human voices land in a different region
        # of the embedding space than keyboard / mouse / paper transients.
        if best_sim < self._reject_threshold:
            return SpeakerMatch("rejected", None, float(best_sim), is_new=False, rejected=True)

        if best_sim >= self._threshold:
            if best_label != "patient":
                _update_centroid(conn, session_id, best_label, embedding)
            return SpeakerMatch(best_label, best_role, float(best_sim), is_new=False)

        # No strong match but above the rejection floor → new in-session speaker.
        existing_letters = {
            label.split("_", 1)[1]
            for (label, _, _) in registered
            if label.startswith("speaker_")
        }
        next_letter = _next_letter(existing_letters)
        new_label = f"speaker_{next_letter}"
        _insert_speaker(conn, session_id, new_label, "unknown", embedding)
        return SpeakerMatch(new_label, None, float(best_sim), is_new=True)

    def enroll_patient(self, conn: sqlite3.Connection, embedding: np.ndarray) -> None:
        _upsert_patient(conn, embedding)


# ============================================================== persistence


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            label TEXT NOT NULL,
            role TEXT,
            embedding BLOB NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
            UNIQUE (session_id, label)
        );
        """
    )


def _load_session_speakers(
    conn: sqlite3.Connection, session_id: int
) -> Iterable[tuple[str, str | None, np.ndarray]]:
    cur = conn.execute(
        "SELECT label, role, embedding FROM speakers"
        " WHERE session_id = ? AND label != 'patient'",
        (session_id,),
    )
    for label, role, blob in cur.fetchall():
        yield label, role, _decode(blob)


def _load_patient(conn: sqlite3.Connection) -> tuple[str, str, np.ndarray] | None:
    row = conn.execute(
        "SELECT embedding FROM speakers WHERE session_id IS NULL AND label = 'patient'"
        " ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return "patient", "patient", _decode(row[0])


def _insert_speaker(
    conn: sqlite3.Connection,
    session_id: int,
    label: str,
    role: str,
    embedding: np.ndarray,
) -> None:
    conn.execute(
        "INSERT INTO speakers (session_id, label, role, embedding) VALUES (?, ?, ?, ?)",
        (session_id, label, role, _encode(embedding)),
    )
    conn.commit()


def _update_centroid(
    conn: sqlite3.Connection,
    session_id: int,
    label: str,
    embedding: np.ndarray,
) -> None:
    row = conn.execute(
        "SELECT embedding, sample_count FROM speakers"
        " WHERE session_id = ? AND label = ?",
        (session_id, label),
    ).fetchone()
    if row is None:
        return
    centroid = _decode(row[0])
    n = int(row[1])
    new_centroid = (centroid * n + embedding) / (n + 1)
    conn.execute(
        "UPDATE speakers SET embedding = ?, sample_count = sample_count + 1"
        " WHERE session_id = ? AND label = ?",
        (_encode(new_centroid), session_id, label),
    )
    conn.commit()


def _upsert_patient(conn: sqlite3.Connection, embedding: np.ndarray) -> None:
    existing = conn.execute(
        "SELECT id FROM speakers WHERE session_id IS NULL AND label = 'patient'"
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO speakers (session_id, label, role, embedding)"
            " VALUES (NULL, 'patient', 'patient', ?)",
            (_encode(embedding),),
        )
    else:
        conn.execute(
            "UPDATE speakers SET embedding = ?, sample_count = sample_count + 1"
            " WHERE id = ?",
            (_encode(embedding), existing[0]),
        )
    conn.commit()


# ============================================================== utilities


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _encode(emb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, emb.astype(np.float32, copy=False), allow_pickle=False)
    return buf.getvalue()


def _decode(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False).astype(np.float32, copy=False)


def _next_letter(existing: set[str]) -> str:
    # Rolls A..Z, then AA..ZZ — plenty for any session.
    import string

    for ch in string.ascii_uppercase:
        if ch not in existing:
            return ch
    for c1 in string.ascii_uppercase:
        for c2 in string.ascii_uppercase:
            if (c1 + c2) not in existing:
                return c1 + c2
    return "X"
