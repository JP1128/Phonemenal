"""SQLite persistence for sessions, utterances, phoneme frames, features."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS utterances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    speaker_label TEXT,
    speaker_role TEXT,
    speaker_similarity REAL,
    is_patient INTEGER NOT NULL DEFAULT 0,
    transcript TEXT,
    audio_path TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_utterances_session_time
    ON utterances (session_id, start_ms);

CREATE TABLE IF NOT EXISTS phoneme_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    utterance_id INTEGER,
    timestamp_ms INTEGER NOT NULL,
    distribution_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_phoneme_frames_session_time
    ON phoneme_frames (session_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_phoneme_frames_utterance
    ON phoneme_frames (utterance_id);

CREATE TABLE IF NOT EXISTS utterance_features (
    utterance_id INTEGER PRIMARY KEY,
    features_json TEXT NOT NULL,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE
);

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

CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    completed_at REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS calibration_utterances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id INTEGER NOT NULL,
    sentence_set_id TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,
    intended_text TEXT NOT NULL,
    intended_phonemes TEXT,
    audio_path TEXT,
    duration_ms INTEGER,
    produced_phoneme_distribution_json TEXT,
    alignment_json TEXT,
    accepted INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (calibration_id) REFERENCES calibrations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cal_utt_cal ON calibration_utterances (calibration_id);

CREATE TABLE IF NOT EXISTS patient_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    profile_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (calibration_id) REFERENCES calibrations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_profile_cal ON patient_profiles (calibration_id, version);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    observed_at REAL NOT NULL,
    keyword TEXT NOT NULL,
    weight REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_topics_session ON topics (session_id, observed_at);

CREATE TABLE IF NOT EXISTS pending_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    asked_at REAL NOT NULL,
    text TEXT NOT NULL,
    qtype TEXT NOT NULL,
    answered INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_questions_session ON pending_questions (session_id, asked_at);

CREATE TABLE IF NOT EXISTS emotional_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    observed_at REAL NOT NULL,
    frustration REAL NOT NULL,
    distress REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_emotion_session ON emotional_states (session_id, observed_at);

CREATE TABLE IF NOT EXISTS session_metadata (
    session_id INTEGER PRIMARY KEY,
    setting TEXT,
    listener_role_hint TEXT,
    time_of_day TEXT,
    activity_context TEXT,
    notes TEXT,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    utterance_id INTEGER NOT NULL,
    primary_text TEXT NOT NULL,
    alternatives_json TEXT NOT NULL,
    per_word_confidence_json TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    emergency INTEGER NOT NULL DEFAULT 0,
    context_hints_json TEXT,
    model TEXT NOT NULL,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    feedback TEXT,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_translations_utt ON translations (utterance_id);
CREATE INDEX IF NOT EXISTS idx_translations_feedback ON translations (feedback)
    WHERE feedback IS NOT NULL;

CREATE TABLE IF NOT EXISTS fatigue_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    error_density REAL NOT NULL,
    phase TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fatigue_session ON fatigue_snapshots (session_id, minute);

CREATE TABLE IF NOT EXISTS online_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id INTEGER,             -- nullable: profile-anchored adaptation
    utterance_id INTEGER NOT NULL,
    observed_at REAL NOT NULL,
    intended TEXT NOT NULL,             -- intended phoneme symbol
    produced TEXT NOT NULL,             -- top-1 produced phoneme symbol
    weight REAL NOT NULL DEFAULT 1.0,   -- 1.0 for confirmed, 0.5 for inferred
    source TEXT NOT NULL,               -- 'confirmed' / 'corrected' / 'inferred'
    FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE,
    FOREIGN KEY (calibration_id) REFERENCES calibrations(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_online_obs_cal ON online_observations (calibration_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_online_obs_utt ON online_observations (utterance_id);

CREATE TABLE IF NOT EXISTS personal_vocabulary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,                  -- 'name' / 'medication' / 'place' / 'preferred' / 'other'
    term TEXT NOT NULL,
    notes TEXT,
    boost REAL NOT NULL DEFAULT 1.5,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE (kind, term)
);

CREATE TABLE IF NOT EXISTS intelligibility_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id INTEGER,
    week_start REAL NOT NULL,            -- Unix timestamp of Monday 00:00 UTC
    n_utterances INTEGER NOT NULL,
    diagonal_mass REAL NOT NULL,         -- effective P(produced=intended) average
    confirmed_rate REAL NOT NULL,        -- fraction of translations confirmed (vs corrected)
    avg_overall_confidence REAL NOT NULL,
    UNIQUE (calibration_id, week_start)
);

CREATE TABLE IF NOT EXISTS torgo_manifest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id TEXT NOT NULL,
    condition TEXT,                    -- e.g. "control", "dysarthria"
    intelligibility TEXT,              -- "high"/"mid"/"low" if known
    prompt_set TEXT,                   -- "intelligibility"/"timit"/"image"
    intended_text TEXT,
    intended_phonemes TEXT,            -- filled in by Phase 2.2 alignment
    wav_path TEXT NOT NULL,
    duration_ms INTEGER,
    sample_rate INTEGER,
    UNIQUE (wav_path)
);
CREATE INDEX IF NOT EXISTS idx_torgo_speaker ON torgo_manifest (speaker_id);
CREATE INDEX IF NOT EXISTS idx_torgo_condition ON torgo_manifest (condition);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


# ----------------------------------------------------------------- sessions


def create_session(conn: sqlite3.Connection, notes: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (started_at, notes) VALUES (?, ?)",
        (time.time(), notes),
    )
    conn.commit()
    return int(cur.lastrowid)


def end_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (time.time(), session_id),
    )
    conn.commit()


# ---------------------------------------------------------------- utterances


def create_utterance(
    conn: sqlite3.Connection,
    session_id: int,
    start_ms: int,
    end_ms: int,
    speaker_label: str | None,
    speaker_role: str | None,
    speaker_similarity: float | None,
    is_patient: bool,
    audio_path: str | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO utterances"
        " (session_id, start_ms, end_ms, speaker_label, speaker_role,"
        "  speaker_similarity, is_patient, audio_path)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            start_ms,
            end_ms,
            speaker_label,
            speaker_role,
            speaker_similarity,
            1 if is_patient else 0,
            audio_path,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_utterance_transcript(
    conn: sqlite3.Connection, utterance_id: int, transcript: str
) -> None:
    conn.execute(
        "UPDATE utterances SET transcript = ? WHERE id = ?",
        (transcript, utterance_id),
    )
    conn.commit()


# ----------------------------------------------------------------- phonemes


def write_frames(
    conn: sqlite3.Connection,
    session_id: int,
    utterance_id: int | None,
    frames: Iterable[tuple[int, list[dict]]],
) -> None:
    rows = [
        (session_id, utterance_id, ts, json.dumps(dist)) for ts, dist in frames
    ]
    if not rows:
        return
    conn.executemany(
        "INSERT INTO phoneme_frames"
        " (session_id, utterance_id, timestamp_ms, distribution_json)"
        " VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ----------------------------------------------------------------- features


def write_features(
    conn: sqlite3.Connection, utterance_id: int, features_json: dict
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO utterance_features (utterance_id, features_json)"
        " VALUES (?, ?)",
        (utterance_id, json.dumps(features_json)),
    )
    conn.commit()


# ----------------------------------------------------------------- TORGO


def upsert_torgo_clip(
    conn: sqlite3.Connection,
    speaker_id: str,
    condition: str | None,
    intelligibility: str | None,
    prompt_set: str | None,
    intended_text: str | None,
    wav_path: str,
    duration_ms: int,
    sample_rate: int,
) -> int:
    cur = conn.execute(
        "INSERT INTO torgo_manifest"
        " (speaker_id, condition, intelligibility, prompt_set, intended_text,"
        "  wav_path, duration_ms, sample_rate)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(wav_path) DO UPDATE SET"
        "   speaker_id = excluded.speaker_id,"
        "   condition = excluded.condition,"
        "   intelligibility = excluded.intelligibility,"
        "   prompt_set = excluded.prompt_set,"
        "   intended_text = excluded.intended_text,"
        "   duration_ms = excluded.duration_ms,"
        "   sample_rate = excluded.sample_rate",
        (
            speaker_id,
            condition,
            intelligibility,
            prompt_set,
            intended_text,
            wav_path,
            duration_ms,
            sample_rate,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_torgo_clips(
    conn: sqlite3.Connection,
    speaker_id: str | None = None,
    condition: str | None = None,
    limit: int = 200,
    with_text_only: bool = True,
) -> list[dict]:
    """List manifest clips. ``with_text_only`` (default) skips clips whose
    ``intended_text`` is NULL or empty — those are unusable for calibration,
    forced alignment, and WER eval. The legacy audio-only Kaggle archive
    layout populates many such rows; callers almost always want them filtered."""
    sql = (
        "SELECT id, speaker_id, condition, intelligibility, prompt_set,"
        " intended_text, wav_path, duration_ms FROM torgo_manifest"
    )
    where = []
    params: list = []
    if speaker_id:
        where.append("speaker_id = ?")
        params.append(speaker_id)
    if condition:
        where.append("condition = ?")
        params.append(condition)
    if with_text_only:
        where.append("intended_text IS NOT NULL AND TRIM(intended_text) != ''")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY speaker_id, id LIMIT ?"
    params.append(int(limit))
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# --------------------------------------------------------------- calibration


def create_calibration(conn: sqlite3.Connection, notes: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO calibrations (started_at, notes) VALUES (?, ?)",
        (time.time(), notes),
    )
    conn.commit()
    return int(cur.lastrowid)


def complete_calibration(conn: sqlite3.Connection, calibration_id: int) -> None:
    conn.execute(
        "UPDATE calibrations SET completed_at = ? WHERE id = ?",
        (time.time(), calibration_id),
    )
    conn.commit()


def add_calibration_utterance(
    conn: sqlite3.Connection,
    calibration_id: int,
    sentence_set_id: str,
    sentence_index: int,
    intended_text: str,
    intended_phonemes: str | None,
    audio_path: str | None,
    duration_ms: int | None,
    produced_phoneme_distribution_json: str | None,
    alignment_json: str | None = None,
    accepted: bool = True,
) -> int:
    cur = conn.execute(
        "INSERT INTO calibration_utterances"
        " (calibration_id, sentence_set_id, sentence_index, intended_text,"
        "  intended_phonemes, audio_path, duration_ms,"
        "  produced_phoneme_distribution_json, alignment_json, accepted)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            calibration_id,
            sentence_set_id,
            sentence_index,
            intended_text,
            intended_phonemes,
            audio_path,
            duration_ms,
            produced_phoneme_distribution_json,
            alignment_json,
            1 if accepted else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_calibration_utterances(
    conn: sqlite3.Connection, calibration_id: int
) -> list[dict]:
    cur = conn.execute(
        "SELECT id, sentence_set_id, sentence_index, intended_text,"
        " intended_phonemes, audio_path, duration_ms,"
        " produced_phoneme_distribution_json, alignment_json, accepted"
        " FROM calibration_utterances WHERE calibration_id = ?"
        " ORDER BY id",
        (calibration_id,),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def update_calibration_utterance_alignment(
    conn: sqlite3.Connection,
    calibration_utterance_id: int,
    alignment_json: str,
) -> None:
    conn.execute(
        "UPDATE calibration_utterances SET alignment_json = ? WHERE id = ?",
        (alignment_json, calibration_utterance_id),
    )
    conn.commit()


def write_patient_profile(
    conn: sqlite3.Connection,
    calibration_id: int,
    profile_json: str,
) -> int:
    cur = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM patient_profiles WHERE calibration_id = ?",
        (calibration_id,),
    )
    next_version = int(cur.fetchone()[0]) + 1
    cur = conn.execute(
        "INSERT INTO patient_profiles (calibration_id, version, profile_json)"
        " VALUES (?, ?, ?)",
        (calibration_id, next_version, profile_json),
    )
    conn.commit()
    return int(cur.lastrowid)


def latest_patient_profile(
    conn: sqlite3.Connection, calibration_id: int
) -> dict | None:
    cur = conn.execute(
        "SELECT id, version, profile_json, created_at FROM patient_profiles"
        " WHERE calibration_id = ? ORDER BY version DESC LIMIT 1",
        (calibration_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([c[0] for c in cur.description], row))


# --------------------------------------------------------------- translations


def write_translation(
    conn: sqlite3.Connection,
    utterance_id: int,
    primary: str,
    alternatives: list[str],
    per_word_confidence: list[float],
    overall_confidence: float,
    emergency: bool,
    context_hints: list[str],
    model: str,
    usage: dict,
) -> int:
    cur = conn.execute(
        "INSERT INTO translations"
        " (utterance_id, primary_text, alternatives_json, per_word_confidence_json,"
        "  overall_confidence, emergency, context_hints_json, model,"
        "  cache_read_tokens, cache_creation_tokens, input_tokens, output_tokens)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            utterance_id,
            primary,
            json.dumps(alternatives),
            json.dumps(per_word_confidence),
            float(overall_confidence),
            1 if emergency else 0,
            json.dumps(context_hints),
            model,
            int(usage.get("cache_read_tokens", 0)),
            int(usage.get("cache_creation_tokens", 0)),
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def write_translation_feedback(
    conn: sqlite3.Connection,
    utterance_id: int,
    confirmed: bool,
    corrected_text: str | None,
) -> bool:
    """Record listener feedback on a translation. Returns True if a row matched."""
    payload = {
        "confirmed": bool(confirmed),
        "corrected_text": corrected_text,
        "at": time.time(),
    }
    cur = conn.execute(
        "UPDATE translations SET feedback = ? WHERE utterance_id = ?",
        (json.dumps(payload), utterance_id),
    )
    conn.commit()
    return cur.rowcount > 0


# --------------------------------------------------------------- adaptation


def write_online_observations(
    conn: sqlite3.Connection,
    calibration_id: int | None,
    utterance_id: int,
    pairs: list[tuple[str, str]],
    source: str,
    weight: float = 1.0,
) -> int:
    """Persist (intended, produced) pairs from a confirmed/corrected translation."""
    if not pairs:
        return 0
    now = time.time()
    rows = [
        (calibration_id, utterance_id, now, intended, produced, weight, source)
        for intended, produced in pairs
    ]
    conn.executemany(
        "INSERT INTO online_observations"
        " (calibration_id, utterance_id, observed_at, intended, produced, weight, source)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def list_online_observations(
    conn: sqlite3.Connection,
    calibration_id: int | None,
    since_ts: float | None = None,
) -> list[dict]:
    sql = (
        "SELECT id, utterance_id, observed_at, intended, produced, weight, source"
        " FROM online_observations WHERE 1=1"
    )
    params: list = []
    if calibration_id is not None:
        sql += " AND calibration_id = ?"
        params.append(calibration_id)
    if since_ts is not None:
        sql += " AND observed_at >= ?"
        params.append(since_ts)
    sql += " ORDER BY observed_at"
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_translations_with_feedback(
    conn: sqlite3.Connection, since_ts: float | None = None
) -> list[dict]:
    sql = (
        "SELECT t.id, t.utterance_id, t.primary_text, t.overall_confidence,"
        " t.feedback, t.created_at, t.alternatives_json,"
        " u.session_id"
        " FROM translations t"
        " LEFT JOIN utterances u ON u.id = t.utterance_id"
        " WHERE t.feedback IS NOT NULL"
    )
    params: list = []
    if since_ts is not None:
        sql += " AND t.created_at >= ?"
        params.append(since_ts)
    sql += " ORDER BY t.created_at"
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def write_intelligibility_snapshot(
    conn: sqlite3.Connection,
    calibration_id: int | None,
    week_start: float,
    n_utterances: int,
    diagonal_mass: float,
    confirmed_rate: float,
    avg_overall_confidence: float,
) -> None:
    conn.execute(
        "INSERT INTO intelligibility_snapshots"
        " (calibration_id, week_start, n_utterances, diagonal_mass,"
        "  confirmed_rate, avg_overall_confidence)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(calibration_id, week_start) DO UPDATE SET"
        "   n_utterances = excluded.n_utterances,"
        "   diagonal_mass = excluded.diagonal_mass,"
        "   confirmed_rate = excluded.confirmed_rate,"
        "   avg_overall_confidence = excluded.avg_overall_confidence",
        (
            calibration_id,
            week_start,
            n_utterances,
            float(diagonal_mass),
            float(confirmed_rate),
            float(avg_overall_confidence),
        ),
    )
    conn.commit()


def list_intelligibility_snapshots(
    conn: sqlite3.Connection,
    calibration_id: int | None = None,
    since_ts: float | None = None,
) -> list[dict]:
    sql = (
        "SELECT calibration_id, week_start, n_utterances, diagonal_mass,"
        " confirmed_rate, avg_overall_confidence FROM intelligibility_snapshots"
        " WHERE 1=1"
    )
    params: list = []
    if calibration_id is not None:
        sql += " AND calibration_id = ?"
        params.append(calibration_id)
    if since_ts is not None:
        sql += " AND week_start >= ?"
        params.append(since_ts)
    sql += " ORDER BY week_start"
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# --------------------------------------------------------------- vocabulary


def list_personal_vocabulary(
    conn: sqlite3.Connection, kind: str | None = None
) -> list[dict]:
    sql = "SELECT id, kind, term, notes, boost, created_at FROM personal_vocabulary"
    params: list = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    sql += " ORDER BY kind, term"
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def add_personal_vocabulary(
    conn: sqlite3.Connection,
    kind: str,
    term: str,
    notes: str | None,
    boost: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO personal_vocabulary (kind, term, notes, boost)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(kind, term) DO UPDATE SET"
        "   notes = excluded.notes,"
        "   boost = excluded.boost",
        (kind, term, notes, float(boost)),
    )
    conn.commit()
    return int(cur.lastrowid)


def delete_personal_vocabulary(conn: sqlite3.Connection, vocab_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM personal_vocabulary WHERE id = ?", (vocab_id,)
    )
    conn.commit()
    return cur.rowcount > 0


def list_recent_patient_utterances(
    conn: sqlite3.Connection, session_id: int, limit: int = 20
) -> list[dict]:
    """Return the last N patient utterances with their primary translation, oldest first."""
    cur = conn.execute(
        "SELECT u.id, u.start_ms, u.end_ms, u.speaker_label, u.is_patient,"
        " u.transcript, t.primary_text"
        " FROM utterances u LEFT JOIN translations t ON t.utterance_id = u.id"
        " WHERE u.session_id = ?"
        " ORDER BY u.id DESC LIMIT ?",
        (session_id, limit),
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return list(reversed(rows))


# --------------------------------------------------------------- context


def write_topic_observation(
    conn: sqlite3.Connection, session_id: int, keyword: str, weight: float
) -> None:
    conn.execute(
        "INSERT INTO topics (session_id, observed_at, keyword, weight) VALUES (?, ?, ?, ?)",
        (session_id, time.time(), keyword, float(weight)),
    )
    conn.commit()


def write_pending_question(
    conn: sqlite3.Connection, session_id: int, text: str, qtype: str
) -> None:
    conn.execute(
        "INSERT INTO pending_questions (session_id, asked_at, text, qtype) VALUES (?, ?, ?, ?)",
        (session_id, time.time(), text, qtype),
    )
    conn.commit()


def mark_questions_answered(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "UPDATE pending_questions SET answered = 1 WHERE session_id = ? AND answered = 0",
        (session_id,),
    )
    conn.commit()


def write_emotion_snapshot(
    conn: sqlite3.Connection, session_id: int, frustration: float, distress: float
) -> None:
    conn.execute(
        "INSERT INTO emotional_states (session_id, observed_at, frustration, distress) VALUES (?, ?, ?, ?)",
        (session_id, time.time(), float(frustration), float(distress)),
    )
    conn.commit()


def upsert_session_metadata(
    conn: sqlite3.Connection,
    session_id: int,
    *,
    setting: str | None = None,
    listener_role_hint: str | None = None,
    time_of_day: str | None = None,
    activity_context: str | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO session_metadata"
        " (session_id, setting, listener_role_hint, time_of_day, activity_context, notes)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(session_id) DO UPDATE SET"
        "   setting = COALESCE(excluded.setting, setting),"
        "   listener_role_hint = COALESCE(excluded.listener_role_hint, listener_role_hint),"
        "   time_of_day = COALESCE(excluded.time_of_day, time_of_day),"
        "   activity_context = COALESCE(excluded.activity_context, activity_context),"
        "   notes = COALESCE(excluded.notes, notes),"
        "   updated_at = strftime('%s','now')",
        (
            session_id,
            setting,
            listener_role_hint,
            time_of_day,
            activity_context,
            notes,
        ),
    )
    conn.commit()


# --------------------------------------------------------------- fatigue


def write_fatigue_snapshot(
    conn: sqlite3.Connection,
    session_id: int,
    minute: int,
    error_density: float,
    phase: str,
) -> None:
    conn.execute(
        "INSERT INTO fatigue_snapshots (session_id, minute, error_density, phase)"
        " VALUES (?, ?, ?, ?)",
        (session_id, minute, error_density, phase),
    )
    conn.commit()


# --------------------------------------------------------------- torgo


def get_torgo_clip(conn: sqlite3.Connection, clip_id: int) -> dict | None:
    cur = conn.execute(
        "SELECT id, speaker_id, condition, intelligibility, prompt_set,"
        " intended_text, intended_phonemes, wav_path, duration_ms, sample_rate"
        " FROM torgo_manifest WHERE id = ?",
        (clip_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([c[0] for c in cur.description], row))
