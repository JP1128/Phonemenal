"""FastAPI app: VAD-gated, speaker-routed phoneme + STT pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import adaptation as adapt_mod
from . import features as feat_mod
from . import llm
from . import vocabulary as vocab_mod
from .alignment import alignment_score, forced_align, text_to_phonemes
from .audio import save_wav
from .config import settings
from .context import ContextManager, classify_question
from .db import (
    add_calibration_utterance,
    complete_calibration,
    connect,
    create_calibration,
    create_session,
    create_training_job,
    create_utterance,
    delete_calibration,
    end_session,
    get_calibration,
    get_torgo_clip,
    get_training_job,
    list_calibrations,
    list_corrective_training_pairs,
    list_training_jobs_for_calibration,
    latest_patient_profile,
    list_calibration_utterances,
    list_recent_patient_utterances,
    list_torgo_clips,
    mark_questions_answered,
    rename_calibration,
    set_utterance_transcript,
    upsert_session_metadata,
    write_emotion_snapshot,
    write_features,
    write_frames,
    write_patient_profile,
    write_pending_question,
    write_topic_observation,
    write_translation,
    write_translation_feedback,
    list_intelligibility_snapshots,
    list_personal_vocabulary,
)
from .distortion import build_profile
from .emergency import check_prosody, check_text
from .phonemes import FRAME_STRIDE_SAMPLES, PhonemeExtractor
from .silero_vad import make_vad
from .speaker_id import SpeakerIdentifier, ensure_schema
from .stt import AdaptedTranscriber, GatedTranscriber, Transcriber
from .vad import Utterance, UtteranceSegmenter

logger = logging.getLogger("phonemer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading phoneme model: %s", settings.phoneme_model_name)
    app.state.extractor = PhonemeExtractor(
        settings.phoneme_model_name, device=settings.device
    )
    logger.info("Phoneme model ready (device=%s)", settings.device)

    logger.info("Loading speaker model: %s", settings.speaker_model_source)
    app.state.speaker = SpeakerIdentifier(
        model_source=settings.speaker_model_source,
        savedir=settings.speaker_model_dir,
        device=settings.device,
        match_threshold=settings.speaker_match_threshold,
        reject_threshold=settings.speaker_reject_threshold,
    )
    logger.info("Speaker model ready")

    logger.info("Loading VAD backend: %s", settings.vad_backend)
    app.state.vad = make_vad(
        settings.vad_backend,
        threshold=settings.vad_speech_threshold,
        aggressiveness=settings.vad_aggressiveness,
    )
    logger.info("VAD ready")

    # Plain Whisper for non-patient (listener / caregiver) speech. faster-
    # whisper is used because it's much faster at inference than HF Whisper
    # and patient personalization isn't needed for unimpaired speakers.
    app.state.transcriber = Transcriber(
        model_size=settings.stt_model_size,
        compute_type=settings.stt_compute_type,
        device=settings.device,
    )

    # Patient transcription path: prefer per-patient PEFT adapter when one
    # exists at ``settings.patient_adapter_dir``; fall back to the plain
    # transcriber when not. The adapter is the calibrated personalization
    # produced by ``server/scripts/train_patient_adapter.py`` —
    # whisper-small base + LoRA delta calibrated to this patient's
    # speech (-32% to -47% relative WER on TORGO dysarthric speakers).
    patient_adapter_path = Path(settings.patient_adapter_dir)
    has_adapter = (
        patient_adapter_path.exists()
        and (patient_adapter_path / "adapter_model.safetensors").exists()
        and (patient_adapter_path / "adapter_config.json").exists()
    )
    if has_adapter:
        logger.info(
            "Patient adapter found at %s — using AdaptedTranscriber",
            patient_adapter_path,
        )
        app.state.patient_transcriber = AdaptedTranscriber(
            adapter_dir=str(patient_adapter_path),
            device=settings.device,
        )
    else:
        logger.info(
            "No patient adapter at %s — patient path uses plain Whisper",
            patient_adapter_path,
        )
        app.state.patient_transcriber = app.state.transcriber

    # Gated transcriber: Whisper + phoneme-alignment safety gate. This is
    # the patient translation path now that empirical evidence (see
    # JOURNAL.md entries 12-13) showed the calibrated LLM layer can't beat
    # naive Whisper. The gate flags suspected hallucinations so the UI
    # can render them as low-trust rather than confidently surfacing
    # made-up content.
    app.state.gate = GatedTranscriber(
        transcriber=app.state.patient_transcriber,
        extractor=app.state.extractor,
        token_to_id=app.state.extractor.processor.tokenizer.get_vocab(),
        blank_id=app.state.extractor.processor.tokenizer.pad_token_id or 0,
        sample_rate=settings.sample_rate,
        min_alignment_score=settings.gate_min_alignment_score,
        min_phoneme_density=settings.gate_min_phoneme_density,
        density_check_min_duration_s=settings.gate_density_check_min_duration_s,
        n_best=settings.gate_n_best,
    )

    # Multi-base support. The default ``patient`` adapter pins a single
    # base (whisper-small.en today). But some speakers — e.g. M01 — have
    # a separately-trained adapter against a *different* base
    # (whisper-medium.en) that meaningfully outperforms the small one.
    # Load one ``AdaptedTranscriber`` + ``GatedTranscriber`` per unique
    # base model referenced by any adapter on disk so a single backend
    # can route each speaker's request to the right base without
    # restart. Memory cost is ~3 GB extra for whisper-medium fp32 — fine
    # on a 24 GB Apple M4. ``_resolve_adapter_dir`` picks the most
    # recently-trained adapter for a speaker; ``_pick_gate_for_adapter``
    # finds the matching gate by reading the adapter's BASE_MODEL.txt.
    app.state.bases: dict[str, dict] = {}
    adapters_root = Path(settings.patient_adapter_dir).parent
    discovered_bases: set[str] = set()
    if adapters_root.exists():
        for d in adapters_root.iterdir():
            if not d.is_dir():
                continue
            if d.name == "patient" or d.name.startswith("pool_"):
                continue
            if not (d / "adapter_model.safetensors").exists():
                continue
            marker = d / "BASE_MODEL.txt"
            if not marker.exists():
                continue
            try:
                discovered_bases.add(marker.read_text().strip())
            except OSError:
                pass
    # Always include the configured default's base.
    default_marker = Path(settings.patient_adapter_dir) / "BASE_MODEL.txt"
    default_base = (
        default_marker.read_text().strip() if default_marker.exists()
        else "openai/whisper-small.en"
    )
    discovered_bases.add(default_base)
    # The default base is already loaded above. Register it as the
    # default and lazy-load any *additional* bases on top.
    app.state.bases[default_base] = {
        "transcriber": app.state.patient_transcriber,
        "gate": app.state.gate,
        "default_adapter": str(patient_adapter_path),
    }
    for base in discovered_bases:
        if base in app.state.bases:
            continue
        # Pick a default adapter for this base (any adapter whose
        # BASE_MODEL.txt matches). Used only as the *initial* adapter
        # the transcriber loads; per-request adapter swaps happen via
        # AdaptedTranscriber._ensure_adapter.
        seed: Path | None = None
        for d in adapters_root.iterdir():
            if not d.is_dir():
                continue
            if d.name == "patient" or d.name.startswith("pool_"):
                continue
            cand_marker = d / "BASE_MODEL.txt"
            if not cand_marker.exists():
                continue
            try:
                if cand_marker.read_text().strip() == base and (
                    d / "adapter_model.safetensors"
                ).exists():
                    seed = d
                    break
            except OSError:
                continue
        if seed is None:
            logger.warning("no seed adapter found for base %s, skipping", base)
            continue
        logger.info("Loading additional base %s with seed adapter %s", base, seed)
        extra_transcriber = AdaptedTranscriber(
            adapter_dir=str(seed),
            base_model=base,
            device=settings.device,
        )
        extra_gate = GatedTranscriber(
            transcriber=extra_transcriber,
            extractor=app.state.extractor,
            token_to_id=app.state.extractor.processor.tokenizer.get_vocab(),
            blank_id=app.state.extractor.processor.tokenizer.pad_token_id or 0,
            sample_rate=settings.sample_rate,
            min_alignment_score=settings.gate_min_alignment_score,
            min_phoneme_density=settings.gate_min_phoneme_density,
            density_check_min_duration_s=settings.gate_density_check_min_duration_s,
            n_best=settings.gate_n_best,
        )
        app.state.bases[base] = {
            "transcriber": extra_transcriber,
            "gate": extra_gate,
            "default_adapter": str(seed),
        }
    logger.info(
        "loaded %d base(s): %s",
        len(app.state.bases), list(app.state.bases.keys()),
    )

    app.state.db = connect(settings.db_path)
    ensure_schema(app.state.db)
    Path(settings.audio_dir).mkdir(parents=True, exist_ok=True)
    logger.info("SQLite ready at %s", settings.db_path)

    # Warm the spaCy pipeline used by topic extraction so the first patient
    # utterance doesn't eat the ~200 ms load latency. No-op if spaCy isn't
    # installed — TopicTracker silently falls back to its regex path.
    from .context import _get_nlp as _warm_nlp
    _warm_nlp()

    try:
        yield
    finally:
        app.state.db.close()


app = FastAPI(title="Phonemer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================ HTTP endpoints


@app.get("/healthz")
async def healthz() -> dict:
    has_adapter = isinstance(app.state.patient_transcriber, AdaptedTranscriber)
    return {
        "ok": True,
        "model": settings.phoneme_model_name,
        "speaker_model": settings.speaker_model_source,
        "stt_model": settings.stt_model_size,
        "sample_rate": settings.sample_rate,
        "patient_adapter": (
            settings.patient_adapter_dir if has_adapter else None
        ),
    }


_CAL_SENTENCES_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration_sentences.json"


def _load_calibration_sentence_sets() -> dict:
    return json.loads(_CAL_SENTENCES_PATH.read_text())


@app.get("/calibration/sentences")
async def calibration_sentences() -> dict:
    return _load_calibration_sentence_sets()


@app.get("/calibrations")
async def calibrations_list() -> dict:
    """List every calibration with its utterance count so the doctor
    can resume an in-progress patient instead of always starting fresh.
    """
    return {"calibrations": list_calibrations(app.state.db)}


@app.patch("/calibrations/{calibration_id}")
async def calibration_rename(
    calibration_id: int, payload: dict | None = Body(None)
) -> dict:
    """Rename a calibration row's display label. The clinician uses this
    to attach a label after the fact ("Room 412") to a row that was
    started without one. Passing an empty string clears the label so
    the UI falls back to ``Calibration #<id>``.
    """
    body = payload or {}
    raw = body.get("display_name")
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(400, "display_name must be a string or null")
    normalized = raw.strip() if isinstance(raw, str) else None
    if normalized == "":
        normalized = None
    if not rename_calibration(app.state.db, calibration_id, normalized):
        raise HTTPException(404, f"calibration {calibration_id} not found")
    return {"id": calibration_id, "display_name": normalized}


@app.delete("/calibrations/{calibration_id}")
async def calibration_delete(calibration_id: int) -> dict:
    """Remove an abandoned or test calibration row + its utterances /
    training jobs (FK CASCADE handles dependents). Returns 404 if the
    row doesn't exist so the UI can refresh the list cleanly.
    """
    if not delete_calibration(app.state.db, calibration_id):
        raise HTTPException(404, f"calibration {calibration_id} not found")
    return {"deleted": calibration_id}


@app.post("/calibration/start")
async def calibration_start(payload: dict | None = Body(None)) -> dict:
    """Start a new calibration or resume an existing one.

    Body (optional):
      * ``calibration_id`` — resume the given calibration; preserves all
        utterances recorded so far. Returns 404 if not found.
      * ``display_name`` — human-readable patient label. Stored on the
        calibrations row; shown in the resume list. Ignored when resuming
        (use a future rename endpoint to change it).
    """
    body = payload or {}
    resume_id = body.get("calibration_id")
    display_name = body.get("display_name") or None
    if isinstance(display_name, str):
        display_name = display_name.strip() or None

    sentence_sets = _load_calibration_sentence_sets()["sets"]

    if resume_id is not None:
        try:
            resume_id_int = int(resume_id)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid calibration_id: {resume_id!r}") from e
        cal = get_calibration(app.state.db, resume_id_int)
        if cal is None:
            raise HTTPException(404, f"calibration {resume_id_int} not found")
        utterances = list_calibration_utterances(app.state.db, resume_id_int)
        return {
            "calibration_id": cal["id"],
            "display_name": cal["display_name"],
            "started_at": cal["started_at"],
            "completed_at": cal["completed_at"],
            "resumed": True,
            "utterance_count": len(utterances),
            "sentence_sets": sentence_sets,
        }

    cid = create_calibration(app.state.db, display_name=display_name)
    return {
        "calibration_id": cid,
        "display_name": display_name,
        "resumed": False,
        "utterance_count": 0,
        "sentence_sets": sentence_sets,
    }


@app.get("/calibration/{calibration_id}")
async def calibration_get(calibration_id: int) -> dict:
    cal = get_calibration(app.state.db, calibration_id)
    if cal is None:
        raise HTTPException(404, f"calibration {calibration_id} not found")
    utterances = list_calibration_utterances(app.state.db, calibration_id)
    profile = latest_patient_profile(app.state.db, calibration_id)
    return {
        "calibration_id": calibration_id,
        "display_name": cal["display_name"],
        "started_at": cal["started_at"],
        "completed_at": cal["completed_at"],
        "utterance_count": len(utterances),
        "utterances": utterances,
        "profile": profile,
    }


_TRAIN_MIN_UTTERANCES = 30
# Per-process registry of the live training subprocesses we've spawned.
# Keyed by training_jobs.id; values are the Popen handles. Used to (a)
# avoid retaining zombies on process exit and (b) liveness-check a row
# whose subprocess crashed before writing its terminal status to the DB.
_TRAINING_PROCS: dict[int, subprocess.Popen] = {}


def _adapter_dir_for_calibration(calibration_id: int, slug: str) -> str:
    """Resolve the adapter output directory used for a calibration's
    trained LoRA. Sibling of ``settings.patient_adapter_dir`` so the
    /adapters endpoint picks it up automatically."""
    base = Path(settings.patient_adapter_dir).parent
    safe_slug = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in slug
    ).strip("_") or f"cal_{calibration_id}"
    return str((base / safe_slug).resolve())


@app.post("/calibrations/{calibration_id}/train-adapter")
async def train_adapter(
    calibration_id: int, payload: dict | None = Body(None)
) -> dict:
    """Spawn a training subprocess that fine-tunes a per-patient LoRA
    adapter on Whisper using this calibration's recorded utterances.

    The subprocess updates its own ``training_jobs`` row as it runs
    (queued → running → succeeded/failed); the caller polls
    ``GET /training-jobs/{job_id}`` for status + log tail.
    """
    body = payload or {}
    output_name = body.get("output_name")
    extra_args: list[str] = body.get("extra_args") or []

    cal = get_calibration(app.state.db, calibration_id)
    if cal is None:
        raise HTTPException(404, f"calibration {calibration_id} not found")

    utterances = list_calibration_utterances(app.state.db, calibration_id)
    usable = [u for u in utterances if u.get("audio_path")]
    if len(usable) < _TRAIN_MIN_UTTERANCES:
        raise HTTPException(
            400,
            f"need at least {_TRAIN_MIN_UTTERANCES} recorded utterances to train"
            f" (have {len(usable)})",
        )

    # Pick adapter slug. Priority: explicit output_name → display_name → cal_<id>.
    slug = (
        (output_name or "").strip()
        or (cal.get("display_name") or "").strip()
        or f"cal_{calibration_id}"
    )
    adapter_dir = _adapter_dir_for_calibration(calibration_id, slug)

    # Log location for live tail.
    logs_dir = Path(settings.audio_dir).parent / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(logs_dir / f"train_{int(time.time())}_{calibration_id}.log")

    job_id = create_training_job(
        app.state.db,
        calibration_id=calibration_id,
        adapter_dir=adapter_dir,
        speaker_label=cal.get("display_name"),
        log_path=log_path,
    )

    # Spawn the trainer. We use the same Python that's running FastAPI so
    # the venv with torch/peft/transformers is guaranteed available.
    server_root = Path(__file__).resolve().parents[1]
    # The adapter slug derived above is what the WS handler stamps on
    # patient utterances when this calibration's adapter is active.
    # Pass it to the trainer so it can pull live confirmed/corrected
    # utterances back into the training set ("feedback retraining").
    safe_slug = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in slug
    ).strip("_") or f"cal_{calibration_id}"
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_patient_adapter",
        "--calibration-id", str(calibration_id),
        "--output-dir", adapter_dir,
        "--job-id", str(job_id),
        "--include-feedback-slug", safe_slug,
    ]
    if output_name:
        cmd.extend(["--output-name", str(output_name)])
    if extra_args:
        cmd.extend([str(a) for a in extra_args])

    try:
        log_handle = open(log_path, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(server_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        _TRAINING_PROCS[job_id] = proc
        logger.info(
            "training job %s spawned (pid=%s, cal=%s, adapter=%s)",
            job_id, proc.pid, calibration_id, adapter_dir,
        )
    except Exception as e:  # noqa: BLE001
        from .db import update_training_job
        update_training_job(
            app.state.db,
            job_id,
            status="failed",
            error_message=f"could not spawn trainer: {e}",
            completed_at=time.time(),
        )
        raise HTTPException(500, f"could not spawn trainer: {e}") from e

    # Report how many extra (audio, listener-validated text) pairs from
    # past corrections/confirmations will be merged into the training
    # set. The clinician sees this count so they understand the
    # "Retrain" button isn't only using calibration prompts.
    feedback_pairs = list_corrective_training_pairs(app.state.db, safe_slug)
    return {
        "job_id": job_id,
        "calibration_id": calibration_id,
        "adapter_dir": adapter_dir,
        "log_path": log_path,
        "status": "queued",
        "calibration_clip_count": len(usable),
        "feedback_clip_count": len(feedback_pairs),
    }


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _tail_log(path: str | None, max_lines: int = 200) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            # Cheap tail — small log files; fine to read all.
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return "\n".join(lines[-max_lines:])
    except FileNotFoundError:
        return ""
    except Exception as e:  # noqa: BLE001
        return f"(could not read log: {e})"


@app.get("/training-jobs/{job_id}")
async def training_job_status(job_id: int) -> dict:
    job = get_training_job(app.state.db, job_id)
    if job is None:
        raise HTTPException(404, f"training job {job_id} not found")
    # Reconcile DB status with reality: if the subprocess died before
    # writing 'succeeded' / 'failed', flip the row to 'failed' so the UI
    # doesn't poll forever.
    if job["status"] in ("queued", "running"):
        pid = job.get("pid")
        if pid is not None and not _is_pid_alive(pid):
            from .db import update_training_job
            update_training_job(
                app.state.db,
                job_id,
                status="failed",
                error_message=(
                    "training process exited without writing a terminal status"
                ),
                completed_at=time.time(),
            )
            job = get_training_job(app.state.db, job_id) or job
    job["log_tail"] = _tail_log(job.get("log_path"))
    return job


@app.get("/calibrations/{calibration_id}/training-jobs")
async def training_jobs_for_calibration(calibration_id: int) -> dict:
    if get_calibration(app.state.db, calibration_id) is None:
        raise HTTPException(404, f"calibration {calibration_id} not found")
    return {
        "calibration_id": calibration_id,
        "jobs": list_training_jobs_for_calibration(app.state.db, calibration_id),
    }


@app.get("/calibration/utterance/{utterance_id}/audio")
async def calibration_utterance_audio(utterance_id: int) -> FileResponse:
    cur = app.state.db.execute(
        "SELECT audio_path FROM calibration_utterances WHERE id = ?",
        (utterance_id,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "audio not available for this utterance")
    return FileResponse(row[0], media_type="audio/wav")


@app.get("/utterance/{utterance_id}/audio")
async def utterance_audio(utterance_id: int) -> FileResponse:
    """Serve the captured PCM/WAV for a live (non-calibration) utterance.

    Used by the listener UI to replay what the patient actually said so the
    listener can sanity-check the model's transcription.
    """
    cur = app.state.db.execute(
        "SELECT audio_path FROM utterances WHERE id = ?",
        (utterance_id,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "audio not available for this utterance")
    return FileResponse(row[0], media_type="audio/wav")


@app.post("/calibration/{calibration_id}/build")
async def calibration_build(calibration_id: int) -> dict:
    """Re-run alignment over every utterance and build the distortion profile."""
    extractor: PhonemeExtractor = app.state.extractor
    db = app.state.db

    rows = list_calibration_utterances(db, calibration_id)
    if not rows:
        raise HTTPException(404, "no calibration utterances yet")

    vocab = list(extractor.processor.tokenizer.get_vocab().keys())
    blank_id = extractor.processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = 0
    token_to_id = extractor.processor.tokenizer.get_vocab()

    aligned_utterances: list[dict] = []
    features_list: list[dict] = []
    rejected_low_score = 0
    loop = asyncio.get_running_loop()

    for row in rows:
        if not row["audio_path"] or not row["intended_text"]:
            continue
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != settings.sample_rate:
            continue

        # Prosody baseline samples — collected over every calibration
        # utterance the patient produced, so the runtime emergency / emotion
        # detectors can threshold relative to the patient's normal speech.
        try:
            feats = await loop.run_in_executor(
                None, feat_mod.extract, audio, settings.sample_rate
            )
            features_list.append(feats.to_json())
        except Exception:  # noqa: BLE001 — baseline is best-effort
            logger.debug(
                "calibration: features.extract failed for utt=%s",
                row.get("id"),
                exc_info=True,
            )

        probs = await loop.run_in_executor(
            None, extractor.extract, audio, settings.sample_rate
        )
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))

        intended_tokens = text_to_phonemes(row["intended_text"])
        word_phonemes = [text_to_phonemes(w) for w in row["intended_text"].split()]

        aligned = forced_align(
            log_probs=log_probs,
            intended_tokens=intended_tokens,
            token_to_id=token_to_id,
            blank_id=blank_id,
        )

        # Drop utterances the model couldn't actually fit to the transcript:
        # silence, throat-clear, or recording where the patient said something
        # other than the prompt. Without this gate, a few bad clips can
        # dominate the confusion matrix.
        score = alignment_score(aligned)
        if score < settings.min_alignment_score:
            rejected_low_score += 1
            logger.info(
                "calibration: dropping utt=%s (alignment score %.2f < %.2f)",
                row.get("id"),
                score,
                settings.min_alignment_score,
            )
            continue

        produced_top = extractor.top_k_per_frame(probs, settings.top_k_phonemes)

        aligned_utterances.append(
            {
                "intended_tokens": intended_tokens,
                "intended_words_phonemes": word_phonemes,
                "aligned": aligned,
                "produced": produced_top,
            }
        )

    profile = build_profile(
        aligned_utterances,
        vocab_symbols=vocab,
        features_list=features_list,
    )
    profile_payload = json.loads(profile.to_json())

    # Optional Opus pass — only if an API key is configured.
    if llm.is_configured():
        try:
            examples = [
                {
                    "intended": rows[i]["intended_text"],
                    "intended_phonemes": rows[i].get("intended_phonemes"),
                }
                for i in range(min(len(rows), 10))
            ]
            insights = await loop.run_in_executor(
                None,
                lambda: llm.analyze_calibration(
                    model=settings.analysis_model,
                    empirical_profile={
                        "n_calibration_utterances": profile.n_calibration_utterances,
                        "syllable_deletion_rate": profile.syllable_deletion_rate,
                        "intended_phoneme_counts": profile.intended_phoneme_counts,
                        "cluster_simplifications": profile.cluster_simplifications,
                    },
                    examples=examples,
                ),
            )
            profile_payload["opus_insights"] = insights
        except Exception:
            logger.exception("Opus calibration analysis failed")

    pid = write_patient_profile(db, calibration_id, json.dumps(profile_payload, ensure_ascii=False))
    complete_calibration(db, calibration_id)
    return {
        "calibration_id": calibration_id,
        "profile_id": pid,
        "n_utterances": profile.n_calibration_utterances,
        "rejected_low_score": rejected_low_score,
        "syllable_deletion_rate": profile.syllable_deletion_rate,
        "opus_summary": profile_payload.get("opus_insights", {}).get("summary"),
    }


@app.websocket("/ws/calibration/{calibration_id}")
async def ws_calibration(websocket: WebSocket, calibration_id: int) -> None:
    """Capture one impaired utterance per intended sentence.

    Wire protocol (text from client):
      {"type": "begin", "set_id": "basic_needs", "sentence_index": 0,
       "intended_text": "I need water."}
      <binary PCM frames>
      {"type": "end"}                       # commit and persist
      {"type": "discard"}                   # throw away buffered audio
      {"type": "stop"}                      # close the calibration session

    Server replies:
      {"type": "calibration_utterance", "utterance_id": int, "duration_ms": int}
      {"type": "ready"} after each begin/end.
    """
    await websocket.accept()
    extractor: PhonemeExtractor = websocket.app.state.extractor
    db = websocket.app.state.db

    pcm = bytearray()
    current: dict | None = None
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            raw = msg.get("bytes")
            if raw is not None and current is not None:
                pcm.extend(raw)
                continue

            text = msg.get("text")
            if text is None:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue

            kind = payload.get("type")
            if kind == "begin":
                pcm = bytearray()
                current = payload
                await websocket.send_text(json.dumps({"type": "ready"}))
            elif kind == "end":
                if current is None:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "calibration_error",
                                "error": "no_active_recording",
                                "message": "Received `end` without a matching `begin`.",
                            }
                        )
                    )
                    continue
                if len(pcm) < settings.sample_rate * 2 // 5:  # < 200 ms at 16 kHz int16
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "calibration_error",
                                "error": "too_short",
                                "message": "Recording was too short — please try again.",
                                "received_bytes": len(pcm),
                            }
                        )
                    )
                    pcm = bytearray()
                    current = None
                    continue
                try:
                    samples = (
                        np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    duration_ms = int(1000 * len(samples) / settings.sample_rate)
                    name = (
                        f"cal{calibration_id}_"
                        f"{current['set_id']}_{current['sentence_index']:03d}"
                    )
                    audio_path = save_wav(
                        samples, settings.sample_rate, settings.audio_dir, name
                    )
                    probs = await loop.run_in_executor(
                        None, extractor.extract, samples, settings.sample_rate
                    )
                    top = extractor.top_k_per_frame(probs, settings.top_k_phonemes)
                    produced_json = json.dumps(
                        [[{"symbol": s, "prob": p} for s, p in frame] for frame in top]
                    )
                    intended_phonemes = " ".join(
                        text_to_phonemes(current["intended_text"])
                    )
                    utt_id = add_calibration_utterance(
                        db,
                        calibration_id=calibration_id,
                        sentence_set_id=current["set_id"],
                        sentence_index=int(current["sentence_index"]),
                        intended_text=current["intended_text"],
                        intended_phonemes=intended_phonemes,
                        audio_path=audio_path,
                        duration_ms=duration_ms,
                        produced_phoneme_distribution_json=produced_json,
                        accepted=True,
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "calibration_utterance",
                                "utterance_id": utt_id,
                                "duration_ms": duration_ms,
                            }
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "Calibration commit failed for cal=%s sentence=%s",
                        calibration_id,
                        current.get("sentence_index"),
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "calibration_error",
                                "error": "commit_failed",
                                "message": f"{type(e).__name__}: {e}",
                            }
                        )
                    )
                finally:
                    pcm = bytearray()
                    current = None
            elif kind == "discard":
                pcm = bytearray()
                current = None
                await websocket.send_text(json.dumps({"type": "ready"}))
            elif kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.post("/feedback/{utterance_id}")
async def post_feedback(utterance_id: int, payload: dict) -> dict:
    """Listener confirms or corrects a translation.

    Body: ``{"confirmed": bool, "corrected_text": str | null}``

    Persists feedback, then aligns the (corrected or original) text against
    the utterance's phoneme distributions and writes the resulting pairs to
    ``online_observations`` so the next translation incorporates the new
    signal via :func:`adaptation.effective_profile_summary`.
    """
    confirmed = bool(payload.get("confirmed", True))
    corrected_text = payload.get("corrected_text")
    if corrected_text is not None and not isinstance(corrected_text, str):
        raise HTTPException(400, "corrected_text must be a string or null")
    db = app.state.db
    matched = write_translation_feedback(db, utterance_id, confirmed, corrected_text)
    if not matched:
        raise HTTPException(404, "no translation row for that utterance_id")

    # Resolve the intended text (corrected if provided; otherwise the LLM's primary).
    cur = db.execute(
        "SELECT primary_text FROM translations WHERE utterance_id = ?",
        (utterance_id,),
    )
    row = cur.fetchone()
    intended_text = corrected_text or (row[0] if row else None) or ""
    cal_id = _latest_calibration_id(db)

    n_pairs = 0
    shift_alert = None
    if intended_text:
        loop = asyncio.get_running_loop()
        try:
            n_pairs = await loop.run_in_executor(
                None,
                lambda: adapt_mod.ingest_feedback(
                    conn=db,
                    extractor=app.state.extractor,
                    utterance_id=utterance_id,
                    intended_text=intended_text,
                    confirmed=confirmed,
                    corrected_text=corrected_text,
                    calibration_id=cal_id,
                ),
            )
            shift_alert = await loop.run_in_executor(
                None, lambda: adapt_mod.detect_shift(db, cal_id)
            )
        except Exception:
            logger.exception("adaptation step failed for utt=%s", utterance_id)
    return {
        "ok": True,
        "utterance_id": utterance_id,
        "observations_written": n_pairs,
        "shift": shift_alert,
    }


def _latest_calibration_id(db) -> int | None:
    cur = db.execute(
        "SELECT id FROM calibrations ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


# ----------------------------------------------------------- vocabulary


@app.get("/vocabulary")
async def get_vocabulary(kind: str | None = None) -> dict:
    return {"items": list_personal_vocabulary(app.state.db, kind)}


@app.post("/vocabulary")
async def post_vocabulary(payload: dict) -> dict:
    kind = str(payload.get("kind", "")).strip()
    term = str(payload.get("term", "")).strip()
    notes = payload.get("notes")
    boost = float(payload.get("boost", 1.5))
    try:
        item = vocab_mod.upsert_term(
            app.state.db, kind=kind, term=term, notes=notes, boost=boost
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "item": item}


@app.delete("/vocabulary/{vocab_id}")
async def delete_vocabulary(vocab_id: int) -> dict:
    ok = vocab_mod.remove_term(app.state.db, vocab_id)
    if not ok:
        raise HTTPException(404, "vocabulary entry not found")
    return {"ok": True}


# ----------------------------------------------------------- progress


@app.get("/progress")
async def get_progress(
    calibration_id: int | None = None,
    rebuild: bool = False,
) -> dict:
    db = app.state.db
    if rebuild or not list_intelligibility_snapshots(db, calibration_id):
        adapt_mod.rebuild_intelligibility_snapshots(db, calibration_id)
    snapshots = list_intelligibility_snapshots(db, calibration_id)
    shift = adapt_mod.detect_shift(db, calibration_id)
    return {
        "calibration_id": calibration_id,
        "snapshots": snapshots,
        "shift": shift,
    }


def _split_for_clip(
    clip_id: int,
    speaker_id: str,
    eval_frac: float = 0.2,
    seed: int = 42,
) -> str:
    """Deterministic cal/eval bucket for a TORGO clip.

    Mirror of ``scripts/splits.split_for_clip`` — duplicated here to
    avoid a cross-package import between ``app/`` and ``scripts/``. The
    SHA-256 hash and parameters MUST match the training script's split
    or the train/test labels in the UI will be wrong.
    """
    import hashlib

    key = f"{seed}|{speaker_id}|{clip_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / (1 << 64)
    return "eval" if bucket < eval_frac else "cal"


@app.get("/torgo/clips")
async def torgo_clips(
    speaker_id: str | None = None,
    condition: str | None = None,
    limit: int = 200,
) -> dict:
    clips = list_torgo_clips(app.state.db, speaker_id, condition, limit)
    # Decorate each clip with its train/eval split so the demo panel can
    # show users which clips a per-speaker adapter actually trained on.
    for c in clips:
        try:
            c["split"] = _split_for_clip(int(c["id"]), c["speaker_id"])
        except (TypeError, ValueError, KeyError):
            c["split"] = None
    return {"clips": clips}


@app.get("/torgo/audio/{clip_id}")
async def torgo_audio(clip_id: int) -> FileResponse:
    clip = get_torgo_clip(app.state.db, clip_id)
    if clip is None:
        raise HTTPException(404, f"TORGO clip {clip_id} not found")
    # ``Vary: Origin`` tells the browser to key its cache on Origin so a
    # no-CORS response (e.g. cached when an <audio> element loaded the
    # file without crossorigin set) is NOT reused for a later fetch()
    # that needs CORS headers. Without this, the browser silently
    # serves a stale Origin-less response to fetch() and the request
    # fails as "blocked by CORS policy" — only avoidable by disabling
    # the cache (e.g. devtools open). Cache-Control: no-cache forces
    # revalidation so fresh CORS headers re-attach on every request.
    return FileResponse(
        clip["wav_path"],
        media_type="audio/wav",
        headers={"Vary": "Origin", "Cache-Control": "no-cache"},
    )


@app.post("/speaker/enroll/patient")
async def enroll_patient(audio: UploadFile) -> dict:
    """Enroll the patient's voice from a 16 kHz mono WAV upload."""
    raw = await audio.read()
    try:
        samples, sr = sf.read(_BytesAdapter(raw), dtype="float32")
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}") from e

    if samples.ndim > 1:
        samples = samples[:, 0]
    if sr != settings.sample_rate:
        raise HTTPException(
            400, f"Expected {settings.sample_rate} Hz audio, got {sr} Hz"
        )
    if samples.size < settings.sample_rate:  # require >= 1 s
        raise HTTPException(400, "Need at least 1 second of audio for enrollment")

    speaker: SpeakerIdentifier = app.state.speaker
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, speaker.embed, samples)
    speaker.enroll_patient(app.state.db, embedding)
    return {
        "ok": True,
        "label": "patient",
        "embedding_dim": int(embedding.size),
        "sample_count": int(samples.size),
    }


_TORGO_ID_RE = __import__("re").compile(r"^[FM]C?\d+", __import__("re").IGNORECASE)


def _classify_adapter_dir(name: str) -> tuple[str, str, str]:
    """Map an adapter directory name to (id, label, kind).

    TORGO speakers follow ``F01`` / ``F03_balanced`` / ``FC02`` / ``M01_small_hicap``
    — first token is the canonical speaker id. Patient adapters use the
    full directory name as both id and (humanized) label so they don't
    collide with TORGO speakers and patient names with underscores or
    digits aren't mangled.
    """
    m = _TORGO_ID_RE.match(name)
    if m:
        sid = m.group(0).upper()
        return sid, sid, "torgo"
    # Patient adapter — humanize the slug for display, keep the raw dir
    # name as the routing id so the resolver can locate it via exact
    # filesystem match.
    label = name.replace("_", " ").strip() or name
    return name, label, "patient"


@app.get("/adapters")
async def list_adapters() -> dict:
    """List per-speaker adapters available on disk.

    Read-only; consumed by the home page's ProfileSelector + the clips
    drawer. The response carries enriched entries so the UI can
    distinguish TORGO-style speakers (F03 / FC02 / M01 — possibly with
    variant suffixes that all roll up to one id) from patient-trained
    adapters (full directory name is the id, prettified for display).

    Response shape::

        {
          "speakers": [
            {"id": "F03",       "label": "F03",      "kind": "torgo"},
            {"id": "Jane_Doe",  "label": "Jane Doe", "kind": "patient"},
            ...
          ],
          "adapters": [{"name": "F03_balanced", "speaker": "F03"}, ...]
        }

    Skips the unnamed default ``patient`` directory and the multi-speaker
    ``pool_*`` adapters — neither maps to a single user-selectable profile.
    """
    base = Path(settings.patient_adapter_dir).parent
    if not base.exists():
        return {"speakers": [], "adapters": []}

    speakers: dict[str, dict] = {}  # id -> {id, label, kind}
    adapters: list[dict] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "patient" or d.name.startswith("pool_"):
            continue
        if not (d / "adapter_model.safetensors").exists():
            continue
        sid, label, kind = _classify_adapter_dir(d.name)
        if sid not in speakers:
            speakers[sid] = {"id": sid, "label": label, "kind": kind}
        adapters.append({"name": d.name, "speaker": sid})

    # Sort: TORGO first (alphabetical), then patients (alphabetical by label).
    def _sort_key(s: dict) -> tuple:
        return (0 if s["kind"] == "torgo" else 1, s["id"])

    return {
        "speakers": sorted(speakers.values(), key=_sort_key),
        "adapters": adapters,
    }


@app.get("/speaker/patient")
async def get_patient_speaker() -> dict:
    """Return whether the patient voice has been enrolled.

    Read-only; consumed by the listener UI to show a calibration-status
    chip near the mic bar. The patient row lives at session_id IS NULL.

    Also reports whether the latest patient profile carries a prosody
    baseline (median jitter/shimmer/pauses across calibration utterances).
    The baseline lets emergency / emotion detectors threshold on deltas
    from this patient's normal speech rather than absolute values that
    fire constantly on dysarthric voices.
    """
    row = app.state.db.execute(
        "SELECT sample_count, created_at FROM speakers"
        " WHERE session_id IS NULL AND label = 'patient'"
    ).fetchone()

    has_baseline = False
    try:
        cur = app.state.db.execute(
            "SELECT profile_json FROM patient_profiles ORDER BY id DESC LIMIT 1"
        )
        latest = cur.fetchone()
        if latest and latest[0]:
            data = json.loads(latest[0])
            has_baseline = isinstance(data.get("prosody_baseline"), dict)
    except (json.JSONDecodeError, sqlite3.Error):
        pass

    if row is None:
        return {"enrolled": False, "baseline": has_baseline}
    return {
        "enrolled": True,
        "baseline": has_baseline,
        "sample_count": int(row[0]) if row[0] is not None else None,
        "enrolled_at": float(row[1]) if row[1] is not None else None,
    }


# ================================================================ WebSocket


@app.post("/demo/whisper/{clip_id}")
async def demo_whisper(clip_id: int) -> dict:
    """Return faster-whisper's raw transcription of a TORGO clip.

    Used as the baseline 'naive ASR' against which the calibrated
    Phonemer pipeline is compared on the /demo page.
    """
    clip = get_torgo_clip(app.state.db, clip_id)
    if clip is None:
        raise HTTPException(404, f"clip {clip_id} not found")

    audio, sr = sf.read(clip["wav_path"], dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != settings.sample_rate:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
        except Exception as e:
            raise HTTPException(500, f"resample failed: {e}") from e

    transcriber: Transcriber = app.state.transcriber
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, transcriber.transcribe, audio)
    return {
        "clip_id": clip_id,
        "speaker_id": clip["speaker_id"],
        "condition": clip["condition"],
        "duration_ms": clip["duration_ms"],
        "transcript": text,
    }


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    """Receive raw 16 kHz int16 PCM, segment by VAD, route by speaker.

    Replies (text JSON, one of):
      {"type": "utterance", "utterance_id": int, "speaker": str, "role": str|null,
       "is_patient": bool, "start_ms": int, "end_ms": int,
       "transcript"?: str}
      {"type": "phoneme_window", "utterance_id": int, "frames": [...]}
      {"type": "features", "utterance_id": int, "features": {...}}

    Query params:
      ?force_patient=1   bypass speaker-ID and treat every utterance as the
                         patient. Used by the /demo page to stream TORGO clips
                         through the calibrated pipeline regardless of who's
                         enrolled.
    """
    bypass_vad = (
        websocket.query_params.get("bypass_vad", "0") in ("1", "true", "yes")
    )
    force_patient = (
        websocket.query_params.get("force_patient", "0") in ("1", "true", "yes")
    )
    # When demo mode streams a TORGO clip we know the speaker id up front
    # (e.g. F03), so the runtime can pin the matching per-speaker adapter
    # for that session instead of the default. Falls back to the default
    # adapter if no matching directory exists on disk.
    requested_adapter_speaker = websocket.query_params.get("adapter_speaker")
    loaded_bases = set(getattr(websocket.app.state, "bases", {}).keys())
    adapter_override = _resolve_adapter_dir(
        requested_adapter_speaker, loaded_bases=loaded_bases
    )

    # Pick the gate that matches the resolved adapter's base model. Each
    # base loaded at lifespan has its own AdaptedTranscriber + gate; the
    # M01 medium adapter wants the medium gate, the F03 / FC02 small
    # adapters want the small gate. Falls back to the default gate when
    # no adapter is resolved (live mic without adapter_speaker) or the
    # adapter's base happens to match the default anyway.
    gate: GatedTranscriber = websocket.app.state.gate
    selected_base: str | None = None
    if adapter_override:
        try:
            cand_marker = Path(adapter_override) / "BASE_MODEL.txt"
            if cand_marker.exists():
                cand_base = cand_marker.read_text().strip()
                if cand_base in websocket.app.state.bases:
                    gate = websocket.app.state.bases[cand_base]["gate"]
                    selected_base = cand_base
        except OSError:
            pass

    await websocket.accept()
    logger.info(
        "WS /ws/audio accepted (force_patient=%s, adapter=%s, base=%s) from %s",
        force_patient,
        adapter_override or "default",
        selected_base or "default",
        websocket.client,
    )

    extractor: PhonemeExtractor = websocket.app.state.extractor
    speaker_id: SpeakerIdentifier = websocket.app.state.speaker
    transcriber: Transcriber = websocket.app.state.transcriber
    db = websocket.app.state.db

    session_id = create_session(db)
    logger.info("Session %d started", session_id)

    # Per-stream diagnostic timeline. Each WS connection gets its own
    # millisecond-anchored timeline that's emitted as ``diag`` messages
    # over the same WS. The frontend collects these per-stream so the
    # user can copy a compact, deterministic log instead of slogging
    # through the server's stdout (which is interleaved with model-load
    # noise, lifespan boilerplate, and any concurrent sessions).
    diag_t0 = time.time()

    async def diag(msg: str, level: str = "info", **data) -> None:
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "diag",
                        "session_id": session_id,
                        "ts_ms": int((time.time() - diag_t0) * 1000),
                        "level": level,
                        "msg": msg,
                        "data": data or None,
                    }
                )
            )
        except Exception:
            # WS may already be closed (e.g. client disconnected before
            # we finished an in-flight diag). Silent — these are
            # diagnostic-only, not load-bearing.
            pass

    await diag(
        "stream opened",
        force_patient=force_patient,
        adapter=adapter_override or "default",
        speaker_requested=requested_adapter_speaker or None,
        bypass_vad=bypass_vad,
    )

    # Load the most recent patient profile's prosody baseline (across all
    # calibrations). Used to threshold emergency alerts and emotional-state
    # computations on DELTAS from this patient's normal voice quality
    # rather than absolute values that are naturally elevated by
    # dysarthria. None when no calibration has been completed yet — in
    # that case the absolute fallback thresholds in check_prosody apply.
    prosody_baseline: dict | None = None
    try:
        cur = db.execute(
            "SELECT profile_json FROM patient_profiles ORDER BY id DESC LIMIT 1"
        )
        latest = cur.fetchone()
        if latest and latest[0]:
            data = json.loads(latest[0])
            pb = data.get("prosody_baseline")
            if isinstance(pb, dict):
                prosody_baseline = pb
                logger.info(
                    "Session %d loaded prosody baseline (n=%s, jitter=%.3f, shimmer=%.3f)",
                    session_id,
                    pb.get("n_samples"),
                    pb.get("jitter_p50", 0),
                    pb.get("shimmer_p50", 0),
                )
    except (json.JSONDecodeError, sqlite3.Error):
        logger.debug("could not load patient prosody baseline", exc_info=True)

    context = ContextManager(
        session_id=session_id,
        max_recent=settings.translation_recent_utterances,
        prosody_baseline=prosody_baseline,
    )
    upsert_session_metadata(
        db, session_id=session_id, time_of_day=context.metadata.time_of_day
    )

    segmenter = UtteranceSegmenter(
        vad=websocket.app.state.vad,
        sample_rate=settings.sample_rate,
        frame_ms=settings.vad_frame_ms,
        silence_close_ms=settings.silence_close_ms,
        min_utterance_ms=settings.min_utterance_ms,
        max_utterance_ms=settings.max_utterance_ms,
        pre_roll_ms=settings.pre_roll_ms,
    )
    frame_stride_ms = int(1000 * FRAME_STRIDE_SAMPLES / settings.sample_rate)
    loop = asyncio.get_running_loop()

    async def handle_utterance(utt: Utterance) -> None:
        if utt.audio.size == 0:
            return

        # Speaker routing first — cheapest signal, and the rejection path
        # short-circuits before we save audio or run heavy models. The /demo
        # page passes ?force_patient=1 so streamed TORGO clips skip this step
        # and always run the calibrated phoneme/LLM pipeline.
        if force_patient:
            from .speaker_id import SpeakerMatch
            match = SpeakerMatch(
                label="patient", role="patient", similarity=1.0, is_new=False
            )
        else:
            embedding = await loop.run_in_executor(
                None, speaker_id.embed, utt.audio
            )
            match = await loop.run_in_executor(
                None, speaker_id.classify, db, session_id, embedding
            )
        if match.rejected:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "utterance_rejected",
                        "session_id": session_id,
                        "start_ms": utt.start_ms,
                        "end_ms": utt.end_ms,
                        "speaker_similarity": match.similarity,
                        "reason": "non_speech",
                    }
                )
            )
            return

        # Real speech (or close enough) — persist audio + utterance row.
        name = f"sess{session_id}_{utt.start_ms:08d}_{int(time.time()*1000)}"
        audio_path = save_wav(utt.audio, settings.sample_rate, settings.audio_dir, name)
        is_patient = match.label == "patient"
        # Stamp the active adapter slug on patient utterances so the
        # trainer can later pull confirmed/corrected ones back into the
        # training set for the same patient (feedback retraining).
        utt_adapter_slug = (
            requested_adapter_speaker
            if is_patient and requested_adapter_speaker
            else None
        )
        utt_id = create_utterance(
            db,
            session_id=session_id,
            start_ms=utt.start_ms,
            end_ms=utt.end_ms,
            speaker_label=match.label,
            speaker_role=match.role,
            speaker_similarity=match.similarity,
            is_patient=is_patient,
            audio_path=audio_path,
            adapter_slug=utt_adapter_slug,
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "utterance",
                    "utterance_id": utt_id,
                    "session_id": session_id,
                    "speaker": match.label,
                    "role": match.role,
                    "is_patient": is_patient,
                    "speaker_similarity": match.similarity,
                    "start_ms": utt.start_ms,
                    "end_ms": utt.end_ms,
                }
            )
        )

        if is_patient:
            await _process_patient(
                websocket, loop, extractor, gate, db, context, session_id,
                utt_id, utt, frame_stride_ms, adapter_dir=adapter_override,
            )
        else:
            await _process_other(websocket, loop, transcriber, db, context, session_id, utt_id, utt)

        # Stream the freshly-updated context snapshot so /debug can render it.
        try:
            await websocket.send_text(
                json.dumps({"type": "context", "session_id": session_id, "context": context.snapshot_for_ui()})
            )
        except Exception:
            logger.debug("failed to emit context snapshot", exc_info=True)

    bytes_received = 0
    # When ``bypass_vad=1`` (TORGO panel use case: the streamed WAV file is
    # known to be a single intended utterance), we skip the silero VAD
    # segmenter entirely. PCM bytes accumulate into ``vad_bypass_buf`` and
    # the whole buffer is processed as one Utterance in the finally block.
    # Solves: silero VAD was non-deterministic on chunked realtime streams
    # of dysarthric speech — same TORGO clip, different VAD splits across
    # runs, producing wildly different transcripts (e.g. M01 "The islands
    # are sparsely populated" → "silent sun" in one run, full sentence in
    # the next). For pre-recorded clips we already know the full audio is
    # one utterance, so VAD is the wrong layer.
    vad_bypass_buf = bytearray() if bypass_vad else None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            raw = message.get("bytes")
            if raw is not None:
                if bytes_received == 0:
                    logger.info("WS session %d first PCM bytes arrived (%d B)",
                                session_id, len(raw))
                    await diag(
                        "first PCM bytes arrived", bytes=len(raw),
                        bypass_vad=bypass_vad,
                    )
                bytes_received += len(raw)
                if vad_bypass_buf is not None:
                    vad_bypass_buf.extend(raw)
                else:
                    segmenter.append_pcm16(raw)
                    for utt in list(segmenter.utterances()):
                        logger.info("WS session %d emitting utterance %d–%d ms (%.1fs audio)",
                                    session_id, utt.start_ms, utt.end_ms,
                                    utt.audio.size / settings.sample_rate)
                        await diag(
                            "VAD emitted utterance",
                            start_ms=utt.start_ms, end_ms=utt.end_ms,
                            duration_ms=int(utt.audio.size / settings.sample_rate * 1000),
                        )
                        await handle_utterance(utt)
                continue

            text = message.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "stop":
                    logger.info("WS session %d received stop after %d B",
                                session_id, bytes_received)
                    await diag("stop received", bytes=bytes_received)
                    break
    except WebSocketDisconnect:
        logger.info("WS session %d disconnected after %d B", session_id, bytes_received)
    except Exception:
        logger.exception("Unexpected error in session %d", session_id)
    finally:
        flushed = 0
        if vad_bypass_buf is not None and len(vad_bypass_buf) > 0:
            # Bypass path: emit the entire accumulated PCM buffer as a
            # single utterance. No VAD trimming — caller (TORGO panel)
            # guaranteed the buffer is one intended utterance.
            audio = np.frombuffer(bytes(vad_bypass_buf), dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            duration_ms = int(audio.size / settings.sample_rate * 1000)
            utt = Utterance(start_ms=0, end_ms=duration_ms, audio=audio)
            try:
                flushed = 1
                logger.info(
                    "WS session %d bypass_vad — single utterance %.1fs (%d B PCM)",
                    session_id, duration_ms / 1000, len(vad_bypass_buf),
                )
                await diag(
                    "bypass_vad: emitting single utterance",
                    duration_ms=duration_ms, bytes=len(vad_bypass_buf),
                )
                await handle_utterance(utt)
            except Exception:
                logger.exception("Failed to process bypass_vad utterance")
        else:
            for utt in list(segmenter.flush()):
                try:
                    flushed += 1
                    logger.info("WS session %d flushing trailing utterance %d–%d ms",
                                session_id, utt.start_ms, utt.end_ms)
                    await diag(
                        "flushing trailing utterance",
                        start_ms=utt.start_ms, end_ms=utt.end_ms,
                        duration_ms=int(utt.audio.size / settings.sample_rate * 1000),
                    )
                    await handle_utterance(utt)
                except Exception:
                    logger.exception("Failed to flush trailing utterance")
        logger.info("WS session %d closing (received %d B, flushed %d trailing utterance(s))",
                    session_id, bytes_received, flushed)
        try:
            await diag(
                "stream closing", bytes=bytes_received,
                trailing_utterances=flushed,
            )
        except Exception:
            pass
        # Mark any unanswered questions as stale.
        mark_questions_answered(db, session_id)
        end_session(db, session_id)
        logger.info("Session %d ended", session_id)
        try:
            await websocket.close()
        except RuntimeError:
            pass


# ============================================================== sub-pipelines


async def _process_patient(
    websocket: WebSocket,
    loop: asyncio.AbstractEventLoop,
    extractor: PhonemeExtractor,
    gate: GatedTranscriber,
    db,
    context: ContextManager,
    session_id: int,
    utterance_id: int,
    utt: Utterance,
    frame_stride_ms: int,
    adapter_dir: str | None = None,
) -> None:
    # Phoneme distributions over the whole utterance.
    probs = await loop.run_in_executor(
        None, extractor.extract, utt.audio, settings.sample_rate
    )
    top_k = extractor.top_k_per_frame(probs, settings.top_k_phonemes)

    response_frames: list[dict] = []
    persist_rows: list[tuple[int, list[dict]]] = []
    for i, frame in enumerate(top_k):
        ts = utt.start_ms + i * frame_stride_ms
        dist = [{"symbol": s, "prob": p} for s, p in frame]
        response_frames.append({"timestamp_ms": ts, "phonemes": dist})
        persist_rows.append((ts, dist))
    write_frames(db, session_id, utterance_id, persist_rows)

    await websocket.send_text(
        json.dumps(
            {
                "type": "phoneme_window",
                "utterance_id": utterance_id,
                "frames": response_frames,
            }
        )
    )

    # Acoustic features (Layer 2) — useful for distortion model + fatigue tracking.
    features = await loop.run_in_executor(
        None, feat_mod.extract, utt.audio, settings.sample_rate
    )
    payload = features.to_json()
    write_features(db, utterance_id, payload)
    await websocket.send_text(
        json.dumps(
            {
                "type": "features",
                "utterance_id": utterance_id,
                "features": payload,
            }
        )
    )

    # Update emotional-state tracker from prosody (per spec — never word content).
    context.observe_features(payload)
    emotion = context.emotion.snapshot()
    write_emotion_snapshot(db, session_id, emotion["frustration"], emotion["distress"])

    # Pre-LLM emergency check on prosody alone — fires even if translation fails.
    prosody_alert = check_prosody(
        payload, emotion, baseline=context.get_prosody_baseline()
    )
    if prosody_alert.triggered:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "emergency",
                    "utterance_id": utterance_id,
                    "phrase": prosody_alert.phrase,
                    "confidence": prosody_alert.confidence,
                    "category": prosody_alert.category,
                    "sources": prosody_alert.sources,
                }
            )
        )

    # Layer 5 — Whisper + phoneme-alignment safety gate.
    # See JOURNAL.md entries 12-13 for why this replaces the calibrated LLM
    # path: empirical eval showed the LLM correction layer can't beat naive
    # Whisper, but the phoneme distribution does reliably catch Whisper's
    # hallucination failure mode. We surface Whisper's text and flag clips
    # where alignment evidence doesn't support it.
    try:
        await _gate_and_emit(
            websocket, loop, gate, db, context, session_id, utterance_id,
            utt, probs, adapter_dir=adapter_dir,
        )
    except WebSocketDisconnect:
        # Client closed the socket while the gate was still decoding /
        # the executor was finishing. This is the normal "stop mic before
        # the in-flight utterance finishes" flow. Log briefly without a
        # stack trace so it doesn't look like a real failure.
        logger.info(
            "WS closed before utt=%s could be emitted (in-flight cancel)",
            utterance_id,
        )
    except Exception as e:
        logger.exception("gated transcription failed for utt=%s", utterance_id)
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "translation_error",
                        "utterance_id": utterance_id,
                        "error": type(e).__name__,
                        "message": str(e)[:500],
                    }
                )
            )
        except Exception:
            pass


async def _gate_and_emit(
    websocket: WebSocket,
    loop: asyncio.AbstractEventLoop,
    gate: GatedTranscriber,
    db,
    context_mgr: ContextManager,
    session_id: int,
    utterance_id: int,
    utt: Utterance,
    probs: np.ndarray,
    adapter_dir: str | None = None,
) -> None:
    """Run Whisper + phoneme-alignment gate, persist, emit to client.

    The gate emits a ``translation`` message whose new ``flagged`` field
    drives the listener UI: red border + low-trust hint when the audio
    doesn't support what Whisper proposed. The text is always surfaced
    — hiding it would defeat the medical-safety story.
    """
    # Build the conversational/vocabulary prior for Whisper. Recent
    # listener turn + active topics + caregiver-supplied personal vocab
    # (family names, medications) get fed in as initial_prompt so
    # Whisper has *some* context for borderline calls. This is the
    # well-validated ASR-conditioning pattern, not the failed
    # phoneme-correction architecture.
    vocab_text = vocab_mod.vocabulary_prompt_block(db)
    initial_prompt = context_mgr.whisper_prompt(vocab_text=vocab_text)
    gated = await loop.run_in_executor(
        None,
        lambda: gate.transcribe_with_gate(
            utt.audio,
            probs=probs,
            initial_prompt=initial_prompt,
            adapter_dir=adapter_dir,
        ),
    )
    # Real per-word + overall confidence derived from the gate's
    # phoneme-alignment scores — the geometric mean per-frame phoneme
    # probability for each word, i.e. exp(mean log-prob). Words whose
    # phonemes all got dropped by the vocab filter (rare, after the
    # compound-split rules in vocab.py these are mostly proper nouns)
    # fall back to the overall confidence rather than 0.0 — reporting
    # 0% would imply "definitely wrong" when the truth is "we have no
    # signal for this word."
    overall_conf = (
        max(0.0, min(1.0, math.exp(gated.align_score)))
        if math.isfinite(gated.align_score) else 0.0
    )
    per_word_conf = [
        max(0.0, min(1.0, math.exp(s))) if math.isfinite(s) else overall_conf
        for s in gated.per_word_scores
    ]
    # UI-level low-confidence override. The phoneme-alignment gate can be
    # fooled by under-production cases where a short transcript phonetically
    # aligns to *part* of the audio while leaving most of it unaccounted
    # for (diagnostic: "The islands are sparsely populated" → "the island
    # arts.", align_score -0.2, dur 2.85 s, only 11 phonemes emitted).
    # The per-word/overall confidences computed above *do* see the issue
    # — they're uniformly mediocre on these cases. Promote that signal
    # into the same `flagged` field the UI consumes for the red ring, so
    # listeners get the low-trust treatment even when the gate didn't
    # structurally trip.
    # Per-patient UI low-confidence threshold. When the active adapter
    # carries a ``calibration.json``, use its
    # ``ui_low_confidence_threshold`` instead of the global default —
    # M01's natural overall_conf distribution sits much lower than
    # FC02's, so a single global threshold over- or under-flags one of
    # them. The gate exposes the patient-specific value via
    # ``get_calibration_thresholds``; fall back to global when not set.
    # When no explicit ``adapter_dir`` is set (live mic stream), consult
    # the default patient adapter's calibration so the live path also
    # gets per-patient thresholds.
    cal_thresholds = gate.get_calibration_thresholds(
        adapter_dir or settings.patient_adapter_dir
    )
    ui_low_conf_threshold = float(
        cal_thresholds.get(
            "ui_low_confidence_threshold",
            settings.ui_low_confidence_threshold,
        )
    )
    ui_flagged = gated.flagged
    ui_reason = gated.reason
    if (
        not ui_flagged
        and overall_conf < ui_low_conf_threshold
        and gated.text
    ):
        ui_flagged = True
        ui_reason = "low_confidence"
    write_translation(
        db,
        utterance_id=utterance_id,
        primary=gated.text,
        alternatives=[],
        per_word_confidence=per_word_conf,
        overall_confidence=overall_conf,
        emergency=False,
        context_hints=[
            f"flagged: {ui_reason}" if ui_reason else "phoneme-aligned"
        ],
        model="whisper+phoneme-gate",
        usage={
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "translation",
                "utterance_id": utterance_id,
                "primary": gated.text,
                "alternatives": [],
                "per_word_confidence": per_word_conf,
                "overall_confidence": overall_conf,
                "emergency": False,
                "context_hints": [],
                # Frontend renders red border when flagged. We forward the
                # UI-promoted `ui_flagged` here (the gate's verdict OR the
                # low-confidence override), not the raw gate field.
                "flagged": ui_flagged,
                "reason": ui_reason,
                "align_score": gated.align_score,
                "phoneme_density": gated.phoneme_density,
            }
        )
    )

    # Only feed un-flagged text into the conversational context window.
    # A flagged transcription is by definition untrusted; we shouldn't
    # let it bias the topic tracker or pending-question matching.
    # Uses the UI-promoted flag so low-confidence outputs are likewise
    # withheld from the topic / question trackers.
    if not ui_flagged and gated.text:
        context_mgr.observe_patient(gated.text)

    # Layer-6 emergency cross-check on text. Priority terms (pain, help,
    # can't breathe) get a lower threshold and fire even when flagged —
    # missed emergencies are worse than false alarms.
    text_alert = check_text(
        primary=gated.text,
        alternatives=[],
        overall_confidence=overall_conf,
    )
    if text_alert.triggered:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "emergency",
                    "utterance_id": utterance_id,
                    "phrase": gated.text,
                    "confidence": text_alert.confidence,
                    "category": text_alert.category or "text",
                    "sources": text_alert.sources or ["text"],
                }
            )
        )


def _recent_for_context(db, session_id: int) -> list[dict]:
    rows = list_recent_patient_utterances(
        db, session_id, settings.translation_recent_utterances
    )
    out: list[dict] = []
    for r in rows:
        text = r.get("primary_text") or r.get("transcript") or ""
        if not text:
            continue
        out.append(
            {
                "speaker": "patient" if r.get("is_patient") else r.get("speaker_label"),
                "text": text,
            }
        )
    return out


async def _process_other(
    websocket: WebSocket,
    loop: asyncio.AbstractEventLoop,
    transcriber: Transcriber,
    db,
    context: ContextManager,
    session_id: int,
    utterance_id: int,
    utt: Utterance,
) -> None:
    transcript = await loop.run_in_executor(None, transcriber.transcribe, utt.audio)
    if transcript:
        set_utterance_transcript(db, utterance_id, transcript)
        # Feed non-patient transcript into the six-layer context window.
        # ``context.observe_non_patient`` updates topics, questions, role
        # inference, and the utterance log atomically.
        speaker_label = _label_from_utterance(db, utterance_id) or "speaker_?"
        context.observe_non_patient(speaker_label, transcript)

        # Persist topic + question observations.
        for topic in context.topics.active(k=5):
            write_topic_observation(db, session_id, topic["keyword"], topic["weight"])
        qtype = classify_question(transcript)
        if qtype:
            write_pending_question(db, session_id, transcript, qtype)
        # Update session metadata if the inferred listener role changed.
        upsert_session_metadata(
            db,
            session_id=session_id,
            listener_role_hint=context.metadata.listener_role_hint,
        )
    await websocket.send_text(
        json.dumps(
            {
                "type": "utterance_transcript",
                "utterance_id": utterance_id,
                "transcript": transcript,
            }
        )
    )


def _label_from_utterance(db, utterance_id: int) -> str | None:
    cur = db.execute(
        "SELECT speaker_label FROM utterances WHERE id = ?", (utterance_id,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def _resolve_adapter_dir(
    speaker_id: str | None,
    loaded_bases: set[str] | None = None,
) -> str | None:
    """Map a TORGO speaker id to its preferred adapter directory.

    Looks under the parent of ``settings.patient_adapter_dir`` (i.e.
    ``data/adapters/``) and matches directories whose leading speaker
    token equals ``speaker_id`` (case-insensitive). Filters by base-
    model compatibility: only adapters whose ``BASE_MODEL.txt`` is in
    ``loaded_bases`` are considered (an adapter whose base isn't loaded
    can't be served, and PEFT shape-mismatch silently degrades to the
    default — see the legacy whisper-tiny.en M01 misload incident).

    ``loaded_bases`` defaults to ``{<live default base>}`` to preserve
    the old single-base behavior for callers that haven't been updated
    to the multi-base lifespan. With multi-base loaded (e.g. small.en
    + medium.en), this allows M01 streams to pick up the medium
    adapter while F03 / FC02 stay on small adapters.

    When multiple compatible candidates exist, prefers the most
    recently-trained one (mtime descending). Returns ``None`` if no
    compatible adapter is on disk — callers fall back to the default.
    """
    if not speaker_id:
        return None
    base = Path(settings.patient_adapter_dir).parent
    if not base.exists():
        return None

    if loaded_bases is None or not loaded_bases:
        # Backward-compat: single-base mode. Read the default's marker.
        live_base: str | None = None
        default_marker = Path(settings.patient_adapter_dir) / "BASE_MODEL.txt"
        if default_marker.exists():
            try:
                live_base = default_marker.read_text().strip() or None
            except OSError:
                live_base = None
        loaded_bases = {live_base} if live_base else set()

    def _base_ok(d: Path) -> bool:
        if not loaded_bases:
            # No constraint — accept anything.
            return True
        cand_marker = d / "BASE_MODEL.txt"
        if not cand_marker.exists():
            # Legacy adapter without marker: accept (failure mode
            # would surface as a PEFT shape error, which we'll log).
            return True
        try:
            return cand_marker.read_text().strip() in loaded_bases
        except OSError:
            return False

    # No fast-path "exact match". An exact bare-name match (e.g.
    # ``data/adapters/M01`` referring to a *legacy* whisper-tiny.en
    # adapter) would short-circuit the newest-wins loop below and
    # return the worse adapter when a newer suffixed sibling like
    # ``M01_medium_hicap`` exists. We always enumerate candidates and
    # sort by mtime.
    speaker_upper = speaker_id.upper()
    candidates: list[Path] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name == "patient" or d.name.startswith("pool_"):
            continue
        if not (d / "adapter_model.safetensors").exists():
            continue
        if d.name.split("_", 1)[0].upper() != speaker_upper:
            continue
        if not _base_ok(d):
            logger.debug(
                "Skipping adapter %s (base not in loaded_bases=%s)",
                d, loaded_bases,
            )
            continue
        candidates.append(d)
    if not candidates:
        return None
    candidates.sort(
        key=lambda d: (d / "adapter_model.safetensors").stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0])


# =============================================================== utilities


class _BytesAdapter:
    """Tiny adapter so soundfile can read from raw bytes without a temp file."""

    def __init__(self, data: bytes) -> None:
        import io

        self._buf = io.BytesIO(data)

    def __getattr__(self, name):
        return getattr(self._buf, name)
