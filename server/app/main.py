"""FastAPI app: VAD-gated, speaker-routed phoneme + STT pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
    create_utterance,
    end_session,
    get_torgo_clip,
    latest_patient_profile,
    list_calibration_utterances,
    list_recent_patient_utterances,
    list_torgo_clips,
    mark_questions_answered,
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
from .llm import TranslationContext, summarize_profile_for_prompt
from .phonemes import FRAME_STRIDE_SAMPLES, PhonemeExtractor
from .silero_vad import make_vad
from .speaker_id import SpeakerIdentifier, ensure_schema
from .stt import Transcriber
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

    app.state.transcriber = Transcriber(
        model_size=settings.stt_model_size,
        compute_type=settings.stt_compute_type,
        device=settings.device,
    )

    app.state.db = connect(settings.db_path)
    ensure_schema(app.state.db)
    Path(settings.audio_dir).mkdir(parents=True, exist_ok=True)
    logger.info("SQLite ready at %s", settings.db_path)

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
    return {
        "ok": True,
        "model": settings.phoneme_model_name,
        "speaker_model": settings.speaker_model_source,
        "stt_model": settings.stt_model_size,
        "sample_rate": settings.sample_rate,
    }


_CAL_SENTENCES_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration_sentences.json"


def _load_calibration_sentence_sets() -> dict:
    return json.loads(_CAL_SENTENCES_PATH.read_text())


@app.get("/calibration/sentences")
async def calibration_sentences() -> dict:
    return _load_calibration_sentence_sets()


@app.post("/calibration/start")
async def calibration_start() -> dict:
    cid = create_calibration(app.state.db)
    return {
        "calibration_id": cid,
        "sentence_sets": _load_calibration_sentence_sets()["sets"],
    }


@app.get("/calibration/{calibration_id}")
async def calibration_get(calibration_id: int) -> dict:
    utterances = list_calibration_utterances(app.state.db, calibration_id)
    profile = latest_patient_profile(app.state.db, calibration_id)
    return {
        "calibration_id": calibration_id,
        "utterances": utterances,
        "profile": profile,
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

    profile = build_profile(aligned_utterances, vocab_symbols=vocab)
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


@app.get("/torgo/clips")
async def torgo_clips(
    speaker_id: str | None = None,
    condition: str | None = None,
    limit: int = 200,
) -> dict:
    return {"clips": list_torgo_clips(app.state.db, speaker_id, condition, limit)}


@app.get("/torgo/audio/{clip_id}")
async def torgo_audio(clip_id: int) -> FileResponse:
    clip = get_torgo_clip(app.state.db, clip_id)
    if clip is None:
        raise HTTPException(404, f"TORGO clip {clip_id} not found")
    return FileResponse(clip["wav_path"], media_type="audio/wav")


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
    force_patient = (
        websocket.query_params.get("force_patient", "0") in ("1", "true", "yes")
    )
    await websocket.accept()
    logger.info(
        "WS /ws/audio accepted (force_patient=%s) from %s",
        force_patient,
        websocket.client,
    )

    extractor: PhonemeExtractor = websocket.app.state.extractor
    speaker_id: SpeakerIdentifier = websocket.app.state.speaker
    transcriber: Transcriber = websocket.app.state.transcriber
    db = websocket.app.state.db

    session_id = create_session(db)
    logger.info("Session %d started", session_id)

    context = ContextManager(session_id=session_id, max_recent=settings.translation_recent_utterances)
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
                websocket, loop, extractor, db, context, session_id, utt_id, utt, frame_stride_ms
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
                bytes_received += len(raw)
                segmenter.append_pcm16(raw)
                for utt in list(segmenter.utterances()):
                    logger.info("WS session %d emitting utterance %d–%d ms (%.1fs audio)",
                                session_id, utt.start_ms, utt.end_ms,
                                utt.audio.size / settings.sample_rate)
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
                    break
    except WebSocketDisconnect:
        logger.info("WS session %d disconnected after %d B", session_id, bytes_received)
    except Exception:
        logger.exception("Unexpected error in session %d", session_id)
    finally:
        flushed = 0
        for utt in list(segmenter.flush()):
            try:
                flushed += 1
                logger.info("WS session %d flushing trailing utterance %d–%d ms",
                            session_id, utt.start_ms, utt.end_ms)
                await handle_utterance(utt)
            except Exception:
                logger.exception("Failed to flush trailing utterance")
        logger.info("WS session %d closing (received %d B, flushed %d trailing utterance(s))",
                    session_id, bytes_received, flushed)
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
    db,
    context: ContextManager,
    session_id: int,
    utterance_id: int,
    utt: Utterance,
    frame_stride_ms: int,
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
    prosody_alert = check_prosody(payload, emotion)
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

    # Layer 5 — LLM translation. Skipped if no Anthropic API key is configured;
    # /debug still works as a phoneme inspector.
    if not llm.is_configured():
        return
    try:
        await _translate_and_emit(
            websocket, loop, db, context, session_id, utterance_id, response_frames
        )
    except Exception as e:
        logger.exception("translation failed for utt=%s", utterance_id)
        # Surface the failure to the client so the UI can show why no
        # translation arrived (most often: invalid API key, no credits,
        # or rate-limit). Without this the page just hangs forever.
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


async def _translate_and_emit(
    websocket: WebSocket,
    loop: asyncio.AbstractEventLoop,
    db,
    context_mgr: ContextManager,
    session_id: int,
    utterance_id: int,
    phoneme_frames: list[dict],
) -> None:
    profile_summary = _profile_summary_for(db)
    ctx_dict = context_mgr.assemble_for_llm()
    translation_ctx = TranslationContext(
        phoneme_frames=phoneme_frames,
        recent_utterances=ctx_dict["recent_utterances"],
        active_topics=ctx_dict["active_topics"],
        pending_questions=ctx_dict["pending_questions"],
        fatigue_phase=ctx_dict["emotion"].get("phase"),
        setting=ctx_dict["session"].get("setting"),
        listener_role=ctx_dict["session"].get("listener_role_hint"),
    )
    result = await loop.run_in_executor(
        None,
        lambda: llm.translate(
            model=settings.translation_model,
            profile_summary=profile_summary,
            context=translation_ctx,
            max_tokens=settings.translation_max_tokens,
        ),
    )
    write_translation(
        db,
        utterance_id=utterance_id,
        primary=result.primary,
        alternatives=result.alternatives,
        per_word_confidence=result.per_word_confidence,
        overall_confidence=result.overall_confidence,
        emergency=result.emergency,
        context_hints=result.context_hints,
        model=settings.translation_model,
        usage={
            "cache_read_tokens": result.cache_read_tokens,
            "cache_creation_tokens": result.cache_creation_tokens,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "translation",
                "utterance_id": utterance_id,
                "primary": result.primary,
                "alternatives": result.alternatives,
                "per_word_confidence": result.per_word_confidence,
                "overall_confidence": result.overall_confidence,
                "emergency": result.emergency,
                "context_hints": result.context_hints,
            }
        )
    )

    # Feed the patient's translation back into the context window.
    context_mgr.observe_patient(result.primary)

    # Layer-6 emergency cross-check on text — fires even when the LLM
    # didn't flag it, since priority terms get a lower threshold.
    text_alert = check_text(
        primary=result.primary,
        alternatives=result.alternatives,
        overall_confidence=result.overall_confidence,
    )
    if result.emergency or text_alert.triggered:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "emergency",
                    "utterance_id": utterance_id,
                    "phrase": result.primary,
                    "confidence": max(
                        result.overall_confidence,
                        text_alert.confidence,
                    ),
                    "category": text_alert.category or "llm_flagged",
                    "sources": text_alert.sources or ["llm"],
                }
            )
        )


def _profile_summary_for(db) -> str:
    """Effective profile (baseline + EMA-decayed online observations) plus
    the personal vocabulary block. Empty string when no calibration exists."""
    cal_id = _latest_calibration_id(db)
    parts: list[str] = []
    eff = adapt_mod.effective_profile_summary(db, cal_id)
    if eff is None:
        cur = db.execute(
            "SELECT profile_json FROM patient_profiles ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            try:
                parts.append(summarize_profile_for_prompt(json.loads(row[0])))
            except Exception:
                pass
    else:
        parts.append(eff)
    vocab_block = vocab_mod.vocabulary_prompt_block(db)
    if vocab_block:
        parts.append("\n" + vocab_block)
    return "\n".join(parts)


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


# =============================================================== utilities


class _BytesAdapter:
    """Tiny adapter so soundfile can read from raw bytes without a temp file."""

    def __init__(self, data: bytes) -> None:
        import io

        self._buf = io.BytesIO(data)

    def __getattr__(self, name):
        return getattr(self._buf, name)
