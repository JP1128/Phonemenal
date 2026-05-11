# Phonemer — Implementation Plan

Live speech translator for brain injury patients with speech impediments.

## Phase 1 — Backend: Audio Pipeline

### 1.1 Voice Activity Detection (Layer 1)
- `webrtcvad`. New `server/app/vad.py` with `UtteranceSegmenter`: 30 ms frames, 400 ms silence-close, pre-roll buffer.
- `utterances` table: `id, session_id, start_ms, end_ms, speaker_id, audio_path`.

### 1.2 Acoustic Feature Extraction (Layer 2)
- `librosa`. `server/app/features.py`: MFCC (13), formants F1–F3, VOT proxy, pitch contour (`librosa.pyin`), jitter/shimmer, spectral flux, pause durations.
- Per-utterance JSON in `utterance_features`.

### 1.3 Speaker Identification
- `speechbrain` ECAPA-TDNN (`spkrec-ecapa-voxceleb`), 192-dim cosine match → `"patient"` / `"speaker_<n>"`.
- `speakers` table; patient is pre-enrolled via `POST /speaker/enroll/patient`.
- WS routes patient utterances to phoneme pipeline; others to STT.

### 1.4 Non-Patient STT (faster-whisper)
- `faster-whisper` (CTranslate2) — 4–6× faster than openai-whisper on CPU, smaller memory.
- `server/app/stt.py`: lazy-loaded `Transcriber`. Output feeds context window.

---

## Phase 1.5 — Load TORGO Dataset

Real impaired speech available before any later layer; synthetic audio gives false confidence.

### 1.5.1 Dataset Acquisition
- Document Kaggle download path; `server/data/torgo/` layout.
- `server/scripts/torgo_loader.py`: walks dataset, parses `(speaker_id, intelligibility_level, intended_text, wav_path)`, normalizes to 16 kHz mono.

### 1.5.2 TORGO Manifest
- `torgo_manifest` table: `speaker_id, condition, intended_text, intended_phonemes, wav_path, duration_ms`.

### 1.5.3 TORGO Smoke Test
- `server/scripts/torgo_smoke.py`: pick N random clips, run through `/ws/audio`, assert non-degenerate phoneme distributions.
- Becomes regression harness used by every later phase.

---

## Phase 1.7 — Speech-vs-Noise Hardening (queued, before Phase 3)

webrtcvad's GMM lets keyboard/mouse/paper transients through as utterances.
Harmless during Phase 1/2 development but becomes wasted Sonnet tokens once
LLM translation is online, so this lands before Phase 3.

### 1.7.1 Replace webrtcvad with Silero VAD
- Add `silero-vad` ONNX model (~1.5 MB) to the model cache.
- New `app/silero_vad.py` wrapping the ONNX session; same `is_speech(frame)`
  contract as the current webrtcvad helper, drop-in replacement inside
  `UtteranceSegmenter`.
- Keep `webrtcvad` available behind a feature flag for A/B comparison.

### 1.7.2 ECAPA-based non-speech rejection
- After speaker-ID classifies, if the cosine similarity to *every* enrolled
  voice (patient + session speakers) is below a hard floor (default 0.15),
  drop the utterance entirely — don't write to `utterances`, don't run
  phoneme/feature extraction, don't run STT.
- Emit `{"type":"utterance_rejected", reason:"non_speech", similarity:…}`
  so `/debug` can show suppressed events without flooding.

### 1.7.3 Heatmap-based sanity check
- Add a regression test: synthesize 1 s of white noise + click train; assert
  the new pipeline emits zero utterances (or only `utterance_rejected`).

---

## Phase 1.6 — Frontend Debug Surface

Bare-bones inspector page that shows every backend message in real time. Polish comes in Phases 6/7. Code is reused unchanged later.

### 1.6.1 Mic Capture
- `client/lib/audio-capture.ts`: `getUserMedia` with `echoCancellation/noiseSuppression/autoGainControl: false`.
- `client/public/pcm-worklet.js`: AudioWorklet → Float32 → Int16 16 kHz, post every 100 ms.

### 1.6.2 WebSocket Client
- `client/lib/phonemer-ws.ts`: `connect()`, `sendChunk(Int16Array)`, typed `onMessage` for full set: `utterance / phoneme_window / features / utterance_transcript / translation / emergency`. Auto-reconnect with backoff.

### 1.6.3 Debug Inspector Page (`/debug`)
- Three columns:
  - **Stream log** — every message with timestamp, color-coded by type. Pause + clear.
  - **Active utterance** — speaker label, is_patient, ms range, top-5 phonemes per frame as heatmap, feature panel.
  - **Speaker registry** — list with similarity scores; "enroll patient" button.
- Mic on/off, levels meter, WS connection state badge.
- File-input fallback: drop in a WAV (e.g. TORGO clip).

### 1.6.4 TORGO Drop-down
- `GET /torgo/clips` exposes manifest. Pick clip + "stream" button. Same code path as mic.

### 1.6.5 Layer-Status Strip
- Top of page: green/red dots per layer (`VAD`, `Speaker ID`, `Phonemes`, `Features`, `STT`, later `LLM`). Each dot turns green the first time a corresponding message arrives.

### 1.6.6 Session Record / Replay
- Toolbar: ⏺ Record · ⏹ Stop · 💾 Download.
- `client/lib/recorder.ts`: subscribes to WS stream, appends `{t_ms, payload}` events with a header (`version, started_at, session_id, source, events`).
- Optional checkbox: include int16 PCM the client sent (base64 chunks) for round-trip replay.
- `client/lib/player.ts`: drag-and-drop a recording → "replay mode" banner. Speed controls (0.5×/1×/2×/4×/MAX), scrubber with utterance markers, ▶︎ Replay / 🔌 Live toggle.
- `server/scripts/replay_session.py`: re-streams captured PCM into `/ws/audio` for backend regression tests.

---

## Phase 2 — Calibration & Distortion Model (Layer 4)

### 2.1 Calibration Endpoint
- `POST /calibration/start` → `{calibration_id, sentence_sets}` (4 sets from `server/data/calibration_sentences.json`).
- WS `/ws/calibration/{id}` receives audio per sentence, tagged with intended text.
- `calibrations`, `calibration_utterances` tables.

### 2.2 Intended-Phoneme Alignment + Vocabulary Verification (REQUIRED GATE)
- `phonemizer` (espeak-ng backend). `text_to_phonemes(sentence)` → IPA.
- **Verification gate:** `server/scripts/verify_phoneme_vocab.py` + `server/tests/test_vocab_alignment.py`:
  - Load wav2vec2 tokenizer vocab (`facebook/wav2vec2-lv-60-espeak-cv-ft`).
  - Phonemize probe sentences with espeak-ng.
  - Diff symbol sets. Fail loudly with missing-symbol report; emit `phoneme_alias_map.json` for spelling/format gaps (e.g. `ɹ` vs `r`, length marks, ties).
- Distortion model must not run until this passes.
- After verification: CTC forced alignment via `torchaudio.functional.forced_align`.

### 2.3 Distortion Model Builder
- `server/app/distortion.py`: `PatientDistortionProfile` with confusion matrix `P(produced | intended)`, positional rules, cluster simplification, syllable deletion, fatigue priors.
- `patient_profiles` table.
- `POST /calibration/{id}/build` triggers Claude Opus analysis (Step 3.3) and writes profile.
- **Validate against TORGO:** build profile per TORGO speaker; verify produced/intended distributions match known intelligibility levels.

### 2.4 Fatigue Curves
- `server/app/fatigue.py`: per-utterance error density bucketed `fresh / mild / moderate / severe`.
- `fatigue_snapshots` table.

---

## Phase 3 — LLM Layer (Layer 5)

### 3.1 Anthropic SDK Setup
- `anthropic>=0.39.0`. `server/app/llm.py` with prompt-cache-enabled client. `.env` + `python-dotenv`.

### 3.2 Real-Time Translation (Sonnet)
- `translate(phoneme_sequence, profile, context) → TranslationResult`.
- System prompt (cached): patient profile + situational context with `cache_control: ephemeral`.
- User prompt: phoneme distributions + last 20 utterances + topics + pending questions.
- `translations` table.

### 3.3 Calibration Analysis (Opus)
- `analyze_calibration(calibration_data) → DistortionInsights`: deeper offline pass identifying systematic patterns; recommends confusion-matrix priors. Feeds Step 2.3.

### 3.4 Ranked Candidates via Prompt (NOT a beam-search module)
- The Claude API IS the beam search. No standalone `beam_search.py`.
- Tool-use schema `submit_translation(primary, alternatives[3], per_word_confidence, emergency_flag, context_hints)` — Sonnet returns ranked candidates in one call.
- Prompt enforces: ≥ 2 alternatives must be context-free (context is a prior, not a constraint).

---

## Phase 4 — Context Window (6 Layers)

### 4.1 Context Manager
- `server/app/context.py` with six classes: `UtteranceLog`, `TopicTracker`, `PendingQuestionQueue`, `SpeakerRegistry`, `EmotionalStateTracker`, `SessionMetadata`.
- `assemble_context()` applies decay weights → LLM-ready dict.

### 4.2 Context Persistence
- Tables: `topics`, `pending_questions`, `emotional_states`, `session_metadata`.

### 4.3 Emergency Detection
- `server/app/emergency.py`: priority vocabulary (pain, help, can't breathe, choking); lower confidence threshold; prosodic urgency cues.
- WS broadcast `{"type":"emergency", phrase, confidence}`.

---

## Phase 5 — Polished Capture Layer

Productionize the capture stack from Phase 1.6 + UX niceties.

### 5.1 Production Capture Component
- `client/components/mic-button.tsx`: permission state + `AnalyserNode` levels meter + error states.

### 5.2 Connection Manager
- Reconnect strategy, queueing during disconnects, surfacing connection health to the UI.

---

## Phase 6 — Calibration UI (Production)

### 6.1 Welcome Screen
- 3-phase explainer; CTA → calibration.

### 6.2 Calibration Wizard
- `/calibrate`: 4 sentence sets, one prompt at a time; record/replay/redo; progress bar.
- Completion triggers `POST /calibration/{id}/build`; "analyzing your speech patterns" runs Opus.

### 6.3 Profile Confirmation
- Plain-language summary of learned distortions; confirm → save profile.

---

## Phase 7 — Live Translation UI (Production)

### 7.1 Translation Display
- `/translate`: full-screen listener-facing view; per-word confidence (green > 0.8, yellow 0.5–0.8, red < 0.5); top-3 alternatives.
- Emergency flag → red flash + chime + persistent banner until dismissed.

### 7.2 Tap-to-Confirm Feedback
- ✓ / ✗ per utterance; ✗ opens inline edit; `POST /feedback/{utterance_id}` feeds Step 8.1.

### 7.3 Accessibility Controls
- Floating panel: font size (16–72 px), high-contrast, dark/light, dyslexia-friendly font; persisted to `localStorage`.

### 7.4 Conversational Context Panel
- Side panel with recent listener transcripts + active topics for misrouting correction.

---

## Phase 8 — Continuous Learning & Adaptation

### 8.1 Online Profile Updates
- EMA (α = 0.05) updates to confusion matrix on confirmed translations.
- Detect > 2σ shift → flag possible neurological event.

### 8.2 Therapy Gain Tracking
- Weekly aggregate of confusion matrix; `/progress` page charts accuracy over time.

### 8.3 Personal Vocabulary
- Caregiver-managed list (family / pet / medication names) with prompt boost.
- `personal_vocabulary` table.

---

## Phase 9 — TORGO Demo UI

### 9.1 Demo Mode WebSocket
- `/ws/demo/{torgo_speaker_id}`: streams TORGO audio in real time as if it were live mic input.

### 9.2 Demo Page
- `client/app/demo/page.tsx`: speaker selector, playback controls, side-by-side intended vs. translated text.

---

## Phase 10 — Productionization

### 10.1 Tests
- `server/tests/`: pytest for `UtteranceSegmenter`, `PhonemeExtractor`, distortion model, context decay, emergency detection, **vocabulary alignment** gate.
- `client/tests/`: Vitest + Testing Library for calibration wizard and translation display.
- E2E: TORGO clip → full pipeline → non-empty translation.

### 10.2 Logging & Observability
- `structlog`; per-session log files in `data/logs/`. Health endpoint reports model load, device, DB row counts.

### 10.3 Performance
- MPS device on Apple Silicon (`PHONEMER_DEVICE=mps`).
- Quantize wav2vec2 to int8 if CPU-only.
- `py-spy` profile; target < 200 ms per utterance.

### 10.4 Packaging
- `Makefile`: `setup / server / client / demo`. `start.sh` boots both. Optional PyInstaller bundle.

### 10.5 Privacy
- All data local. `data/` git-ignored. Optional `sqlcipher` for clinical settings. Settings panel "clear data" cascades by patient ID.

---

## Build Order

1. **Phase 1 (1.1–1.4)** — utterance segmentation + speaker routing + faster-whisper STT ✓
2. **Phase 1.5** — Load TORGO dataset
3. **Phase 1.6** — Frontend debug surface (mic + WS + `/debug` + TORGO picker + layer-status strip + record/replay)
4. **Phase 2** — calibration + distortion model (vocab-alignment gate; validated against TORGO)
5. **Phase 1.7** — Silero VAD swap + ECAPA non-speech rejection (lands before LLM is online)
6. **Phase 3 + Phase 4** — LLM translation with full context (LLM is the beam search)
6. **Phase 6** — Calibration UI (production)
7. **Phase 7** — Live translation UI (production)
8. **Phase 9** — TORGO demo UI
9. **Phase 8** — Adaptation loops
10. **Phase 10** — Hardening
