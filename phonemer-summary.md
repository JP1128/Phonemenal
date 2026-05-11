# Phonemer — Project Summary

## A live speech translator for brain injury patients with speech impediments

---

## The Problem

Brain injury patients with speech impediments can often understand everything said to them and know exactly what they want to say — but their motor systems can't execute the speech correctly. This creates a devastating communication gap between patients and their doctors, nurses, family members, and friends. Current solutions are either too limited (pre-programmed phrase boards) or too generic (standard speech recognition that fails on non-standard speech).

---

## The Idea

Build a system that:

1. **Calibrates** to a specific patient by having them read known sentences while the system captures raw audio
2. **Learns** the systematic discrepancies between how the patient produces speech and how it should sound — at the phoneme level, not the word level
3. **Translates** the patient's live speech in real time, displaying the intended sentence on a screen for the listener (doctor, nurse, family member) to read

The core innovation is using an LLM layer that receives raw audio analysis (not pre-processed speech-to-text) along with the patient's learned distortion profile and conversational context to determine what the patient is trying to say.

---

## Existing Landscape

Two major efforts exist in this space:

**Voiceitt** is the closest commercial product. It uses AI-powered speech recognition that adapts to individual users' non-standard speech patterns. Users train the app by recording phrases, and the system learns to recognize their specific pronunciations. It claims up to 3x better recognition rates than standard speech recognition for mild-to-moderate impairments. Integrations exist for Zoom, WebEx, and Chrome.

**Google Project Euphonia** is a research initiative building the world's largest dataset of disordered speech (1.5M+ utterances from ~3,000 speakers as of early 2025). Rather than a consumer product, it aims to make speech recognition models themselves more inclusive, with open-source tools available on GitHub. It is expanding beyond English into Spanish, Japanese, French, and Hindi.

**Key Differentiators for Phonemer:**

- Voiceitt trains individual phrases — Phonemer learns generalizable phoneme-level motor patterns that apply to any word, including words never seen in calibration
- Voiceitt doesn't model fatigue — Phonemer tracks how the patient's distortion patterns degrade over a conversation and adjusts dynamically
- Neither existing solution uses an LLM layer for contextual disambiguation — Phonemer uses conversational context to resolve ambiguous phoneme sequences

---

## System Architecture — 6 Processing Layers

### Layer 1: Raw Audio Capture
Continuous microphone stream with NO preprocessing by browser speech APIs. The Web Speech API is explicitly bypassed because it applies noise cancellation and speech normalization that destroys the distortion signals the system needs to learn. Raw PCM audio at 16kHz+ with Voice Activity Detection to segment speech from silence.

### Layer 2: Acoustic Feature Extraction
Extract low-level speech features that capture motor patterns: Mel-Frequency Cepstral Coefficients (vocal tract shape), formant frequencies (vowel positioning), Voice Onset Time (consonant timing precision), pitch contour and jitter/shimmer (prosody breakdown), spectral flux (consonant articulation), and silence/pause duration mapping (motor planning delays).

### Layer 3: Distorted Phoneme Recognition
Map audio features to what the patient actually produced — not what they meant. A custom phoneme recognizer outputs probability distributions over ~44 English phonemes rather than hard labels. The distribution itself is the signal: a sound that's 40% /θ/, 30% /d/, 20% /t/ is that patient's signature for attempted /θ/.

### Layer 4: Patient Distortion Model
The learned map of this specific patient's speech impediment patterns. Includes a phoneme confusion matrix (P(intended | produced)), positional rules (distortions differ word-initial vs. word-final), consonant cluster simplification patterns, syllable deletion tendencies, fatigue curves, and compensatory strategies the patient develops over time.

### Layer 5: LLM Contextual Correction
The LLM receives phoneme probability sequences, the patient's distortion profile, conversational context, and situational context. It applies constrained beam search to generate multiple candidate sentences, scoring each against all available signals, and outputs the top interpretations with per-word confidence scores.

### Layer 6: Translation Display
Real-time display of translated speech on a screen facing the listener. Includes confidence indicators per phrase, alternative interpretations when confidence is low, tap-to-confirm feedback for continuous learning, and emergency phrase detection for pain/distress signals.

---

## LLM Layer — Feature Specification

### Inputs (What the LLM Receives)

**Critical Priority:**
- **Phoneme Probability Sequences** — The full probability distribution over phonemes for each time-slice of audio, including timing information, partial phoneme attempts, and co-articulation context. Hard labels destroy the ambiguity signal the LLM needs for soft decoding.
- **Patient Distortion Profile** — The calibration-derived confusion matrix and rules. Without this, the LLM is a generic speech corrector. With it, the LLM knows that when THIS patient produces /d/, there's a 70% chance they meant /θ/.

**High Priority:**
- **Fatigue & Session State** — Minutes elapsed, speech rate trends, error density trajectory, and current fatigue phase (fresh/mild/moderate/severe). The confusion matrix at minute 2 differs from minute 20.
- **Conversational Context Window** — Last N translated utterances from the patient, responses from listeners, topic tracking, unresolved questions, emotional trajectory, and repeated attempt detection.

**Medium Priority:**
- **Situational Context** — Setting (hospital, home, therapy), listener identity (doctor, family), time of day, activity context (mealtime, medication time), communication urgency level.
- **Patient Personal Vocabulary** — Family member names, pet names, medication names, preferred terms, cultural/linguistic background. These words get elevated probability during beam search.

### Processing (What the LLM Does)

1. **Constrained Beam Search Decoding** — Generate top-K candidate sentences scored against distortion model, language model probability, conversational context, and situational context. Surface top-3 with confidence.

2. **Missing Word & Syllable Recovery** — Detect gaps suggesting dropped words using timing/prosody, apply grammatical expectations and the patient's syllable deletion profile, distinguish intentional pauses from motor-failure gaps.

3. **Repetition & Self-Correction Detection** — Recognize retry sequences and "groping" patterns, distinguish self-correction from perseveration (involuntary repetition common in frontal lobe injuries), collapse retry sequences into single output words.

4. **Confidence Calibration** — Compute per-word and per-sentence confidence, calibrate against historical accuracy, flag specific uncertain words visually, detect when confidence is too low to display any translation.

5. **Emergency Detection** — Maintain priority vocabulary (pain, help, can't breathe, choking), apply lower confidence thresholds for emergency words, detect prosodic urgency cues, trigger visual/audio alerts. False positives are acceptable; missed emergencies are not.

6. **Temporal Model Adaptation** — Continuously update distortion probabilities based on confirmed translations, track fatigue curves, detect sudden changes that might indicate a new neurological event, adapt to therapy gains over weeks.

### Outputs (What the LLM Returns)

Each utterance produces a structured response containing: primary translation with per-word confidence, top-3 alternative interpretations, uncertain word highlighting, emergency content flag, context hints when context heavily influenced the translation, intelligent retry suggestions when confidence is too low, and a feedback capture mechanism for the listener to confirm or correct.

**Core design principle: Transparency over confidence.** In medical settings, a confident wrong translation is far more dangerous than an honest "I'm not sure." The system never hides uncertainty to appear smoother.

---

## Context Window Architecture

The context window is assembled from six distinct layers, each with its own data format, update rules, and temporal decay behavior:

### 1. Utterance Log (Core)
Chronological record of everything said in the room by anyone. Patient utterances include raw phonemes and translated text; non-patient utterances include standard STT transcription. Last 20 utterances sent to LLM verbatim; older ones compressed into the Topic Tracker.

### 2. Active Topic Tracker (Core)
Running summary of active discussion subjects. Keywords extracted from non-patient speech, with relevance weights that decay over time (half-life ~4 minutes). Triggers a vocabulary_boost array — when "arm condition" is the active topic, arm-related medical vocabulary gets elevated probability in beam search.

### 3. Pending Question Queue (High)
Questions asked by non-patients that the patient hasn't answered. Each question is typed (yes/no, open-ended, numeric scale, etc.) and given expected response vocabulary. A yes/no question about arm pain means the patient's next utterance is very likely "yes," "no," or a qualified response. Questions follow a bell curve of urgency — low immediately (patient hasn't had time to respond), peak at ~5 seconds, expiring after ~30 seconds.

### 4. Speaker Registry (High)
Tracks who is in the room using voice embeddings. The patient is pre-enrolled from calibration. Non-patient speakers get automatic within-session labels (Speaker A, Speaker B) with inferred roles (medical_professional, family_member) based on vocabulary patterns and conversational style. Role inference influences vocabulary expectations.

### 5. Patient Emotional State Tracker (Medium)
Inferred from prosodic features (pitch, volume, rate, pauses) — not from word content, to avoid circular reasoning. Tracks frustration indicators (repeated failed attempts, abandoned utterances, sighs). When frustration is detected, the system suggests the listener slow down. Distress flags trigger the emergency pathway.

### 6. Session Metadata (Medium)
Static and slowly-changing information: setting, time of day, scheduled context, patient profile (injury type, severity, personal vocabulary, communication strengths/challenges), and recent medical context. Acts as the stable backdrop — the "system prompt equivalent" — against which all dynamic context is interpreted.

### Decay Rules
- **Utterance relevance:** Exponential decay, half-life ~14 seconds. The question asked 5 seconds ago matters enormously; the statement from 3 minutes ago is background.
- **Topic relevance:** Slow decay, half-life ~4 minutes, with a floor of 0.1 so topics never fully disappear within a session.
- **Pending question urgency:** Bell curve peaking at ~5 seconds, expiring at ~30 seconds.
- **Vocabulary boost:** Decays with parent topic but lingers slightly longer (1.3x multiplier).

### Edge Case Design Principles
1. **Context is a prior, not a constraint** — context influences probability weights but never overrides strong phoneme evidence
2. **All speech is context, but only direct address creates response expectation** — doctor-to-nurse conversation still feeds the topic tracker
3. **Always maintain context-free candidates as a safety net** — beam search must keep 2-3 candidates that ignore topic context entirely
4. **The system must work without context; context makes it better, not functional** — stale context triggers fallback to basic needs vocabulary
5. **Only human-present-in-room speech feeds context** — TV/radio audio is detected and excluded
6. **Solo speech is still monitored but interpreted more cautiously** — higher confidence thresholds, more sensitive emergency detection

---

## Speaker Identification

The system uses a single microphone but separates patient speech from all other audio. During calibration, the system captures the patient's vocal identity (fundamental frequency, formant structure, vocal tract resonance) alongside their distortion patterns. Every detected speech segment is classified: "enrolled patient" or "someone else."

Patient speech routes through the full distortion-correction pipeline. Non-patient speech routes through standard speech-to-text and feeds the conversational context window. This enables the doctor scenario: a doctor asks "How are your arms feeling?", the system transcribes it as context, and when the patient responds, the LLM knows to weight arm-related medical vocabulary higher.

---

## Hardware

A USB condenser microphone (e.g., Blue Yeti Nano or Audio-Technica ATR2100x) connected to a laptop running the web application. The computer screen serves as the listener-facing display. OS-level noise suppression disabled in the browser's getUserMedia constraints to preserve raw distortion signal. Everything runs locally — the browser on localhost:3000 captures audio, FastAPI on localhost:8000 processes it, and the only external call is to the Claude API.

---

## Brain Injury Speech Pattern Types

The system must handle different distortion signatures based on injury type:

**Dysarthria** (muscle weakness/paralysis): Consistent consonant imprecision, hypernasality, monotone pitch, slow rate. Modeled with systematic consonant weakening rules and a voicing confusion matrix.

**Apraxia of Speech** (motor planning disruption): Inconsistent errors, sound groping, transpositions, islands of clarity. The same word may be produced differently each time. Modeled with probabilistic phoneme sequencing and variability ranges.

**Mixed Presentation** (multiple injury sites): Combination of consistent and variable errors with fatigue-dependent shifts. Modeled with multiple distortion profiles that activate based on detected fatigue phase.

---

## Why Existing STT Fails for This Use Case

The Web Speech API and similar speech-to-text services are fundamentally unsuited because:

1. They are trained on typical speech and actively normalize input toward standard pronunciation — destroying the distortion signals needed for pattern learning
2. They apply noise cancellation that erases subtle phoneme information
3. They output hard text labels, discarding the probability distributions that carry critical disambiguation information
4. They cannot be fine-tuned, retrained, or adapted to individual speakers
5. They treat the speech recognition model as a black box with no access to intermediate representations

The Phonemer architecture bypasses all standard STT and works directly with raw audio → acoustic features → phoneme probability distributions → LLM correction.

---

## Datasets

These datasets are essential — not optional. None of the developers on the team have speech impediments, so there is no way to develop, test, or demo the pipeline without real impaired speech audio. All three datasets below are **prompted speech**: the patient was given a known sentence to read, and their impaired attempt was recorded. This gives us the exact pairing Phonemer's calibration system produces (intended text + impaired audio), enabling full pipeline development and live demos without access to actual patients.

### Pre-trained Models (No Dataset Needed)

The wav2vec2 phoneme model (facebook/wav2vec2-lv-60-espeak-cv-ft) was trained on 60,000 hours of speech and fine-tuned on Common Voice for phoneme recognition. The SpeechBrain ECAPA-TDNN was trained on VoxCeleb (1M+ utterances, 7,000+ speakers) for speaker embeddings. Both work out of the box.

### Development & Demo Datasets

**TORGO** — University of Toronto. ~23 hours of English speech from 8 speakers with cerebral palsy or ALS + 7 control speakers. Includes paired audio + intended text across 162 intelligibility assessment sentences, 460 TIMIT-derived sentences, and 30 spontaneous image descriptions. Available on Kaggle. **This is the primary development dataset** — use TORGO audio to simulate a patient during development, feed it through the pipeline, and demo the system translating impaired speech into intended sentences in real time.

**UASpeech** — University of Illinois. Recordings from 15 patients with cerebral palsy + 13 control speakers. Prompted words at three intelligibility levels (high, mid, low), each labeled with the intended word. Access by emailing uaspeech-requests@lists.illinois.edu.

**SAP-Hypo5** — Built on the Speech Accessibility Project data. 35,000 utterances where each is paired with the reference transcript AND the top-5 ASR hypotheses from existing speech recognition. This lets you directly measure how much improvement Phonemer adds over baseline ASR. The full SAP dataset (400+ hours, 500+ speakers) requires a signed data use agreement and one-page project plan submitted to speechaccessibility@beckman.illinois.edu.

### Demo Strategy

During a live demo, instead of a patient speaking into the microphone, play TORGO audio files through the pipeline. The system processes the impaired speech, applies the distortion model (pre-built from TORGO calibration data), sends it through the LLM, and displays the translated sentence — demonstrating the full pipeline end-to-end. The audience sees impaired audio go in and clear intended text come out.

### Self-Generated Data

The calibration pipeline itself is a dataset creation tool. Every patient who eventually calibrates generates paired training data (raw audio + intended text + phoneme-level alignment + fatigue context) that no existing dataset provides at this granularity. This data feeds back into improving the system for future patients.

---

## Prototype Built

An initial browser-based prototype ("Phonemer") was built as a React application demonstrating the user experience flow:

- Welcome screen explaining the 3-phase process
- Calibration interface with 4 sentence sets (Basic Needs, Medical Communication, Emotional Expression, Common Sounds)
- Pattern engine using edit-distance alignment to learn word-level substitutions, prefix/suffix distortions, and phrase-level corrections
- Live translation mode with real-time speech capture, pattern application, and translated output display
- Adjustable font size and high-contrast mode for accessibility
- Pattern visualization showing learned word corrections and calibration history

The prototype uses the Web Speech API as a placeholder (acknowledging its limitations) to demonstrate the UX flow. The production system replaces this with the raw audio pipeline described above.
