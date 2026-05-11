"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AccessibilityPanel } from "../../components/accessibility-panel";
import { useAccessibility } from "../../lib/accessibility";
import { createAudioCapture } from "../../lib/audio-capture";
import { createCalibrationClient } from "../../lib/calibration-client";

const SERVER_HTTP = "http://localhost:8000";
const wsBase = SERVER_HTTP.replace(/^http/, "ws");

type SentenceSet = {
  id: string;
  label: string;
  description: string;
  sentences: string[];
};

type CommittedUtterance = {
  set_id: string;
  sentence_index: number;
  utterance_id: number;
  duration_ms: number;
  intended_text: string;
};

type Stage = "welcome" | "wizard" | "building" | "confirmation" | "error";

type AutoState =
  | "idle"            // mic off / between recordings, waiting to arm
  | "warming"         // about to arm — short pause so the user can read the sentence
  | "listening"       // mic on, begin sent, waiting for speech onset
  | "speaking"        // speech detected, recording in progress
  | "committing"      // silence detected, end sent, waiting for commit ack
  | "all_done";

type ProfileSummary = {
  calibration_id: number;
  profile_id: number;
  n_utterances: number;
  syllable_deletion_rate: number;
  opus_summary?: string | null;
  detail?: {
    profile?: { profile_json: string };
  };
};

// VAD knobs (client-side, energy-based on the 100 ms RMS callback).
// Voice/silence thresholds are RECOMPUTED per-sentence relative to the
// noise floor measured during the warm-up window — see armCurrent().
const VOICE_FLOOR_MULT = 3.0;        // voice = level > floor * this OR > VOICE_FLOOR_MIN
const SILENCE_FLOOR_MULT = 1.5;      // silence = level < floor * this AND < SILENCE_FLOOR_CAP
const VOICE_FLOOR_MIN = 0.022;       // absolute floor for "voice" in a quiet room
const SILENCE_FLOOR_CAP = 0.030;     // absolute ceiling for "silence" in a noisy room
const ONSET_FRAMES = 3;              // ~300 ms continuous voice → "speaking"
const OFFSET_FRAMES = 9;             // ~900 ms continuous silence after voice → commit
const WARMUP_MS = 900;               // pause between landing on a sentence and arming
const MAX_RECORDING_MS = 15000;      // safety cap on a single utterance
const NO_SPEECH_TIMEOUT_MS = 12000;  // listening with no speech detected → hint to user

export default function CalibratePage() {
  const a11y = useAccessibility();

  const [stage, setStage] = useState<Stage>("welcome");
  const [error, setError] = useState<string | null>(null);

  const [sets, setSets] = useState<SentenceSet[] | null>(null);
  const [calibrationId, setCalibrationId] = useState<number | null>(null);
  const [setIndex, setSetIndex] = useState(0);
  const [sentenceIndex, setSentenceIndex] = useState(0);
  const [committed, setCommitted] = useState<CommittedUtterance[]>([]);
  const [autoState, setAutoState] = useState<AutoState>("idle");
  const [level, setLevel] = useState(0);
  const [profile, setProfile] = useState<ProfileSummary | null>(null);

  // ---- refs that mirror state for use inside long-lived callbacks ----
  const setsRef = useRef<SentenceSet[] | null>(null);
  const setIndexRef = useRef(0);
  const sentenceIndexRef = useRef(0);
  const stageRef = useRef<Stage>("welcome");
  const autoStateRef = useRef<AutoState>("idle");
  const speechFramesRef = useRef(0);
  const silenceFramesRef = useRef(0);
  const peakLevelRef = useRef(0);
  // Rolling history of recent RMS values, used to estimate the noise floor.
  const ambientHistoryRef = useRef<number[]>([]);
  const voiceThresholdRef = useRef(VOICE_FLOOR_MIN);
  const silenceThresholdRef = useRef(VOICE_FLOOR_MIN * 0.7);
  const recordingStartedAtRef = useRef<number | null>(null);
  const safetyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [noiseFloor, setNoiseFloor] = useState(0);

  useEffect(() => { setsRef.current = sets; }, [sets]);
  useEffect(() => { setIndexRef.current = setIndex; }, [setIndex]);
  useEffect(() => { sentenceIndexRef.current = sentenceIndex; }, [sentenceIndex]);
  useEffect(() => { stageRef.current = stage; }, [stage]);
  useEffect(() => { autoStateRef.current = autoState; }, [autoState]);

  const captureRef = useRef<ReturnType<typeof createAudioCapture> | null>(null);
  const calRef = useRef<ReturnType<typeof createCalibrationClient> | null>(null);
  const lastCommitRef = useRef<CommittedUtterance | null>(null);

  // ============================================================ fetch sets

  useEffect(() => {
    fetch(`${SERVER_HTTP}/calibration/sentences`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => setSets(data.sets))
      .catch((err) => setError(`Could not load sentence sets: ${err}`));
  }, []);

  // ============================================================ derived

  const totalSentences = useMemo(
    () => (sets ?? []).reduce((n, s) => n + s.sentences.length, 0),
    [sets],
  );
  const completedCount = committed.length;
  const set = sets?.[setIndex];
  const sentence = set?.sentences[sentenceIndex];
  const allDone =
    sets !== null &&
    totalSentences > 0 &&
    completedCount >= totalSentences;

  // ============================================================ teardown

  useEffect(() => {
    return () => {
      captureRef.current?.stop();
      captureRef.current = null;
      calRef.current?.disconnect();
      calRef.current = null;
    };
  }, []);

  // ============================================================ VAD

  const setAuto = useCallback((next: AutoState) => {
    autoStateRef.current = next;
    setAutoState(next);
  }, []);

  const onLevel = useCallback(
    (lv: number) => {
      setLevel(lv);
      if (lv > peakLevelRef.current) peakLevelRef.current = lv;

      const state = autoStateRef.current;

      // During warm-up we sample ambient levels to compute the noise floor.
      if (state === "warming") {
        const hist = ambientHistoryRef.current;
        hist.push(lv);
        if (hist.length > 20) hist.shift();
        return;
      }

      if (state === "listening") {
        if (lv > voiceThresholdRef.current) {
          speechFramesRef.current += 1;
          silenceFramesRef.current = 0;
        } else {
          speechFramesRef.current = 0;
        }
        if (speechFramesRef.current >= ONSET_FRAMES) {
          setAuto("speaking");
          recordingStartedAtRef.current = performance.now();
        }
      } else if (state === "speaking") {
        if (lv < silenceThresholdRef.current) {
          silenceFramesRef.current += 1;
        } else {
          silenceFramesRef.current = 0;
        }
        // Either silence-driven OR safety-cap commit.
        const tooLong =
          recordingStartedAtRef.current !== null &&
          performance.now() - recordingStartedAtRef.current >
            MAX_RECORDING_MS;
        if (silenceFramesRef.current >= OFFSET_FRAMES || tooLong) {
          setAuto("committing");
          calRef.current?.end();
        }
      }
    },
    [setAuto],
  );

  const finalizeRecording = useCallback(() => {
    // Manual "Done speaking" — works in any active state.
    const state = autoStateRef.current;
    if (state !== "listening" && state !== "speaking") return;
    setAuto("committing");
    calRef.current?.end();
  }, [setAuto]);

  // ============================================================ arm/end

  const ensureCapture = useCallback(async () => {
    if (captureRef.current?.isRunning()) return;
    const cap = createAudioCapture({
      onPcm: (chunk) => calRef.current?.sendPcm(chunk),
      onLevel,
      onError: (err) => {
        setError(err.message || String(err));
        setAuto("idle");
      },
    });
    captureRef.current = cap;
    try {
      await cap.start();
    } catch {
      // onError already fired.
    }
  }, [onLevel, setAuto]);

  const armCurrent = useCallback(async () => {
    const sets_ = setsRef.current;
    if (!sets_) return;
    if (stageRef.current !== "wizard") return;
    const si = setIndexRef.current;
    const senI = sentenceIndexRef.current;
    const cur = sets_[si];
    if (!cur) return;
    const sent = cur.sentences[senI];
    if (!sent) return;
    if (!calRef.current) return;

    await ensureCapture();

    // Compute adaptive thresholds from the warm-up ambient samples.
    const hist = ambientHistoryRef.current.slice();
    let floor = VOICE_FLOOR_MIN * 0.5;
    if (hist.length >= 4) {
      hist.sort((a, b) => a - b);
      // 90th percentile of the warm-up window — robust to brief blips.
      const idx = Math.min(hist.length - 1, Math.floor(hist.length * 0.9));
      floor = hist[idx];
    }
    setNoiseFloor(floor);
    voiceThresholdRef.current = Math.max(
      VOICE_FLOOR_MIN,
      floor * VOICE_FLOOR_MULT,
    );
    silenceThresholdRef.current = Math.min(
      SILENCE_FLOOR_CAP,
      Math.max(floor * SILENCE_FLOOR_MULT, VOICE_FLOOR_MIN * 0.7),
    );
    // Silence threshold must always be < voice threshold.
    if (silenceThresholdRef.current >= voiceThresholdRef.current) {
      silenceThresholdRef.current = voiceThresholdRef.current * 0.7;
    }

    speechFramesRef.current = 0;
    silenceFramesRef.current = 0;
    peakLevelRef.current = 0;
    recordingStartedAtRef.current = null;

    lastCommitRef.current = {
      set_id: cur.id,
      sentence_index: senI,
      utterance_id: -1,
      duration_ms: 0,
      intended_text: sent,
    };
    calRef.current.begin({
      set_id: cur.id,
      sentence_index: senI,
      intended_text: sent,
    });
    setAuto("listening");

    // Safety: if no speech is detected after a long delay, hint the user.
    if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
    safetyTimerRef.current = setTimeout(() => {
      if (autoStateRef.current === "listening") {
        setError(
          "Didn't hear anything yet — try moving closer to the mic, or use 'Done speaking' once you've finished.",
        );
      }
    }, NO_SPEECH_TIMEOUT_MS);
  }, [ensureCapture, setAuto]);

  // Auto-arm whenever we land on a fresh sentence, with a short warm-up.
  useEffect(() => {
    if (stage !== "wizard") return;
    if (allDone) return;
    if (!sets) return;
    if (
      autoStateRef.current === "speaking" ||
      autoStateRef.current === "committing"
    )
      return;
    // Reset ambient sampling for the new sentence.
    ambientHistoryRef.current = [];
    setAuto("warming");
    const timer = setTimeout(() => {
      armCurrent();
    }, WARMUP_MS);
    return () => clearTimeout(timer);
  }, [stage, sets, setIndex, sentenceIndex, allDone, armCurrent, setAuto]);

  // ============================================================ controls

  const skipCurrent = useCallback(async () => {
    // Discard server-side buffered audio for the current sentence and advance
    // without a commit. Good escape hatch when the patient can't produce one.
    calRef.current?.discard();
    setAuto("idle");
    advance();
  }, []);

  const redoLast = useCallback(() => {
    setError(null);
    if (committed.length === 0) return;
    setCommitted((prev) => prev.slice(0, -1));
    // Step backwards. Easier to recompute from the popped row.
    const last = committed[committed.length - 1];
    const lastSetIndex =
      sets?.findIndex((s) => s.id === last.set_id) ?? setIndex;
    setSetIndex(lastSetIndex);
    setSentenceIndex(last.sentence_index);
    setAuto("idle");
  }, [committed, setIndex, sets, setAuto]);

  const replayLast = useCallback(() => {
    const last = committed[committed.length - 1];
    if (!last) return;
    new Audio(
      `${SERVER_HTTP}/calibration/utterance/${last.utterance_id}/audio`,
    )
      .play()
      .catch(() => {
        /* requires a user gesture in some browsers */
      });
  }, [committed]);

  // ============================================================ start session

  const advance = useCallback(() => {
    const sets_ = setsRef.current;
    if (!sets_) return;
    const si = setIndexRef.current;
    const senI = sentenceIndexRef.current;
    const cur = sets_[si];
    if (!cur) return;

    if (senI + 1 < cur.sentences.length) {
      setSentenceIndex(senI + 1);
      sentenceIndexRef.current = senI + 1;
      return;
    }
    if (si + 1 < sets_.length) {
      const ni = si + 1;
      setSetIndex(ni);
      setSentenceIndex(0);
      setIndexRef.current = ni;
      sentenceIndexRef.current = 0;
      return;
    }
    setAuto("all_done");
  }, [setAuto]);

  const beginCalibration = useCallback(async () => {
    setError(null);
    setProfile(null);
    setCommitted([]);
    setSetIndex(0);
    setSentenceIndex(0);
    setIndexRef.current = 0;
    sentenceIndexRef.current = 0;
    setAuto("idle");
    try {
      const res = await fetch(`${SERVER_HTTP}/calibration/start`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setCalibrationId(data.calibration_id);
      setSets(data.sentence_sets);
      setsRef.current = data.sentence_sets;

      const client = createCalibrationClient(
        `${wsBase}/ws/calibration/${data.calibration_id}`,
        {
          onMessage: (msg) => {
            if (msg.type === "ready") {
              // Just an ack; nothing to do — VAD drives the UI.
              return;
            }
            if (msg.type === "calibration_utterance") {
              const last = lastCommitRef.current;
              if (!last) return;
              const committedRow: CommittedUtterance = {
                set_id: last.set_id,
                sentence_index: last.sentence_index,
                utterance_id: msg.utterance_id,
                duration_ms: msg.duration_ms,
                intended_text: last.intended_text,
              };
              setCommitted((prev) => [...prev, committedRow]);
              setError(null);
              setAuto("idle");
              advance();
            }
            if (msg.type === "calibration_error") {
              const friendly =
                msg.error === "too_short"
                  ? "That recording was too quiet — try reading the sentence aloud at a normal volume."
                  : msg.error === "no_active_recording"
                    ? "Lost track of the current sentence — re-arming."
                    : msg.message;
              setError(friendly);
              setAuto("idle");
            }
          },
          onState: (s) => {
            if (s === "closed") {
              setAuto("idle");
            }
          },
        },
      );
      calRef.current = client;
      client.connect();
      setStage("wizard");
    } catch (err) {
      setError(`Could not start calibration: ${err}`);
      setStage("error");
    }
  }, [advance, setAuto]);

  // Auto-finish once everything is recorded.
  useEffect(() => {
    if (allDone && stage === "wizard") {
      // Stop the mic; there's nothing left to record.
      captureRef.current?.stop();
      captureRef.current = null;
      setAuto("all_done");
    }
  }, [allDone, stage, setAuto]);

  // ============================================================ build profile

  const buildProfile = useCallback(async () => {
    if (!calibrationId) return;
    setStage("building");
    setError(null);
    try {
      const res = await fetch(
        `${SERVER_HTTP}/calibration/${calibrationId}/build`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(await res.text());
      const summary = await res.json();
      const detail = await fetch(
        `${SERVER_HTTP}/calibration/${calibrationId}`,
      ).then((r) => r.json());
      setProfile({ ...summary, detail });
      setStage("confirmation");
    } catch (err) {
      setError(`Could not build profile: ${err}`);
      setStage("error");
    }
  }, [calibrationId]);

  // ============================================================ render

  return (
    <main
      className="min-h-screen px-6 py-6"
      style={{
        ...a11y.styleVars,
        background: "var(--phonemer-bg-color)",
        color: "var(--phonemer-text-color)",
      }}
    >
      <header className="flex items-center gap-3 max-w-3xl mx-auto mb-6">
        <h1 className="text-lg font-semibold tracking-wide">
          Phonemer · Calibration
        </h1>
        {calibrationId && (
          <span className="text-xs opacity-50">cal #{calibrationId}</span>
        )}
        <div className="ml-auto flex items-center gap-2 text-xs opacity-70">
          <a href="/" className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10">
            ← Home
          </a>
        </div>
      </header>

      <div className="max-w-3xl mx-auto">
        {stage === "welcome" && (
          <WelcomeScreen
            sets={sets}
            error={error}
            onBegin={beginCalibration}
          />
        )}

        {stage === "wizard" && set && sentence && !allDone && (
          <Wizard
            set={set}
            setIndex={setIndex}
            setCount={sets?.length ?? 0}
            sentence={sentence}
            sentenceIndex={sentenceIndex}
            sentenceCountInSet={set.sentences.length}
            completedCount={completedCount}
            totalSentences={totalSentences}
            autoState={autoState}
            level={level}
            noiseFloor={noiseFloor}
            voiceThreshold={voiceThresholdRef.current}
            silenceThreshold={silenceThresholdRef.current}
            error={error}
            canRedo={committed.length > 0}
            onFinalize={finalizeRecording}
            onSkip={skipCurrent}
            onRedo={redoLast}
            onReplay={replayLast}
          />
        )}

        {stage === "wizard" && allDone && (
          <DoneStep
            committed={completedCount}
            onReplay={replayLast}
            onBuild={buildProfile}
            onRedoSession={beginCalibration}
          />
        )}

        {stage === "building" && <BuildingStep />}

        {stage === "confirmation" && profile && (
          <ConfirmationStep
            profile={profile}
            onRedoSession={beginCalibration}
          />
        )}

        {stage === "error" && (
          <ErrorStep error={error} onRetry={beginCalibration} />
        )}
      </div>

      <AccessibilityPanel
        settings={a11y.settings}
        onChange={a11y.setSettings}
      />
    </main>
  );
}

// ============================================================ welcome

function WelcomeScreen({
  sets,
  error,
  onBegin,
}: {
  sets: SentenceSet[] | null;
  error: string | null;
  onBegin: () => void;
}) {
  return (
    <div className="space-y-8">
      <div className="space-y-3 text-center">
        <h2 className="text-3xl font-semibold">Calibrate Phonemer</h2>
        <p className="opacity-80 max-w-xl mx-auto">
          Phonemer captures your speech as you read short sentences and learns
          how you produce phonemes so it can translate your live speech later.
          The calibrator listens automatically — just read each sentence aloud.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <PhaseCard
          step={1}
          label="Capture"
          body="The mic stays open while you read. Each sentence advances on its own once you finish saying it."
        />
        <PhaseCard
          step={2}
          label="Calibrate"
          body="Phonemer compares what you said to the intended sentence and builds your speech profile."
        />
        <PhaseCard
          step={3}
          label="Translate"
          body="In a session, your live speech is translated in real time using the profile we just built."
        />
      </div>

      {sets && (
        <div className="rounded-lg border border-white/10 p-4 space-y-2">
          <div className="text-sm opacity-70">You'll read:</div>
          <ul className="space-y-1">
            {sets.map((s) => (
              <li key={s.id} className="flex items-baseline gap-2">
                <span className="font-medium">{s.label}</span>
                <span className="text-xs opacity-60">
                  · {s.sentences.length} sentences · {s.description}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="rounded bg-rose-900/40 px-3 py-2 text-rose-200 text-sm">
          {error}
        </div>
      )}

      <div className="flex justify-center pt-2">
        <button
          onClick={onBegin}
          disabled={!sets}
          className="px-6 py-3 rounded-lg disabled:opacity-50 font-semibold text-base"
          style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
        >
          Begin calibration
        </button>
      </div>
    </div>
  );
}

function PhaseCard({
  step,
  label,
  body,
}: {
  step: number;
  label: string;
  body: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 p-4 space-y-2">
      <div className="text-xs opacity-50">Phase {step}</div>
      <div className="text-lg font-semibold">{label}</div>
      <div className="text-sm opacity-80 leading-snug">{body}</div>
    </div>
  );
}

// ============================================================ wizard

function Wizard(props: {
  set: SentenceSet;
  setIndex: number;
  setCount: number;
  sentence: string;
  sentenceIndex: number;
  sentenceCountInSet: number;
  completedCount: number;
  totalSentences: number;
  autoState: AutoState;
  level: number;
  noiseFloor: number;
  voiceThreshold: number;
  silenceThreshold: number;
  error: string | null;
  canRedo: boolean;
  onFinalize: () => void;
  onSkip: () => void;
  onRedo: () => void;
  onReplay: () => void;
}) {
  const overallPct = Math.round(
    (props.completedCount / Math.max(1, props.totalSentences)) * 100,
  );

  const statusText = ((): string => {
    switch (props.autoState) {
      case "warming":
        return "Get ready — read the sentence aloud";
      case "listening":
        return "Listening for your voice…";
      case "speaking":
        return "Recording — keep reading";
      case "committing":
        return "Saving…";
      case "all_done":
        return "All sentences captured";
      default:
        return "Preparing…";
    }
  })();

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <div className="flex items-baseline justify-between text-xs opacity-70">
          <span>
            {props.set.label} ·{" "}
            <span className="opacity-60">{props.set.description}</span>
          </span>
          <span>
            {props.completedCount} / {props.totalSentences}
          </span>
        </div>
        <ProgressBar pct={overallPct} />
        <div className="flex gap-1 pt-1">
          {Array.from({ length: props.setCount }).map((_, i) => {
            const cur = i === props.setIndex;
            const past = i < props.setIndex;
            return (
              <div
                key={i}
                className={`flex-1 h-0.5 rounded-sm transition-all ${
                  past
                    ? "bg-emerald-500"
                    : cur
                      ? "bg-emerald-400/50"
                      : "bg-white/10"
                }`}
              />
            );
          })}
        </div>
      </div>

      <div
        className="rounded-2xl border p-8 text-center min-h-[8rem] flex items-center justify-center transition-all"
        style={{
          background: "rgba(255,255,255,0.03)",
          borderColor:
            props.autoState === "speaking"
              ? "rgba(78, 205, 196, 0.6)"
              : "rgba(255,255,255,0.1)",
          boxShadow:
            props.autoState === "speaking"
              ? "0 0 20px rgba(78, 205, 196, 0.15)"
              : "none",
          fontSize: "calc(var(--phonemer-base-size) * 1.6)",
          lineHeight: 1.4,
        }}
      >
        {props.sentence}
      </div>

      <ListeningIndicator state={props.autoState} level={props.level} />

      <div className="text-center text-sm opacity-80">{statusText}</div>

      {props.error && (
        <div className="rounded bg-rose-900/40 px-3 py-2 text-rose-200 text-sm text-center">
          {props.error}
        </div>
      )}

      <div className="flex items-center justify-center gap-3 flex-wrap">
        {(props.autoState === "listening" || props.autoState === "speaking") && (
          <button
            onClick={props.onFinalize}
            className="px-4 py-2 rounded text-sm font-medium"
            style={{
              background:
                props.autoState === "speaking"
                  ? "var(--phonemer-accent)"
                  : "rgba(78, 205, 196, 0.2)",
              color:
                props.autoState === "speaking" ? "#0a0a0a" : "#e5e7eb",
              border:
                props.autoState === "speaking"
                  ? "none"
                  : "1px solid rgba(78, 205, 196, 0.4)",
            }}
            title="Click when you've finished saying the sentence"
          >
            ✓ Done speaking
          </button>
        )}
        <button
          onClick={props.onSkip}
          className="px-3 py-1.5 rounded text-xs border border-white/10 hover:bg-white/5"
        >
          ⏭ Skip this sentence
        </button>
        {props.canRedo && (
          <>
            <button
              onClick={props.onReplay}
              className="px-3 py-1.5 rounded text-xs border border-white/10 hover:bg-white/5"
            >
              ▶ Replay last
            </button>
            <button
              onClick={props.onRedo}
              className="px-3 py-1.5 rounded text-xs border border-white/10 hover:bg-white/5"
            >
              ↩ Redo previous
            </button>
          </>
        )}
      </div>

      <div className="text-center text-[10px] font-mono opacity-40 tabular-nums">
        sentence {props.sentenceIndex + 1}/{props.sentenceCountInSet} ·{" "}
        floor {props.noiseFloor.toFixed(3)} · voice ≥{" "}
        {props.voiceThreshold.toFixed(3)} · silence ≤{" "}
        {props.silenceThreshold.toFixed(3)} · level{" "}
        {props.level.toFixed(3)}
      </div>
    </div>
  );
}

function ListeningIndicator({
  state,
  level,
}: {
  state: AutoState;
  level: number;
}) {
  const active = state === "listening" || state === "speaking";
  const speaking = state === "speaking";
  // 18-segment level meter, mirrored about center for a balanced look.
  const segments = 18;
  const v = Math.min(1, Math.pow(level * 6, 0.7));
  const lit = Math.round(v * segments);
  return (
    <div className="flex items-center justify-center gap-[3px] h-7">
      {Array.from({ length: segments }).map((_, i) => {
        const distance = Math.abs(i - segments / 2);
        const isLit = active && i < lit;
        const baseColor = speaking ? "#4ecdc4" : "#10b981";
        const dim = "rgba(64,64,64,0.6)";
        return (
          <div
            key={i}
            className="rounded-sm"
            style={{
              width: 4,
              height: 12 + Math.max(0, segments / 2 - distance) * 1.6,
              background: isLit ? baseColor : dim,
              boxShadow: isLit && speaking ? `0 0 6px ${baseColor}77` : "none",
              transition: "background 80ms",
            }}
          />
        );
      })}
    </div>
  );
}

function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-2 bg-white/10 rounded">
      <div
        className="h-2 rounded transition-all duration-500"
        style={{
          width: `${pct}%`,
          background: "var(--phonemer-accent)",
        }}
      />
    </div>
  );
}

// ============================================================ done step

function DoneStep({
  committed,
  onReplay,
  onBuild,
  onRedoSession,
}: {
  committed: number;
  onReplay: () => void;
  onBuild: () => void;
  onRedoSession: () => void;
}) {
  return (
    <div className="text-center space-y-6 py-8">
      <div className="text-5xl">✓</div>
      <div>
        <div className="text-2xl font-semibold">All sentences recorded</div>
        <div className="opacity-70 text-sm mt-1">
          {committed} recordings ready. Building the profile takes ~30 seconds and
          requires the Anthropic API key on the server.
        </div>
      </div>
      <div className="flex justify-center gap-3 flex-wrap">
        <button
          onClick={onReplay}
          className="px-3 py-2 rounded text-sm border border-white/10 hover:bg-white/5"
        >
          ▶ Replay last
        </button>
        <button
          onClick={onRedoSession}
          className="px-3 py-2 rounded text-sm border border-white/10 hover:bg-white/5"
        >
          ↻ Start over
        </button>
        <button
          onClick={onBuild}
          className="px-5 py-2 rounded text-sm font-semibold"
          style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
        >
          Build distortion profile →
        </button>
      </div>
    </div>
  );
}

// ============================================================ building

function BuildingStep() {
  return (
    <div className="text-center space-y-4 py-16">
      <div
        className="mx-auto rounded-full w-12 h-12 border-2 border-white/10"
        style={{
          borderTopColor: "var(--phonemer-accent)",
          animation: "spin 1.2s linear infinite",
        }}
      />
      <div className="text-lg font-semibold">Analyzing your speech patterns</div>
      <div className="opacity-70 text-sm max-w-md mx-auto">
        Aligning each recording to its intended phoneme sequence, then asking
        Claude Opus to identify systematic patterns in your distortion profile.
      </div>
      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// ============================================================ confirmation

function ConfirmationStep({
  profile,
  onRedoSession,
}: {
  profile: ProfileSummary;
  onRedoSession: () => void;
}) {
  const summary = useMemo(() => buildPlainLanguageSummary(profile), [profile]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="text-5xl mb-2">🎯</div>
        <div className="text-2xl font-semibold">Your profile is ready</div>
      </div>

      <div className="rounded-lg border border-white/10 p-5 space-y-3">
        <div className="text-sm opacity-70">Summary</div>
        <p className="leading-relaxed text-base">{summary.intro}</p>
        {summary.bullets.length > 0 && (
          <ul className="space-y-2 mt-2">
            {summary.bullets.map((b, i) => (
              <li key={i} className="flex gap-3">
                <span className="opacity-50 mt-0.5">·</span>
                <span>{b}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="Recordings used" value={String(profile.n_utterances)} />
        <Stat
          label="Syllable deletion"
          value={`${(profile.syllable_deletion_rate * 100).toFixed(1)}%`}
        />
      </div>

      {profile.opus_summary && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
          <div className="text-xs font-mono opacity-60 mb-1">Claude Opus says</div>
          <div className="text-sm leading-relaxed">{profile.opus_summary}</div>
        </div>
      )}

      <div className="flex justify-center gap-3 flex-wrap pt-2">
        <a
          href="/"
          className="px-5 py-2 rounded text-sm font-semibold"
          style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
        >
          Use this profile to translate →
        </a>
        <button
          onClick={onRedoSession}
          className="px-3 py-2 rounded text-sm border border-white/10 hover:bg-white/5"
        >
          Recalibrate
        </button>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 p-4">
      <div className="text-xs opacity-60">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
    </div>
  );
}

// ============================================================ error

function ErrorStep({
  error,
  onRetry,
}: {
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <div className="space-y-4 text-center py-8">
      <div className="text-4xl">⚠</div>
      <div className="text-xl font-semibold">Something went wrong</div>
      <div className="text-sm text-rose-200 max-w-lg mx-auto">{error}</div>
      <button
        onClick={onRetry}
        className="px-4 py-2 rounded text-sm bg-emerald-600 hover:bg-emerald-500"
      >
        Try again
      </button>
    </div>
  );
}

// ============================================================ summary builder

type PlainSummary = {
  intro: string;
  bullets: string[];
};

function buildPlainLanguageSummary(profile: ProfileSummary): PlainSummary {
  let parsed: any = null;
  try {
    const raw = profile.detail?.profile?.profile_json;
    if (raw) parsed = JSON.parse(raw);
  } catch {
    /* ignore */
  }

  const bullets: string[] = [];

  if (profile.n_utterances < 20) {
    bullets.push(
      `Profile built from ${profile.n_utterances} recordings — accuracy will improve with more.`,
    );
  } else {
    bullets.push(
      `Profile built from ${profile.n_utterances} recordings, which is a solid baseline.`,
    );
  }

  if (profile.syllable_deletion_rate > 0.05) {
    bullets.push(
      `You drop syllables at about ${(profile.syllable_deletion_rate * 100).toFixed(0)}% — Phonemer will look out for missing word endings and short forms.`,
    );
  } else {
    bullets.push(
      `Your syllable structure is largely intact. Phonemer will rely mostly on phoneme-level corrections.`,
    );
  }

  if (parsed?.confusion_overall?.counts) {
    const top = topConfusions(parsed.confusion_overall, 3);
    if (top.length > 0) {
      bullets.push(
        `Most-frequent substitutions: ${top
          .map((t) => `/${t.intended}/ → /${t.produced}/`)
          .join(", ")}.`,
      );
    }
  }

  if (parsed?.cluster_simplifications) {
    const clusters = Object.keys(parsed.cluster_simplifications).slice(0, 3);
    if (clusters.length > 0) {
      bullets.push(
        `Consonant clusters Phonemer expects to be simplified: ${clusters.join(", ")}.`,
      );
    }
  }

  const intro =
    profile.opus_summary ??
    "Phonemer learned how you tend to produce phonemes when reading these sentences. Here's what stood out.";

  return { intro, bullets };
}

function topConfusions(
  matrix: { rows: string[]; cols: string[]; counts: number[][] },
  k: number,
): { intended: string; produced: string; count: number }[] {
  const out: { intended: string; produced: string; count: number }[] = [];
  for (let i = 0; i < matrix.counts.length; i++) {
    const row = matrix.counts[i];
    for (let j = 0; j < row.length; j++) {
      if (i === j || row[j] === 0) continue;
      out.push({ intended: matrix.rows[i], produced: matrix.cols[j], count: row[j] });
    }
  }
  out.sort((a, b) => b.count - a.count);
  return out.slice(0, k);
}
