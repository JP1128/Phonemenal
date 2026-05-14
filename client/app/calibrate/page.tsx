"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AccessibilityPanel } from "../../components/accessibility-panel";
import { useAccessibility } from "../../lib/accessibility";
import { createAudioCapture } from "../../lib/audio-capture";
import { createCalibrationClient } from "../../lib/calibration-client";
import { saveProfile } from "../../lib/profile";

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

type CalibrationListItem = {
  id: number;
  started_at: number;
  completed_at: number | null;
  notes: string | null;
  display_name: string | null;
  utterance_count: number;
  // Most recent training job for this calibration (if any). The server
  // joins on MAX(training_jobs.id) per calibration so resubmissions only
  // surface the latest attempt — see ``list_calibrations`` in db.py.
  last_training: {
    id: number;
    status: "queued" | "running" | "succeeded" | "failed";
    adapter_dir: string | null;
    completed_at: number | null;
  } | null;
};

type ExistingUtterance = {
  id: number;
  sentence_set_id: string;
  sentence_index: number;
  intended_text: string;
  duration_ms: number | null;
};

type TrainingJob = {
  id: number;
  calibration_id: number;
  adapter_dir: string;
  speaker_label: string | null;
  status: "queued" | "running" | "succeeded" | "failed";
  pid: number | null;
  log_path: string | null;
  eval_summary_json: string | null;
  error_message: string | null;
  started_at: number;
  completed_at: number | null;
  log_tail?: string;
  // Returned by POST /train-adapter so the banner can show what's in
  // the training set ("145 calibration + 12 corrected clips"). The
  // poll endpoint doesn't re-emit these; client-side we spread them
  // onto subsequent poll responses to keep the banner stable.
  calibration_clip_count?: number;
  feedback_clip_count?: number;
};

const TRAIN_MIN_UTTERANCES = 30;

type Stage = "welcome" | "wizard" | "building" | "confirmation" | "error";

type AutoState =
  | "idle"            // mic off / between recordings, waiting to arm
  | "warming"         // about to arm — short pause so the user can read the sentence
  | "listening"       // mic on, begin sent, waiting for speech onset
  | "speaking"        // speech detected, recording in progress
  | "committing"      // silence detected, end sent, waiting for commit ack
  | "all_done";

/**
 * Walk the sets in order and return the first (setIndex, sentenceIndex)
 * pair whose (sentence_set_id, sentence_index) isn't in `committed`.
 * Used on resume to skip past sentences the patient already recorded.
 * Falls back to the last position if everything is recorded.
 */
function firstUnrecorded(
  sets: SentenceSet[],
  committed: CommittedUtterance[],
): { setIndex: number; sentenceIndex: number } {
  const done = new Set(
    committed.map((c) => `${c.set_id}:${c.sentence_index}`),
  );
  for (let i = 0; i < sets.length; i++) {
    for (let j = 0; j < sets[i].sentences.length; j++) {
      if (!done.has(`${sets[i].id}:${j}`)) {
        return { setIndex: i, sentenceIndex: j };
      }
    }
  }
  // Everything done — point at the very last sentence (the all-done
  // effect picks up from there).
  const lastSet = Math.max(0, sets.length - 1);
  const lastSentence = Math.max(
    0,
    (sets[lastSet]?.sentences.length ?? 1) - 1,
  );
  return { setIndex: lastSet, sentenceIndex: lastSentence };
}

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
  const [displayName, setDisplayName] = useState<string | null>(null);
  const [setIndex, setSetIndex] = useState(0);
  const [sentenceIndex, setSentenceIndex] = useState(0);
  const [committed, setCommitted] = useState<CommittedUtterance[]>([]);
  const [autoState, setAutoState] = useState<AutoState>("idle");
  const [level, setLevel] = useState(0);
  const [profile, setProfile] = useState<ProfileSummary | null>(null);
  // Existing calibrations from the server, used to populate the resume
  // picker on the welcome screen. Refreshed every time we land on welcome.
  const [existing, setExisting] = useState<CalibrationListItem[] | null>(
    null,
  );
  // Active or most-recent training job for the current calibration. When
  // status is queued/running we poll /training-jobs/{id} every few seconds.
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

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

  const refreshExisting = useCallback(async () => {
    try {
      const r = await fetch(`${SERVER_HTTP}/calibrations`);
      if (!r.ok) return;
      const data = await r.json();
      setExisting(
        Array.isArray(data.calibrations) ? data.calibrations : [],
      );
    } catch {
      setExisting([]);
    }
  }, []);

  // Fire a training job for a calibration directly from the resume list
  // (no need to enter the wizard). After POST, mark the row optimistically
  // as queued so the UI updates immediately; the list-refresh effect below
  // polls every 4 s while anything is in-flight to surface real status.
  const trainFromWelcome = useCallback(
    async (calibrationId: number) => {
      setExisting((cur) =>
        cur
          ? cur.map((c) =>
              c.id === calibrationId
                ? {
                    ...c,
                    last_training: {
                      id: c.last_training?.id ?? -1,
                      status: "queued",
                      adapter_dir: c.last_training?.adapter_dir ?? null,
                      completed_at: null,
                    },
                  }
                : c,
            )
          : cur,
      );
      try {
        const r = await fetch(
          `${SERVER_HTTP}/calibrations/${calibrationId}/train-adapter`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({}),
          },
        );
        if (!r.ok) {
          await refreshExisting();
        }
      } catch {
        await refreshExisting();
      }
    },
    [refreshExisting],
  );

  // Patch a calibration's display label. Optimistic update first so the
  // edit pops back to read-mode immediately; we re-fetch on server error
  // to revert. Empty/whitespace string clears the label.
  const renameCalibration = useCallback(
    async (calibrationId: number, displayName: string) => {
      const normalized = displayName.trim();
      setExisting((cur) =>
        cur
          ? cur.map((c) =>
              c.id === calibrationId
                ? { ...c, display_name: normalized || null }
                : c,
            )
          : cur,
      );
      try {
        const r = await fetch(
          `${SERVER_HTTP}/calibrations/${calibrationId}`,
          {
            method: "PATCH",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ display_name: normalized }),
          },
        );
        if (!r.ok) {
          await refreshExisting();
        }
      } catch {
        await refreshExisting();
      }
    },
    [refreshExisting],
  );

  // Optimistic delete: drop from the list first, then DELETE. On error
  // we re-fetch so the row reappears if the server still has it.
  const deleteCalibration = useCallback(
    async (calibrationId: number) => {
      setExisting((cur) =>
        cur ? cur.filter((c) => c.id !== calibrationId) : cur,
      );
      try {
        const r = await fetch(
          `${SERVER_HTTP}/calibrations/${calibrationId}`,
          { method: "DELETE" },
        );
        if (!r.ok && r.status !== 404) {
          await refreshExisting();
        }
      } catch {
        await refreshExisting();
      }
    },
    [refreshExisting],
  );

  // Refresh the resume picker whenever we (re)land on the welcome screen
  // so freshly-created calibrations from other tabs / sessions show up.
  useEffect(() => {
    if (stage === "welcome") {
      refreshExisting();
    }
  }, [stage, refreshExisting]);

  // While any visible row has a queued/running training job, poll the
  // list every 4 s so the badge transitions to succeeded/failed without
  // requiring a manual refresh. Stops as soon as no row is in-flight.
  useEffect(() => {
    if (stage !== "welcome" || !existing) return;
    const anyInFlight = existing.some(
      (c) =>
        c.last_training?.status === "queued" ||
        c.last_training?.status === "running",
    );
    if (!anyInFlight) return;
    const id = setInterval(refreshExisting, 4_000);
    return () => clearInterval(id);
  }, [stage, existing, refreshExisting]);

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

  const beginCalibration = useCallback(
    async (opts?: { resumeId?: number; displayName?: string | null }) => {
      const resumeId = opts?.resumeId;
      const displayNameInput =
        opts?.displayName != null ? opts.displayName.trim() : "";

      setError(null);
      setProfile(null);
      setCommitted([]);
      setSetIndex(0);
      setSentenceIndex(0);
      setIndexRef.current = 0;
      sentenceIndexRef.current = 0;
      setAuto("idle");
      try {
        const body: Record<string, unknown> = {};
        if (resumeId != null) body.calibration_id = resumeId;
        if (displayNameInput) body.display_name = displayNameInput;
        const res = await fetch(`${SERVER_HTTP}/calibration/start`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setCalibrationId(data.calibration_id);
        setDisplayName(data.display_name ?? null);
        setSets(data.sentence_sets);
        setsRef.current = data.sentence_sets;

        // Resume path: pull what was already recorded so the wizard
        // shows real progress and skips past sentences the patient has
        // already done in a previous session.
        if (resumeId != null) {
          try {
            const detailRes = await fetch(
              `${SERVER_HTTP}/calibration/${data.calibration_id}`,
            );
            if (detailRes.ok) {
              const detail = await detailRes.json();
              const utts = (detail.utterances ?? []) as ExistingUtterance[];
              const preloaded: CommittedUtterance[] = utts.map((u) => ({
                set_id: u.sentence_set_id,
                sentence_index: u.sentence_index,
                utterance_id: u.id,
                duration_ms: u.duration_ms ?? 0,
                intended_text: u.intended_text,
              }));
              setCommitted(preloaded);
              const start = firstUnrecorded(
                data.sentence_sets as SentenceSet[],
                preloaded,
              );
              setSetIndex(start.setIndex);
              setSentenceIndex(start.sentenceIndex);
              setIndexRef.current = start.setIndex;
              sentenceIndexRef.current = start.sentenceIndex;
            }
          } catch {
            // best-effort — wizard still works, it just won't skip
          }
        }

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
    },
    [advance, setAuto],
  );

  // Auto-finish once everything is recorded.
  useEffect(() => {
    if (allDone && stage === "wizard") {
      // Stop the mic; there's nothing left to record.
      captureRef.current?.stop();
      captureRef.current = null;
      setAuto("all_done");
    }
  }, [allDone, stage, setAuto]);

  // ============================================================ training

  const startTraining = useCallback(async () => {
    if (!calibrationId) return;
    setTrainingError(null);
    try {
      const res = await fetch(
        `${SERVER_HTTP}/calibrations/${calibrationId}/train-adapter`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as TrainingJob;
      setTrainingJob(data);
    } catch (e) {
      setTrainingError(
        e instanceof Error ? e.message : String(e),
      );
    }
  }, [calibrationId]);

  // Poll the active training job every 3 s until it terminates.
  useEffect(() => {
    if (!trainingJob) return;
    if (
      trainingJob.status === "succeeded" ||
      trainingJob.status === "failed"
    ) {
      return;
    }
    const jobId = trainingJob.id;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch(`${SERVER_HTTP}/training-jobs/${jobId}`);
        if (!r.ok) return;
        const data = (await r.json()) as TrainingJob;
        if (cancelled) return;
        // /training-jobs/{id} doesn't re-emit the clip counts that came
        // back from the POST. Spread the prior values forward so the
        // banner's composition summary stays stable across polls.
        setTrainingJob((prev) => ({
          ...data,
          calibration_clip_count:
            data.calibration_clip_count ?? prev?.calibration_clip_count,
          feedback_clip_count:
            data.feedback_clip_count ?? prev?.feedback_clip_count,
        }));
      } catch {
        // poll error — keep going; the loop interval handles retry
      }
    };
    const id = setInterval(tick, 3_000);
    tick();
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [trainingJob]);

  // On mount / when calibrationId changes, fetch any existing training
  // job(s) for this calibration so reopening the page resumes the
  // status display instead of starting fresh.
  useEffect(() => {
    if (!calibrationId) {
      setTrainingJob(null);
      return;
    }
    fetch(`${SERVER_HTTP}/calibrations/${calibrationId}/training-jobs`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => {
        const jobs = (data?.jobs ?? []) as TrainingJob[];
        // Most recent first per backend ordering — pick the newest.
        if (jobs.length > 0) setTrainingJob(jobs[0]);
        else setTrainingJob(null);
      })
      .catch(() => setTrainingJob(null));
  }, [calibrationId]);

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
      <header className="flex items-center gap-3 max-w-3xl mx-auto mb-6 min-h-[28px]">
        <span className="text-[10px] tracking-[0.3em] font-mono opacity-50 uppercase">
          phonemer
        </span>
        <span className="text-[10px] tracking-[0.25em] font-mono opacity-40 uppercase">
          · calibration
        </span>
        {calibrationId && (
          <span className="text-xs opacity-70 font-mono">
            {displayName ? displayName : `cal #${calibrationId}`}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2 text-xs">
          {stage === "wizard" && calibrationId && (
            <>
              {(() => {
                const canTrain =
                  committed.length >= TRAIN_MIN_UTTERANCES &&
                  (!trainingJob ||
                    trainingJob.status === "succeeded" ||
                    trainingJob.status === "failed");
                const inFlight =
                  trainingJob &&
                  (trainingJob.status === "queued" ||
                    trainingJob.status === "running");
                return (
                  <button
                    onClick={startTraining}
                    disabled={!canTrain}
                    title={
                      inFlight
                        ? "A training job is currently running"
                        : committed.length < TRAIN_MIN_UTTERANCES
                          ? `Record at least ${TRAIN_MIN_UTTERANCES} clips to train (${committed.length} so far)`
                          : "Train a per-patient Whisper adapter from this calibration"
                    }
                    className={`px-2 py-0.5 rounded-md ${
                      canTrain
                        ? "bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-100"
                        : "bg-white/5 opacity-50 cursor-not-allowed"
                    }`}
                  >
                    {inFlight ? "Training…" : "Train adapter"}
                  </button>
                );
              })()}
              <button
                onClick={() => {
                  // End the recording session locally. The calibration
                  // row stays on the server — the doctor can resume from
                  // the welcome screen.
                  captureRef.current?.stop();
                  captureRef.current = null;
                  calRef.current?.disconnect();
                  calRef.current = null;
                  setAuto("idle");
                  setCalibrationId(null);
                  setDisplayName(null);
                  setCommitted([]);
                  setStage("welcome");
                }}
                className="px-2 py-0.5 rounded-md bg-white/5 hover:bg-white/10 opacity-80 hover:opacity-100"
                title="Save and exit. You can resume this patient from the welcome screen."
              >
                End session
              </button>
            </>
          )}
          <AccessibilityPanel
            settings={a11y.settings}
            onChange={a11y.setSettings}
          />
          <a
            href="/"
            className="px-2 py-0.5 rounded-md bg-white/5 hover:bg-white/10 opacity-80 hover:opacity-100"
          >
            ← Home
          </a>
        </div>
      </header>

      {(trainingJob || trainingError) && stage === "wizard" && (
        <div className="max-w-3xl mx-auto mb-4">
          <TrainingBanner
            job={trainingJob}
            error={trainingError}
            onDismiss={() => {
              setTrainingJob(null);
              setTrainingError(null);
            }}
          />
        </div>
      )}

      <div className="max-w-3xl mx-auto">
        {stage === "welcome" && (
          <WelcomeScreen
            sets={sets}
            existing={existing}
            error={error}
            onStartNew={(name) => beginCalibration({ displayName: name })}
            onResume={(id) => beginCalibration({ resumeId: id })}
            onDelete={deleteCalibration}
            onTrain={trainFromWelcome}
            onRename={renameCalibration}
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
            trainingJob={trainingJob}
            onTrain={startTraining}
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

    </main>
  );
}

// ============================================================ welcome

function WelcomeScreen({
  sets,
  existing,
  error,
  onStartNew,
  onResume,
  onDelete,
  onTrain,
  onRename,
}: {
  sets: SentenceSet[] | null;
  existing: CalibrationListItem[] | null;
  error: string | null;
  onStartNew: (displayName: string) => void;
  onResume: (calibrationId: number) => void;
  onDelete: (calibrationId: number) => void;
  onTrain: (calibrationId: number) => void;
  onRename: (calibrationId: number, displayName: string) => void;
}) {
  const [newName, setNewName] = useState("");
  // Empty 0-clip sessions accumulate when someone clicks "Start new" and
  // walks away without recording anything. Hide them by default so the
  // resume list only shows real patient work; expose a toggle for the
  // edge case where someone just opened a session a moment ago and
  // hasn't recorded yet but wants to come back to it.
  const [showEmpty, setShowEmpty] = useState(false);
  const totalSentences = (sets ?? []).reduce(
    (n, s) => n + s.sentences.length,
    0,
  );
  // Sort: in-progress first (no completed_at), then completed.
  const allSorted = (existing ?? [])
    .slice()
    .sort((a, b) => {
      const aDone = a.completed_at ? 1 : 0;
      const bDone = b.completed_at ? 1 : 0;
      if (aDone !== bDone) return aDone - bDone;
      return b.started_at - a.started_at;
    });
  const emptyCount = allSorted.filter((c) => c.utterance_count === 0).length;
  const sortedExisting = showEmpty
    ? allSorted
    : allSorted.filter((c) => c.utterance_count > 0);

  return (
    <div className="space-y-8">
      <div className="space-y-3 text-center">
        <h2 className="text-3xl font-semibold">Calibrate Phonemer</h2>
        <p className="opacity-80 max-w-xl mx-auto">
          Record a patient reading short sentences so the transcription model
          can be tuned to their voice. Each session is 10–20 minutes; come
          back over multiple days and resume where you left off.
        </p>
      </div>

      <div className="rounded-lg border border-white/10 p-5 space-y-3">
        <h3 className="text-lg font-semibold">Start a new patient</h3>
        <p className="text-sm opacity-70 leading-snug">
          Label the patient so you can find them again later — a room
          number, name, or study id all work. If you skip it, the row
          will show up as “Calibration #&lt;id&gt;”.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!sets) return;
            onStartNew(newName.trim());
          }}
          className="flex flex-wrap gap-2 items-center"
        >
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            autoFocus
            placeholder="e.g. Room 412 · Jane D. · study-104"
            className="flex-1 min-w-[200px] px-3 py-2 rounded-md bg-black/40 border border-white/10 focus:outline-none focus:border-teal-400/40 text-sm"
          />
          <button
            type="submit"
            disabled={!sets}
            className="px-5 py-2 rounded-md disabled:opacity-50 font-semibold text-sm"
            style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
          >
            {newName.trim() ? "Start calibration" : "Start without label"}
          </button>
        </form>
      </div>

      {allSorted.length > 0 && (
        <div className="rounded-lg border border-white/10 p-5 space-y-3">
          <div className="flex items-baseline justify-between gap-3 flex-wrap">
            <h3 className="text-lg font-semibold">
              Resume an existing patient
            </h3>
            {emptyCount > 0 && (
              <button
                onClick={() => setShowEmpty((v) => !v)}
                className="text-xs opacity-70 hover:opacity-100 underline-offset-2 hover:underline"
              >
                {showEmpty
                  ? `Hide empty (${emptyCount})`
                  : `Show empty (${emptyCount})`}
              </button>
            )}
          </div>
          {sortedExisting.length === 0 ? (
            <div className="text-sm opacity-60 italic">
              No patients with recorded clips yet.
              {emptyCount > 0 &&
                " Use “Show empty” to see calibration rows that were started but never recorded."}
            </div>
          ) : (
            <ul className="divide-y divide-white/5">
              {sortedExisting.map((c) => {
                const done = c.completed_at != null;
                const pct = totalSentences
                  ? Math.min(
                      100,
                      Math.round((c.utterance_count / totalSentences) * 100),
                    )
                  : 0;
                const tj = c.last_training;
                const trainingInFlight =
                  tj?.status === "queued" || tj?.status === "running";
                const trainable =
                  c.utterance_count >= TRAIN_MIN_UTTERANCES &&
                  !trainingInFlight;
                return (
                  <li
                    key={c.id}
                    className="py-3 flex flex-wrap items-center gap-3"
                  >
                    <div className="flex-1 min-w-[200px]">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <EditableCalibrationLabel
                          calibrationId={c.id}
                          displayName={c.display_name}
                          onRename={onRename}
                        />
                        <span className="text-[10px] font-mono opacity-50">
                          cal #{c.id}
                        </span>
                        {done && (
                          <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-emerald-400/30 bg-emerald-400/10 text-emerald-200">
                            completed
                          </span>
                        )}
                        <TrainingStatusChip tj={tj} />
                      </div>
                      <div className="text-xs opacity-60 mt-0.5">
                        {c.utterance_count} clip
                        {c.utterance_count === 1 ? "" : "s"}
                        {totalSentences > 0 &&
                          ` · ~${pct}% of ${totalSentences}`}
                        {" · "}
                        started{" "}
                        {new Date(c.started_at * 1000).toLocaleString()}
                      </div>
                      {totalSentences > 0 && c.utterance_count > 0 && (
                        <div
                          className="h-1 mt-1.5 rounded bg-white/5 max-w-[260px]"
                          aria-hidden
                        >
                          <div
                            className="h-1 rounded transition-all"
                            style={{
                              width: `${pct}%`,
                              background:
                                c.utterance_count >= TRAIN_MIN_UTTERANCES
                                  ? "rgb(52 211 153 / 0.7)"
                                  : "rgb(148 163 184 / 0.5)",
                            }}
                          />
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => onResume(c.id)}
                      disabled={!sets}
                      className="px-3 py-1.5 rounded-md text-sm border border-white/15 hover:bg-white/[0.06] disabled:opacity-50"
                    >
                      {done ? "Reopen" : "Resume"}
                    </button>
                    {tj?.status === "succeeded" && tj.adapter_dir && (
                      <a
                        href="/"
                        onClick={() => {
                          const slug = tj.adapter_dir
                            ?.split("/")
                            .filter(Boolean)
                            .pop();
                          if (slug) {
                            saveProfile({
                              kind: "calibrated",
                              speakerId: slug,
                            });
                          }
                        }}
                        title="Open the home page with this patient's adapter selected"
                        className="px-3 py-1.5 rounded-md text-sm border border-emerald-400/40 text-emerald-200 hover:bg-emerald-400/10"
                      >
                        Use →
                      </a>
                    )}
                    {(trainable || trainingInFlight) && (
                      <button
                        onClick={() => onTrain(c.id)}
                        disabled={trainingInFlight}
                        title={
                          trainingInFlight
                            ? "A training job is currently running"
                            : tj?.status === "succeeded"
                              ? "Re-train using all current recordings"
                              : "Fine-tune a per-patient Whisper adapter"
                        }
                        className="px-3 py-1.5 rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-wait"
                        style={{
                          background: "var(--phonemer-accent)",
                          color: "#0a0a0a",
                        }}
                      >
                        {trainingInFlight
                          ? "Training…"
                          : tj?.status === "succeeded"
                            ? "Re-train"
                            : "Train"}
                      </button>
                    )}
                    <ConfirmDeleteButton
                      onConfirm={() => onDelete(c.id)}
                      losesClips={c.utterance_count}
                    />
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <PhaseCard
          step={1}
          label="Record"
          body="Patient reads the prompts. The mic auto-advances on silence; sessions are resumable across multiple sittings."
        />
        <PhaseCard
          step={2}
          label="Train"
          body="Once ≥30 clips are recorded, a clinician triggers training. A LoRA adapter is fine-tuned on top of frozen Whisper-small.en."
        />
        <PhaseCard
          step={3}
          label="Use"
          body="The new adapter appears in the home-page profile dropdown. Selecting it routes live transcription through the patient's adapter."
        />
      </div>

      {sets && (
        <div className="rounded-lg border border-white/10 p-4 space-y-2">
          <div className="text-sm opacity-70">
            {totalSentences} prompts across {sets.length} sets:
          </div>
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
        <div className="rounded-md bg-rose-900/40 px-3 py-2 text-rose-200 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}

// Click-to-edit label. Used in the resume picker so a clinician who
// skipped the label at start can attach one (or fix a typo) without
// blowing the row away. Saves on Enter or blur; cancels on Escape.
function EditableCalibrationLabel({
  calibrationId,
  displayName,
  onRename,
}: {
  calibrationId: number;
  displayName: string | null;
  onRename: (calibrationId: number, displayName: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing) {
      setDraft(displayName ?? "");
      requestAnimationFrame(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      });
    }
  }, [editing, displayName]);

  const commit = () => {
    setEditing(false);
    const next = draft.trim();
    const prev = displayName ?? "";
    if (next === prev) return;
    onRename(calibrationId, next);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setEditing(false);
          }
        }}
        placeholder="Patient label"
        className="font-medium bg-black/40 border border-teal-400/40 rounded px-1.5 py-0 text-sm focus:outline-none focus:border-teal-400"
      />
    );
  }

  const label = displayName || `Calibration #${calibrationId}`;
  const unlabeled = !displayName;
  return (
    <button
      type="button"
      onClick={() => setEditing(true)}
      title="Click to rename"
      className={`font-medium text-left rounded px-1 -mx-1 hover:bg-white/[0.06] ${
        unlabeled ? "italic opacity-70" : ""
      }`}
    >
      {label}
    </button>
  );
}

// Two-stage delete: first click arms, second click within 4 s confirms.
// Auto-disarms on timeout or when the user clicks outside the button.
// Chosen over native confirm() because the dark UI looks broken next to
// a system modal; chosen over "Delete + Cancel" inline because the row
// is already tight on horizontal space.
function ConfirmDeleteButton({
  onConfirm,
  losesClips,
}: {
  onConfirm: () => void;
  losesClips: number;
}) {
  const [armed, setArmed] = useState(false);
  const btnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!armed) return;
    const onDocClick = (e: MouseEvent) => {
      if (btnRef.current && !btnRef.current.contains(e.target as Node)) {
        setArmed(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setArmed(false);
    };
    const timer = window.setTimeout(() => setArmed(false), 4000);
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [armed]);

  const handle = () => {
    if (!armed) {
      setArmed(true);
      return;
    }
    setArmed(false);
    onConfirm();
  };

  const armedTitle =
    losesClips > 0
      ? `Click again to confirm — ${losesClips} clip${
          losesClips === 1 ? "" : "s"
        } will be lost`
      : "Click again to confirm";

  return (
    <button
      ref={btnRef}
      onClick={handle}
      title={armed ? armedTitle : "Delete this calibration row"}
      className={`px-2 py-1.5 rounded-md text-xs transition-colors ${
        armed
          ? "border border-rose-400/60 bg-rose-500/20 text-rose-100"
          : "border border-rose-400/20 text-rose-200/80 hover:bg-rose-400/10 hover:text-rose-100"
      }`}
    >
      {armed ? "Confirm delete" : "Delete"}
    </button>
  );
}

function TrainingStatusChip({
  tj,
}: {
  tj: CalibrationListItem["last_training"];
}) {
  if (!tj) return null;
  const map: Record<
    NonNullable<CalibrationListItem["last_training"]>["status"],
    { label: string; cls: string }
  > = {
    queued: {
      label: "training queued",
      cls: "border-amber-400/30 bg-amber-400/10 text-amber-200",
    },
    running: {
      label: "training…",
      cls: "border-amber-400/30 bg-amber-400/10 text-amber-200",
    },
    succeeded: {
      label: "trained",
      cls: "border-sky-400/30 bg-sky-400/10 text-sky-200",
    },
    failed: {
      label: "train failed",
      cls: "border-rose-400/30 bg-rose-400/10 text-rose-200",
    },
  };
  const entry = map[tj.status];
  // Show just the slug (last path segment) in the hover title — full
  // absolute paths are too long to scan in a tooltip and rarely useful.
  const slug = tj.adapter_dir?.split("/").filter(Boolean).pop();
  return (
    <span
      title={slug ? `${entry.label} · adapter: ${slug}` : entry.label}
      className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${entry.cls}`}
    >
      {entry.label}
    </span>
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
        <div className="rounded-md bg-rose-900/40 px-3 py-2 text-rose-200 text-sm text-center">
          {props.error}
        </div>
      )}

      <div className="flex items-center justify-center gap-3 flex-wrap">
        {(props.autoState === "listening" || props.autoState === "speaking") && (
          <button
            onClick={props.onFinalize}
            className="px-4 py-2 rounded-md text-sm font-medium"
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
            Done speaking
          </button>
        )}
        <button
          onClick={props.onSkip}
          className="px-3 py-1.5 rounded-md text-xs border border-white/10 hover:bg-white/5"
        >
          Skip this sentence
        </button>
        {props.canRedo && (
          <>
            <button
              onClick={props.onReplay}
              className="px-3 py-1.5 rounded-md text-xs border border-white/10 hover:bg-white/5"
            >
              Replay last
            </button>
            <button
              onClick={props.onRedo}
              className="px-3 py-1.5 rounded-md text-xs border border-white/10 hover:bg-white/5"
            >
              Redo previous
            </button>
          </>
        )}
      </div>

      <div className="text-center text-xs font-mono opacity-50 tabular-nums">
        sentence {props.sentenceIndex + 1}/{props.sentenceCountInSet}
      </div>
      <details className="max-w-md mx-auto text-center">
        <summary className="cursor-pointer text-[10px] font-mono opacity-30 hover:opacity-70 uppercase tracking-wider list-none">
          show VAD diagnostics
        </summary>
        <div className="mt-1 text-[10px] font-mono opacity-50 tabular-nums">
          floor {props.noiseFloor.toFixed(3)} · voice ≥{" "}
          {props.voiceThreshold.toFixed(3)} · silence ≤{" "}
          {props.silenceThreshold.toFixed(3)} · level{" "}
          {props.level.toFixed(3)}
        </div>
      </details>
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
  trainingJob,
  onTrain,
  onReplay,
  onBuild,
  onRedoSession,
}: {
  committed: number;
  trainingJob: TrainingJob | null;
  onTrain: () => void;
  onReplay: () => void;
  onBuild: () => void;
  onRedoSession: () => void;
}) {
  // Adapter training is the value-producing artifact (drives live
  // transcription). The distortion profile is a severity diagnostic —
  // kept around for therapy tracking but not the primary CTA.
  const enoughClips = committed >= TRAIN_MIN_UTTERANCES;
  const inFlight =
    trainingJob?.status === "queued" || trainingJob?.status === "running";
  const succeeded = trainingJob?.status === "succeeded";
  const failed = trainingJob?.status === "failed";
  const canTrain = enoughClips && !inFlight;
  const trainLabel = inFlight
    ? "Training…"
    : succeeded
      ? "Re-train adapter"
      : failed
        ? "Retry training"
        : "Train adapter";

  return (
    <div className="text-center space-y-6 py-8">
      <div className="text-5xl">✓</div>
      <div className="space-y-1">
        <div className="text-2xl font-semibold">All sentences recorded</div>
        <div className="opacity-70 text-sm">
          {committed} recordings ready. Training the patient adapter takes
          75–120 minutes; the banner above shows live progress.
        </div>
        {succeeded && (
          <div className="text-sm text-emerald-300 mt-2">
            This patient's adapter is ready — pick them from the profile
            menu on the home page to use it.
          </div>
        )}
      </div>

      <div className="flex justify-center gap-3 flex-wrap">
        <button
          onClick={onTrain}
          disabled={!canTrain}
          title={
            inFlight
              ? "A training job is currently running"
              : !enoughClips
                ? `Record at least ${TRAIN_MIN_UTTERANCES} clips to train (${committed} so far)`
                : succeeded
                  ? "Train a fresh adapter using all current recordings"
                  : "Fine-tune a per-patient Whisper adapter from this calibration"
          }
          className="px-6 py-2.5 rounded-md text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
          style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
        >
          {trainLabel}
        </button>
        {succeeded && (
          <a
            href="/"
            onClick={() => {
              // Pre-select the freshly-trained adapter on the home page.
              // The slug is the basename of adapter_dir (the trainer
              // safe-slugs display_name / cal_<id> identically on both
              // sides). Home's reconcileProfile gracefully falls back
              // to demo if the slug isn't in /adapters yet — but by
              // the time the user clicks here, training_jobs.status is
              // 'succeeded' which means the filesystem write has
              // finished, so /adapters will include it on next refresh.
              const dir = trainingJob?.adapter_dir;
              if (dir) {
                const slug = dir.split("/").filter(Boolean).pop();
                if (slug) {
                  saveProfile({ kind: "calibrated", speakerId: slug });
                }
              }
            }}
            className="px-4 py-2.5 rounded-md text-sm border border-emerald-400/40 text-emerald-200 hover:bg-emerald-400/10"
          >
            Use this patient →
          </a>
        )}
      </div>

      <div className="flex justify-center gap-3 flex-wrap text-sm">
        <button
          onClick={onReplay}
          className="px-3 py-1.5 rounded-md border border-white/10 hover:bg-white/5"
        >
          Replay last
        </button>
        <button
          onClick={onRedoSession}
          className="px-3 py-1.5 rounded-md border border-white/10 hover:bg-white/5"
        >
          Start over
        </button>
      </div>

      <details className="text-left max-w-md mx-auto pt-2 opacity-70">
        <summary className="cursor-pointer text-xs uppercase tracking-wider opacity-80 hover:opacity-100">
          Run severity analysis (optional)
        </summary>
        <div className="text-xs opacity-80 mt-2 leading-relaxed">
          Builds a separate severity / therapy-tracking profile (syllable
          deletion, phoneme confusion patterns) from these recordings.
          Doesn't affect transcription accuracy.
        </div>
        <button
          onClick={onBuild}
          className="mt-2 px-3 py-1.5 rounded-md text-xs border border-white/10 hover:bg-white/5"
        >
          Build severity profile
        </button>
      </details>
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
      <div className="text-lg font-semibold">Analyzing speech patterns</div>
      <div className="opacity-70 text-sm max-w-md mx-auto">
        Aligning each recording against its intended phonemes to identify
        systematic patterns in articulation.
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
      <div className="text-center space-y-2">
        <div className="text-[10px] tracking-[0.3em] font-mono uppercase opacity-50">
          severity profile
        </div>
        <div className="text-2xl font-semibold">Profile is ready</div>
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
          <div className="text-xs font-mono opacity-60 mb-1">
            Clinical summary
          </div>
          <div className="text-sm leading-relaxed">{profile.opus_summary}</div>
        </div>
      )}

      <div className="flex justify-center gap-3 flex-wrap pt-2">
        <a
          href="/"
          className="px-5 py-2 rounded-md text-sm font-semibold"
          style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
        >
          Back to home →
        </a>
        <button
          onClick={onRedoSession}
          className="px-3 py-2 rounded-md text-sm border border-white/10 hover:bg-white/5"
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
      <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-rose-500/15 border border-rose-500/40 text-rose-200 font-mono text-xl">
        !
      </div>
      <div className="text-xl font-semibold">Something went wrong</div>
      <div className="text-sm text-rose-200 max-w-lg mx-auto">{error}</div>
      <button
        onClick={onRetry}
        className="px-4 py-2 rounded-md text-sm bg-emerald-600 hover:bg-emerald-500"
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

// ============================================================ training banner

function TrainingBanner({
  job,
  error,
  onDismiss,
}: {
  job: TrainingJob | null;
  error: string | null;
  onDismiss: () => void;
}) {
  if (error && !job) {
    return (
      <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-3 text-rose-100">
        <div className="flex items-center gap-2 text-sm font-medium">
          <span>Could not start training.</span>
          <button
            onClick={onDismiss}
            className="ml-auto opacity-60 hover:opacity-100 text-xs"
          >
            ✕
          </button>
        </div>
        <div className="text-xs opacity-80 mt-1 font-mono whitespace-pre-wrap">
          {error}
        </div>
      </div>
    );
  }
  if (!job) return null;

  const tone =
    job.status === "succeeded"
      ? "border-emerald-400/40 bg-emerald-950/30 text-emerald-100"
      : job.status === "failed"
        ? "border-rose-500/40 bg-rose-950/30 text-rose-100"
        : "border-amber-400/40 bg-amber-950/20 text-amber-100";

  const inFlight = job.status === "queued" || job.status === "running";
  const label =
    job.status === "queued"
      ? "Training queued…"
      : job.status === "running"
        ? "Training in progress"
        : job.status === "succeeded"
          ? "Adapter trained ✓"
          : "Training failed";

  // Composition summary: how the training set is built up from prompted
  // calibration clips vs listener-validated live clips. Surfacing this
  // gives clinicians visibility that their "Looks right" / Edit clicks
  // actually feed back into the model on the next retrain.
  const calClips = job.calibration_clip_count ?? null;
  const fbClips = job.feedback_clip_count ?? null;
  const showComposition = calClips !== null || fbClips !== null;

  return (
    <div className={`rounded-lg border p-3 ${tone}`}>
      <div className="flex items-center gap-2 text-sm font-medium">
        {inFlight && <Spinner />}
        <span>{label}</span>
        <span className="opacity-50 text-[10px] font-mono">
          job #{job.id}
        </span>
        {job.adapter_dir && (
          <span className="ml-2 text-[10px] font-mono opacity-50 truncate">
            → {job.adapter_dir}
          </span>
        )}
        <button
          onClick={onDismiss}
          className="ml-auto opacity-60 hover:opacity-100 text-xs"
          title="Hide this banner. Training continues in the background."
        >
          ✕
        </button>
      </div>
      {showComposition && (
        <div className="text-xs opacity-80 mt-1">
          training on{" "}
          <span className="font-mono tabular-nums">
            {calClips ?? 0}
          </span>{" "}
          calibration{calClips === 1 ? " clip" : " clips"}
          {fbClips !== null && fbClips > 0 && (
            <>
              {" + "}
              <span className="font-mono tabular-nums text-emerald-300">
                {fbClips}
              </span>{" "}
              listener-validated{fbClips === 1 ? " clip" : " clips"}{" "}
              <span className="opacity-50">
                (confirmed / corrected from prior sessions)
              </span>
            </>
          )}
        </div>
      )}
      {job.status === "succeeded" && (
        <div className="text-xs opacity-80 mt-1">
          The new profile will show up in the home page's profile selector.
        </div>
      )}
      {job.status === "failed" && job.error_message && (
        <div className="text-xs opacity-80 mt-1 font-mono whitespace-pre-wrap">
          {job.error_message}
        </div>
      )}
      {job.log_tail && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[10px] opacity-60 hover:opacity-100">
            training log (tail)
          </summary>
          <pre className="mt-1 text-[10px] font-mono opacity-70 whitespace-pre-wrap max-h-48 overflow-y-auto bg-black/30 rounded-md p-2 leading-snug">
            {job.log_tail}
          </pre>
        </details>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <span
      className="inline-block w-3 h-3 rounded-full border-2 border-current border-t-transparent animate-spin"
      aria-hidden
    />
  );
}
