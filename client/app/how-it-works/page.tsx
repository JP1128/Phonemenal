"use client";

// MAINTENANCE NOTE — this page is the user-facing explanation of the live
// pipeline. The slides below describe what each stage does and show data
// from a real run. KEEP THE SLIDE DESCRIPTIONS IN SYNC with the
// corresponding server code:
//
//   Slide 1 Audio input        ← server/app/audio.py  + main.py WS handler
//   Slide 2 Voice Activity     ← server/app/silero_vad.py + vad.py
//   Slide 3 Speaker ID         ← server/app/speaker_id.py
//   Slide 4 Phoneme extraction ← server/app/phonemes.py
//   Slide 5 Whisper + adapter  ← server/app/stt.py (Transcriber, AdaptedTranscriber)
//   Slide 6 Safety gate        ← server/app/stt.py (GatedTranscriber)
//   Slide 7 Final output       ← server/app/main.py _gate_and_emit
//
// If you change a model, threshold, or pipeline stage on the server side,
// update the matching slide in the same commit. See CLAUDE.md "Documentation
// invariant" for the full rule and the most-common changes that need it.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPhonemerWs } from "../../lib/phonemer-ws";
import type {
  PhonemeWindowMessage,
  TranslationMessage,
  UtteranceMessage,
  WsMessage,
} from "../../lib/messages";

// ──────────────────────────────────────────────────────────── shared utilities

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";

const audioCtxRef: { current: AudioContext | null } = { current: null };
function getAudioContext(): AudioContext {
  if (audioCtxRef.current) return audioCtxRef.current;
  const Ctor =
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext || window.AudioContext;
  audioCtxRef.current = new Ctor();
  return audioCtxRef.current;
}

type TorgoClip = {
  id: number;
  speaker_id: string;
  condition: string;
  intelligibility: string | null;
  prompt_set: string | null;
  intended_text: string | null;
  duration_ms: number;
};

type PipelineState = {
  status: "idle" | "running" | "done" | "error";
  vanillaText: string | null;
  vanillaError: string | null;
  // Multiple per clip — VAD splits clips with internal pauses (e.g. F03 reading
  // "If you are losing water [pause] replace it immediately") into separate
  // utterances. Each utterance produces its own phoneme_window + translation,
  // and the page must show ALL of them or it'll silently drop half the clip.
  utterances: UtteranceMessage[];
  phonemeFramesByUtt: Record<number, PhonemeWindowMessage["frames"]>;
  translations: TranslationMessage[];
  errorMessage: string | null;
  startedAt: number | null;
  finishedAt: number | null;
  waveform: Float32Array | null;
  audioDurationMs: number | null;
};

const initialPipelineState: PipelineState = {
  status: "idle",
  vanillaText: null,
  vanillaError: null,
  utterances: [],
  phonemeFramesByUtt: {},
  translations: [],
  errorMessage: null,
  startedAt: null,
  finishedAt: null,
  waveform: null,
  audioDurationMs: null,
};

// Presentation flow:
//   Phase 1 (Problem)      — what we're up against
//   Phase 2 (Architecture) — the system we built
//   Phase 3 (Calibration)  — the trained component, deep-dived
//   Phase 4 (Live demo)    — drive a real clip through it
//
// Phase identity is a vibe (color), surfaced in the slide nav and in
// each slide's accent. Helps an audience locate "where are we" mid-talk.
//
// `steps` lets a single slide have internal pages — pressing → advances
// the inner step before advancing the slide. The "problem" slide uses
// this so the same patient cards morph from "what they said" to "what
// whisper heard" without a hard slide transition.
const SLIDES = [
  { id: "title",        title: "Intro",                     phase: "problem",      steps: 1 },
  { id: "problem",      title: "Patients & off-the-shelf Whisper", phase: "problem", steps: 2 },
  { id: "stack",        title: "What it's built with",      phase: "architecture", steps: 1 },
  { id: "architecture", title: "The Phonemer pipeline",     phase: "architecture", steps: 1 },
  { id: "strengths",    title: "The two strengths",         phase: "calibration",  steps: 1 },
  { id: "demo",         title: "Live demo · the difference", phase: "demo",        steps: 1 },
] as const;

const PHASE_META: Record<string, { label: string; color: string }> = {
  problem:      { label: "1 · The problem",      color: "amber"   },
  architecture: { label: "2 · The system",       color: "cyan"    },
  calibration:  { label: "3 · The two strengths", color: "violet" },
  demo:         { label: "4 · Live demo",        color: "emerald" },
};

const PHASE_ACCENT_HEX: Record<string, string> = {
  problem:      "#f59e0b", // amber-500
  architecture: "#06b6d4", // cyan-500
  calibration:  "#8b5cf6", // violet-500
  demo:         "#10b981", // emerald-500
};

// Wrap fetch with a label so any "Failed to fetch" surfaces *which* URL
// failed. The browser's native TypeError doesn't say. Adds a 30 s timeout
// guard — pipeline runs are bounded; an indefinite hang isn't useful.
async function fetchWithLabel(
  label: string,
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  try {
    const r = await fetch(url, { ...(init ?? {}), signal: controller.signal });
    return r;
  } catch (e) {
    const err = e as Error;
    // Re-throw with the label so the message is actionable in the UI.
    if (err.name === "AbortError") {
      throw new Error(`${label}: timed out after 30 s (${url})`);
    }
    throw new Error(`${label}: ${err.message} (${url})`);
  } finally {
    clearTimeout(timer);
  }
}

// Auto-start audio when this hook is mounted (i.e. when the slide becomes
// active). Resets to t=0 every time. Pauses on unmount. Fails silently if
// the browser blocks autoplay (some restrictive policies require user
// interaction first; in our flow the user has already clicked "Run pipeline"
// so autoplay reliably works from there).
function useAudioAutoplay(
  audio: HTMLAudioElement | null,
  ready: boolean,
): void {
  useEffect(() => {
    if (!audio || !ready) return;
    // play() implicitly triggers a load when readyState < HAVE_CURRENT_DATA,
    // so we don't need a separate loadeddata gate. The shared audio element
    // uses preload="none" to avoid racing the pipeline's GET /torgo/audio
    // fetch; the resulting load + play just rolls into one network round
    // trip. Catch suppresses NotAllowedError on browsers with strict
    // autoplay policies (we don't surface a play prompt here; the user can
    // still click the PlayButton).
    try {
      audio.currentTime = 0;
    } catch {}
    void audio.play().catch(() => {});
    return () => {
      audio.pause();
    };
  }, [audio, ready]);
}

// Track an HTMLAudioElement's currentTime via rAF. Returns ms since start.
// Returns 0 when audio is paused or null. Polled at frame rate (60 Hz) so
// the playhead in the visualizations is buttery-smooth, not stepped.
function useAudioPlayhead(
  audio: HTMLAudioElement | null,
): { timeMs: number; playing: boolean } {
  const [timeMs, setTimeMs] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!audio) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => setPlaying(false);
    const onSeek = () => setTimeMs(Math.round((audio.currentTime || 0) * 1000));
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("seeked", onSeek);
    audio.addEventListener("loadedmetadata", onSeek);
    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("seeked", onSeek);
      audio.removeEventListener("loadedmetadata", onSeek);
    };
  }, [audio]);

  useEffect(() => {
    if (!audio || !playing) return;
    let raf = 0;
    const tick = () => {
      setTimeMs(Math.round((audio.currentTime || 0) * 1000));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [audio, playing]);

  return { timeMs, playing };
}

// ────────────────────────────────────────────────────────────────── main page

export default function HowItWorksPage() {
  const [clips, setClips] = useState<TorgoClip[]>([]);
  const [speakerFilter, setSpeakerFilter] = useState<string>("F03");
  const [selectedClipId, setSelectedClipId] = useState<number | null>(null);
  const [adapterPath, setAdapterPath] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineState>(initialPipelineState);
  const [slide, setSlide] = useState(0);
  // Sub-step within the active slide. Slides with steps > 1 advance through
  // their internal states before the next slide advances. Reset on slide
  // change so re-entering a slide always starts at step 0.
  const [subStep, setSubStep] = useState(0);
  // Single shared <audio> ref so the playhead persists across slide changes
  // and the visualizations on every slide can sync to the same playback.
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playhead = useAudioPlayhead(audioRef.current);

  // Load TORGO clips list once.
  useEffect(() => {
    let cancelled = false;
    fetchWithLabel("GET /torgo/clips", `${SERVER_HTTP}/torgo/clips?limit=20000`)
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        const list: TorgoClip[] = data.clips ?? [];
        setClips(list);
        const first =
          list.find(
            (c) => c.speaker_id === "F03" && c.prompt_set === "timit",
          ) ?? list.find((c) => c.speaker_id === "F03") ?? list[0];
        if (first) setSelectedClipId(first.id);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        // Surface the bootstrap fetch failure on the page so we don't get
        // a silent empty clip list.
        setPipeline((p) => ({
          ...p,
          errorMessage: `bootstrap: ${e.message}`,
        }));
      });
    fetchWithLabel("GET /healthz", `${SERVER_HTTP}/healthz`)
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        setAdapterPath(data.patient_adapter ?? null);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Stable "go forward" / "go back" helpers that respect sub-steps.
  // Forward: if there's another sub-step in this slide, advance it; else
  // advance to the next slide and reset sub-step to 0.
  // Backward: if sub-step > 0, decrement it; else go to the previous
  // slide and land at its LAST sub-step (so retracing reads naturally).
  const goNext = useCallback(() => {
    setSubStep((sub) => {
      const cur = SLIDES[slide];
      if (sub < cur.steps - 1) return sub + 1;
      // advance slide
      if (slide < SLIDES.length - 1) {
        setSlide(slide + 1);
        return 0;
      }
      return sub;
    });
  }, [slide]);
  const goPrev = useCallback(() => {
    setSubStep((sub) => {
      if (sub > 0) return sub - 1;
      if (slide > 0) {
        const target = slide - 1;
        setSlide(target);
        return SLIDES[target].steps - 1;
      }
      return sub;
    });
  }, [slide]);

  // Keyboard navigation: ←/→ to move (with sub-step awareness), 1-9 to jump.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowRight" || e.key === " ") {
        goNext();
        e.preventDefault();
      } else if (e.key === "ArrowLeft") {
        goPrev();
        e.preventDefault();
      } else if (e.key >= "1" && e.key <= "9") {
        const idx = Number(e.key) - 1;
        if (idx < SLIDES.length) {
          setSlide(idx);
          setSubStep(0);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goNext, goPrev]);

  const speakers = useMemo(() => {
    const set = new Set<string>();
    for (const c of clips) set.add(c.speaker_id);
    return Array.from(set).sort();
  }, [clips]);

  const visibleClips = useMemo(() => {
    return clips
      .filter((c) => c.speaker_id === speakerFilter)
      .filter((c) => c.prompt_set !== "non-word")
      .sort((a, b) => a.id - b.id)
      .slice(0, 60);
  }, [clips, speakerFilter]);

  const selectedClip = useMemo(
    () => clips.find((c) => c.id === selectedClipId) ?? null,
    [clips, selectedClipId],
  );

  const runPipeline = useCallback(async () => {
    if (!selectedClip) return;
    setPipeline({
      ...initialPipelineState,
      status: "running",
      startedAt: performance.now(),
    });

    const vanillaPromise = fetchWithLabel(
      "POST /demo/whisper",
      `${SERVER_HTTP}/demo/whisper/${selectedClip.id}`,
      { method: "POST" },
    )
      .then(async (r) => {
        if (!r.ok) {
          throw new Error(`POST /demo/whisper → ${r.status}: ${await r.text()}`);
        }
        const data = await r.json();
        return data.transcript as string;
      })
      .catch((e: Error) => {
        setPipeline((p) => ({ ...p, vanillaError: e.message }));
        return null;
      });

    let audioBytes: ArrayBuffer;
    try {
      const audioRes = await fetchWithLabel(
        "GET /torgo/audio",
        `${SERVER_HTTP}/torgo/audio/${selectedClip.id}`,
        // ``cache: "no-store"`` bypasses any cached response for this
        // URL. The Problem slide's per-clip <audio> elements (or the
        // shared header audio) can populate the browser cache via
        // media-element loads, which the browser stores WITHOUT CORS
        // response headers (media loads don't enforce CORS). A later
        // fetch() to the same URL then retrieves that cached, CORS-
        // less response and the browser blocks it — visible to the
        // user as "Failed to fetch" or "blocked by CORS policy"
        // unless devtools "Disable cache" is on. Forcing a fresh
        // request guarantees a response with the right CORS headers.
        { cache: "no-store" },
      );
      if (!audioRes.ok) {
        throw new Error(
          `GET /torgo/audio → ${audioRes.status}: ${await audioRes.text()}`,
        );
      }
      audioBytes = await audioRes.arrayBuffer();
    } catch (e) {
      setPipeline((p) => ({
        ...p,
        status: "error",
        errorMessage: `${(e as Error).name}: ${(e as Error).message}`,
      }));
      return;
    }

    try {
      const decoded = await getAudioContext().decodeAudioData(
        audioBytes.slice(0),
      );
      setPipeline((p) => ({
        ...p,
        waveform: decoded.getChannelData(0).slice(),
        audioDurationMs: Math.round(decoded.duration * 1000),
      }));
    } catch {}

    const pcm = new Int16Array(audioBytes, 44);
    // The WS path needs an explicit ``adapter_speaker`` so the server
    // doesn't fall back to its default adapter (whatever is symlinked
    // at ``data/adapters/patient``). Without this, e.g. M01 audio gets
    // decoded with F03's adapter, producing empty/nonsense output even
    // though the model is loaded — the adapter is mismatched. Each
    // demo speaker has its own per-patient adapter under
    // ``data/adapters/<speaker_id>*`` (the server's _resolve_adapter_dir
    // does a prefix-match so a directory like ``M01_small_hicap``
    // resolves for ``adapter_speaker=M01``).
    const wsUrl =
      SERVER_HTTP.replace(/^http/, "ws") +
      `/ws/audio?force_patient=1&adapter_speaker=${encodeURIComponent(selectedClip.speaker_id)}`;
    let didStream = false;
    // Track expected vs received translations so we can use a *longer* idle
    // window while Whisper is still working on a known-pending utterance.
    // Whisper-small + LoRA adapter on CPU can take 3-5 s per utterance
    // (mostly the encoder pass on the 30 s mel buffer), and a 2 s idle
    // timer would prematurely disconnect before the translation arrives.
    let utteranceCount = 0;
    let translationCount = 0;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    // PENDING_IDLE_MS is the window while we're waiting on a translation
    // we know is coming (we saw the utterance event). M01's per-patient
    // adapter is whisper-medium.en (the small.en base wasn't capacity
    // enough for severe dysarthria), and medium.en on CPU takes 20-30 s
    // per utterance — so the pending window needs comfortable headroom
    // above that. QUIET_IDLE_MS is the post-stop wait before the first
    // utterance event; VAD + segmentation + first inference can also
    // run long on cold start, so 20 s avoids the "Phonemer card
    // permanently empty" failure mode while still recovering from a
    // genuinely hung pipeline in reasonable time.
    const PENDING_IDLE_MS = 45000;
    const QUIET_IDLE_MS = 20000;
    const armIdleTimer = () => {
      if (idleTimer) clearTimeout(idleTimer);
      const pendingTrans = utteranceCount > translationCount;
      const ms = pendingTrans ? PENDING_IDLE_MS : QUIET_IDLE_MS;
      idleTimer = setTimeout(() => {
        setPipeline((p) =>
          p.status === "running"
            ? { ...p, status: "done", finishedAt: performance.now() }
            : p,
        );
        ws.disconnect();
      }, ms);
    };

    const ws = createPhonemerWs(wsUrl, {
      onMessage: (msg: WsMessage) => {
        if (msg.type === "utterance") {
          const u = msg as UtteranceMessage;
          utteranceCount += 1;
          setPipeline((p) => ({
            ...p,
            utterances: [...p.utterances, u].sort(
              (a, b) => a.start_ms - b.start_ms,
            ),
          }));
          armIdleTimer();
        } else if (msg.type === "phoneme_window") {
          const wm = msg as PhonemeWindowMessage;
          setPipeline((p) => ({
            ...p,
            phonemeFramesByUtt: {
              ...p.phonemeFramesByUtt,
              [wm.utterance_id]: wm.frames,
            },
          }));
          armIdleTimer();
        } else if (msg.type === "translation") {
          const t = msg as TranslationMessage;
          translationCount += 1;
          setPipeline((p) => ({ ...p, translations: [...p.translations, t] }));
          armIdleTimer();
        } else if (msg.type === "translation_error") {
          // Count the error as resolving the pending translation so we don't
          // wait the full PENDING window for nothing.
          translationCount += 1;
          setPipeline((p) => ({
            ...p,
            status: "error",
            errorMessage: (msg as { message?: string }).message ?? "error",
          }));
          armIdleTimer();
        }
      },
      onState: (state) => {
        if (state === "open" && !didStream) {
          didStream = true;
          const send = async () => {
            const CHUNK = 1600;
            for (let i = 0; i < pcm.length; i += CHUNK) {
              const slice = pcm.subarray(i, Math.min(i + CHUNK, pcm.length));
              ws.sendPcm(slice);
              await new Promise((r) => setTimeout(r, 30));
            }
            ws.sendControl({ type: "stop" });
            armIdleTimer();
          };
          void send();
        } else if (state === "closed") {
          if (idleTimer) clearTimeout(idleTimer);
          setPipeline((p) =>
            p.status === "running"
              ? { ...p, status: "done", finishedAt: performance.now() }
              : p,
          );
          // Server-initiated close: ``createPhonemerWs`` would otherwise
          // schedule an infinite reconnect cycle because manualClose is
          // false. Calling disconnect() here flips manualClose true so
          // the reconnect timer is canceled and we land in a stable
          // terminal state.
          ws.disconnect();
        }
      },
    });
    ws.connect();

    const vanillaText = await vanillaPromise;
    setPipeline((p) => ({ ...p, vanillaText }));
  }, [selectedClip]);

  const slideProps: SlideProps = {
    selectedClip,
    pipeline,
    adapterPath,
    playheadMs: playhead.timeMs,
    isPlaying: playhead.playing,
    audio: audioRef.current,
    subStep,
  };
  const isFirst = slide === 0 && subStep === 0;
  const isLast =
    slide === SLIDES.length - 1 && subStep === SLIDES[slide].steps - 1;

  // Source for the shared audio element — switches with the selected clip.
  const audioSrc = selectedClip
    ? `${SERVER_HTTP}/torgo/audio/${selectedClip.id}`
    : undefined;

  return (
    <main className="h-screen w-screen overflow-hidden bg-neutral-950 text-neutral-100 flex flex-col">
      {/* Hidden persistent audio element — playback controls live inside
          slides via PlayButton. Keeping it here means seek state survives
          slide changes. */}
      <audio
        ref={audioRef}
        src={audioSrc}
        // preload="none" so the audio element doesn't race the explicit
        // ``GET /torgo/audio`` that ``runPipeline`` issues for waveform
        // decoding. Two concurrent fetches to the same localhost URL
        // intermittently fail one of them with "Failed to fetch" in dev
        // (worse without devtools open, since cache/network throttling
        // is off). The audio element only needs bytes when the user
        // hits Play; by then the pipeline has finished and the request
        // can complete uncontested.
        preload="none"
        className="hidden"
      />
      {/* Top bar — minimal in presentation mode. Clip picker is now scoped
          to the demo phase (slides 6-7) where it actually matters. */}
      <header className="flex items-center justify-between gap-4 px-6 py-3 border-b border-neutral-800 bg-neutral-950 text-sm">
        <div className="flex items-center gap-3">
          <div className="font-semibold tracking-tight text-neutral-100">
            Phonemer
          </div>
          <span
            className="rounded px-2 py-0.5 text-[10px] uppercase tracking-wider font-mono"
            style={{
              color: PHASE_ACCENT_HEX[SLIDES[slide].phase],
              backgroundColor: PHASE_ACCENT_HEX[SLIDES[slide].phase] + "1a",
              border: `1px solid ${PHASE_ACCENT_HEX[SLIDES[slide].phase]}55`,
            }}
          >
            {PHASE_META[SLIDES[slide].phase].label}
          </span>
          <div className="text-sm text-neutral-300">
            {SLIDES[slide].title}
          </div>
        </div>
        <div className="text-[11px] text-neutral-500 font-mono">
          adapter:{" "}
          {adapterPath ? (
            <span className="text-emerald-400">{adapterPath}</span>
          ) : (
            <span className="text-amber-400">none</span>
          )}
        </div>
      </header>

      {/* Slide content area */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {slide === 0 && <TitleSlide {...slideProps} />}
        {slide === 1 && <ProblemSlide {...slideProps} clips={clips} />}
        {slide === 2 && <TechStackSlide {...slideProps} />}
        {slide === 3 && <ArchitectureSlide {...slideProps} />}
        {slide === 4 && <StrengthsSlide {...slideProps} />}
        {slide === 5 && (
          <DemoSlide
            {...slideProps}
            clips={clips}
            setSelectedClipId={setSelectedClipId}
            setSpeakerFilter={setSpeakerFilter}
            runPipeline={runPipeline}
          />
        )}

        {/* Per-slide error overlay */}
        {pipeline.errorMessage && (
          <div className="absolute top-4 right-4 max-w-md rounded bg-red-950/80 border border-red-700 px-3 py-2 text-xs text-red-200">
            error: {pipeline.errorMessage}
          </div>
        )}
      </div>

      {/* Navigation footer */}
      <footer className="flex items-center justify-between gap-3 px-6 py-3 border-t border-neutral-800 bg-neutral-950">
        <button
          onClick={goPrev}
          disabled={isFirst}
          className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs disabled:opacity-30 hover:bg-neutral-700"
          aria-label="Previous slide"
        >
          ← prev
        </button>
        <div className="flex items-center gap-2">
          {SLIDES.map((s, i) => {
            const accent = PHASE_ACCENT_HEX[s.phase];
            const active = i === slide;
            // Active slide: show pill with sub-step ticks if it has > 1 step.
            return (
              <button
                key={s.id}
                onClick={() => {
                  setSlide(i);
                  setSubStep(0);
                }}
                title={`${PHASE_META[s.phase].label} — ${s.title}`}
                className="h-1.5 rounded-full transition-all relative"
                style={{
                  width: active ? `${0.6 + s.steps * 0.5}rem` : "0.375rem",
                  backgroundColor: active
                    ? "#262626"
                    : i < slide
                      ? accent + "80"
                      : "#404040",
                  border: active ? `1px solid ${accent}80` : "none",
                }}
              >
                {active &&
                  Array.from({ length: s.steps }).map((_, k) => (
                    <span
                      key={k}
                      className="absolute top-0 bottom-0 rounded-full transition-all"
                      style={{
                        left: `${(k / s.steps) * 100}%`,
                        width: `${100 / s.steps}%`,
                        backgroundColor: k <= subStep ? accent : "transparent",
                      }}
                    />
                  ))}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-neutral-500 font-mono">
            {String(slide + 1).padStart(2, "0")} / {String(SLIDES.length).padStart(2, "0")}
            {"  ·  "}
            <span className="hidden sm:inline">←/→ to navigate</span>
          </span>
          <button
            onClick={goNext}
            disabled={isLast}
            className="rounded-md border border-emerald-700/60 bg-emerald-700/30 px-3 py-1.5 text-xs hover:bg-emerald-700/50 disabled:opacity-30"
            aria-label="Next slide"
          >
            next →
          </button>
        </div>
      </footer>
    </main>
  );
}

// ────────────────────────────────────────────────────────────── slide layout

type SlideProps = {
  selectedClip: TorgoClip | null;
  pipeline: PipelineState;
  adapterPath: string | null;
  playheadMs: number;
  isPlaying: boolean;
  audio: HTMLAudioElement | null;
  // Sub-step within the active slide (0-indexed). Used by slides that have
  // internal pages, e.g. the merged Problem slide.
  subStep: number;
  // Optional — only the demo slides use these.
  clips?: TorgoClip[];
  setSelectedClipId?: (id: number) => void;
  setSpeakerFilter?: (id: string) => void;
  runPipeline?: () => Promise<void>;
};

function SlideShell({
  number,
  title,
  description,
  codeRef,
  children,
}: {
  number: number | null;
  title: string;
  description: string[];
  codeRef?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="h-full flex flex-col px-10 py-8 max-w-7xl mx-auto w-full">
      <div className="flex items-baseline gap-4 mb-2">
        {number !== null && (
          <span className="text-emerald-500 font-mono text-sm">
            {String(number).padStart(2, "0")}
          </span>
        )}
        <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
        {codeRef && (
          <code className="ml-auto text-[11px] text-neutral-500 font-mono">
            {codeRef}
          </code>
        )}
      </div>
      <div className="space-y-1.5 mb-6 max-w-4xl">
        {description.map((line, i) => (
          <p key={i} className="text-sm text-neutral-400 leading-relaxed">
            {line}
          </p>
        ))}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">{children}</div>
    </section>
  );
}

// ───────────────────────────────────────────────────────────────────── slides

// ─────────────────────────────────────────────────────── Phase 1 — Problem

function TitleSlide({ pipeline, adapterPath }: SlideProps) {
  return (
    <div className="h-full flex flex-col justify-center px-16 max-w-7xl mx-auto w-full">
      <div className="space-y-5">
        <h1 className="text-[9rem] font-semibold leading-[0.95] tracking-tight">
          Phonemenal
        </h1>
        <div className="text-xl text-neutral-400 font-mono tracking-tight">
          by JP Park · Codey Borrelli · Ken Cula
        </div>
        <p className="text-2xl text-neutral-300 max-w-3xl leading-relaxed pt-8">
          Per-patient calibration for traditional speech models —{" "}
          <span className="text-emerald-300">better transcripts</span>{" "}
          for brain-injury speech, with honest uncertainty when the model
          isn&apos;t sure.
        </p>
        <div className="pt-10 flex items-center gap-6 text-xs font-mono text-neutral-500">
          <span>← / → to navigate</span>
          <span>SPACE for next</span>
          <span>1–8 to jump</span>
          {adapterPath && (
            <span className="ml-auto text-emerald-400">
              ● adapter loaded · ready
            </span>
          )}
          {pipeline.status === "running" && (
            <span className="ml-auto text-amber-400">● pipeline running</span>
          )}
        </div>
      </div>
    </div>
  );
}

// Curated TIMIT clips per speaker for the problem slides. Picked to give
// the audience a clean visual gradient on the sub-step-1 reveal:
//
//   FC02 (healthy) : vanilla WER ~0% → ~all green, "Whisper works fine"
//   F03  (mild)    : vanilla WER ~50-80% → mixed green/red, errors visible
//   M01  (severe)  : vanilla WER ~100-130% → nearly all red, hallucinated
//
// Selected from the TORGO eval split (held out from calibration) by
// running scripts/find_demo_clips on a length-filtered candidate set
// (4-10 words so cards aren't dominated by one long sentence).
//
// Re-rank by running:  cd server && .venv/bin/python -m scripts.find_demo_clips
const FEATURED_CLIP_HINTS: Record<string, string[]> = {
  FC02: [
    "shells on the beach",
    "try to do my best",
    "while we were away",
  ],
  F03: [
    "fish swam across",
    "train approached the depot",
    "celebrate your brother",
  ],
  M01: [
    "all else fails use force",
    "misguided souls",
    "islands are sparsely",
  ],
};

function findFeaturedClips(
  clips: TorgoClip[],
  speaker: string,
): TorgoClip[] {
  const hints = FEATURED_CLIP_HINTS[speaker] ?? [];
  const out: TorgoClip[] = [];
  for (const h of hints) {
    const m = clips.find(
      (c) =>
        c.speaker_id === speaker &&
        c.prompt_set === "timit" &&
        (c.intended_text ?? "").toLowerCase().includes(h),
    );
    if (m && !out.some((c) => c.id === m.id)) out.push(m);
  }
  // Fallback: if no hinted clips were found (fresh ingest), grab the
  // first few TIMIT clips for this speaker so the card isn't empty.
  if (out.length === 0) {
    return clips
      .filter((c) => c.speaker_id === speaker && c.prompt_set === "timit")
      .slice(0, 3);
  }
  return out;
}

function ProblemSlide({ clips, subStep }: SlideProps) {
  // Merged slide: sub-step 0 shows what each patient said across several
  // sentences (just the reference text). Sub-step 1 reveals what
  // off-the-shelf Whisper transcribed for each, with wrong words
  // highlighted in the severity accent — the audience sees "the more
  // severe the patient, the more red on the card" at a glance.
  const featured: { speaker: string; severity: string; clips: TorgoClip[]; werExpected: number }[] = useMemo(
    () => [
      { speaker: "FC02", severity: "Healthy control", clips: findFeaturedClips(clips ?? [], "FC02"), werExpected: 0.27 },
      { speaker: "F03", severity: "Mild dysarthria", clips: findFeaturedClips(clips ?? [], "F03"), werExpected: 0.7 },
      { speaker: "M01", severity: "Severe dysarthria", clips: findFeaturedClips(clips ?? [], "M01"), werExpected: 1.02 },
    ],
    [clips],
  );

  // Pre-fetch vanilla Whisper transcripts for every featured clip as soon
  // as the slide mounts. Keyed by clip id (not speaker) since each card
  // now shows multiple clips. By the time the user advances to sub-step
  // 1, all transcripts are already populated and the morph feels instant.
  const [transcripts, setTranscripts] = useState<Record<number, string | null>>(
    {},
  );
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      for (const f of featured) {
        for (const c of f.clips) {
          if (transcripts[c.id] !== undefined) continue;
          try {
            const r = await fetchWithLabel(
              `POST /demo/whisper (${c.id})`,
              `${SERVER_HTTP}/demo/whisper/${c.id}`,
              { method: "POST" },
            );
            const data = await r.json();
            if (cancelled) return;
            setTranscripts((t) => ({ ...t, [c.id]: data.transcript }));
          } catch (e) {
            if (cancelled) return;
            setTranscripts((t) => ({
              ...t,
              [c.id]: `(error: ${(e as Error).message})`,
            }));
          }
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [featured]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-full flex flex-col px-12 py-8 max-w-7xl mx-auto w-full">
      {/* Header morphs between two states — both keyed by subStep so the
          fade-in animation re-fires when you advance. */}
      <header className="mb-6 min-h-[120px]">
        <div
          key={`hdr-${subStep}`}
          className="animate-[fadeUp_400ms_ease-out_both]"
        >
          {subStep === 0 ? (
            <>
              <h1 className="text-4xl font-semibold tracking-tight mb-2">
                Three patients, each reading a few TIMIT sentences.
              </h1>
              <p className="text-base text-neutral-400 max-w-3xl">
                A healthy control, a mild dysarthric speaker, and a severely
                impaired speaker — all reading from the standard TIMIT
                corpus.
              </p>
            </>
          ) : (
            <>
              <h1 className="text-4xl font-semibold tracking-tight mb-2">
                Off-the-shelf Whisper:{" "}
                <span className="text-amber-300">
                  it gets worse the worse it gets
                </span>
                .
              </h1>
              <p className="text-base text-neutral-400 max-w-3xl">
                Same recordings, fed to baseline{" "}
                <code className="text-amber-200 text-sm">whisper-base.en</code>
                . Highlighted words are wrong; the more severe the patient,
                the more red on the card.
              </p>
            </>
          )}
        </div>
      </header>

      <div className="grid md:grid-cols-3 gap-5 flex-1 min-h-0">
        {featured.map((p, i) => (
          <UnifiedPatientCard
            key={p.speaker}
            {...p}
            index={i}
            subStep={subStep}
            transcripts={transcripts}
          />
        ))}
      </div>

      {/* Bottom section: morphs from the qualitative footnote to the WER
          bar chart. */}
      <div className="mt-6 min-h-[110px]">
        <div
          key={`bot-${subStep}`}
          className="animate-[fadeUp_400ms_ease-out_both]"
        >
          {subStep === 0 ? (
            <div className="text-sm text-neutral-400 max-w-3xl">
              Healthy control: easy. Mild: slow but intelligible. Severe:
              genuinely hard to follow without context.{" "}
              <span className="text-amber-300">
                Press → to see what a state-of-the-art ASR makes of these
                same recordings.
              </span>
            </div>
          ) : (
            <WerBarChart
              rows={featured.map((f) => ({
                // Aggregate WER over the TORGO eval split — not the single
                // demo clip's measured WER. Per-clip values bounce around
                // (clip-level TIMIT noise) and have been observed to break
                // the monotonic severity ordering (e.g. FC02's specific
                // clip can score worse than F03's). The bar chart's
                // subtitle says "eval split" — keep it honest.
                label: f.speaker,
                severity: f.severity.replace(" dysarthria", ""),
                wer: f.werExpected,
              }))}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// One card whose inner content swaps with subStep. The shell (border,
// speaker badge) stays put — that anchoring is what makes the transition
// feel like the same patient revealing more, not a new slide. Renders a
// LIST of intended/Whisper pairs so the audience can scan across multiple
// sentences and form a "this whole patient is mistranscribed" impression
// at a glance, instead of one cherry-picked clip per patient.
function UnifiedPatientCard({
  speaker,
  severity,
  clips,
  index,
  subStep,
  transcripts,
  werExpected,
}: {
  speaker: string;
  severity: string;
  clips: TorgoClip[];
  index: number;
  subStep: number;
  transcripts: Record<number, string | null>;
  werExpected: number;
}) {
  const palette = ["emerald", "amber", "red"][index] ?? "amber";
  const sub0Accent =
    palette === "emerald" ? "#10b981" : palette === "amber" ? "#f59e0b" : "#ef4444";
  // Color-code by *aggregate* severity (werExpected), not any individual
  // clip's measured WER. Per-clip values can break monotonic ordering with
  // severity (TIMIT clip noise), which would paint FC02 redder than F03
  // and undercut the "it gets worse the worse it gets" narrative.
  const sub1 = werSeverityStyle(werExpected);
  const finalAccent = subStep === 0 ? sub0Accent : sub1.accent;
  // Scale the card background intensity with aggregate WER so the audience
  // doesn't have to count red words to feel "M01 is worse than F03". Both
  // cards have lots of highlights — the *proportion* is what differs, and
  // a heavier bg + a small WER bar below makes that proportion visible
  // without re-introducing per-clip WER numbers (which were non-monotonic).
  const bgAlphaSub1 =
    werExpected >= 1.0 ? 0.16 : werExpected >= 0.5 ? 0.09 : 0.04;
  const borderAlpha =
    werExpected >= 1.0 ? "ee" : werExpected >= 0.5 ? "bb" : "88";
  return (
    <div
      className="relative rounded-2xl border bg-neutral-900/40 p-5 flex flex-col gap-4 transition-all duration-500"
      style={{
        borderColor: finalAccent + (subStep === 1 ? borderAlpha : "55"),
        backgroundColor:
          subStep === 1
            ? `rgba(${sub1.bgRgb}, ${bgAlphaSub1})`
            : "rgba(23,23,23,0.4)",
        boxShadow:
          subStep === 1 && werExpected >= 1.0
            ? `0 0 24px rgba(${sub1.bgRgb}, 0.18)`
            : undefined,
      }}
    >
      <div className="flex items-baseline justify-between">
        <div className="flex-1">
          <div className="text-xs font-mono uppercase tracking-wider text-neutral-500">
            {speaker}
          </div>
          <div
            className="text-xl font-semibold transition-colors duration-500"
            style={{ color: finalAccent }}
          >
            {severity}
          </div>
          {/* Per-card aggregate WER bar — only on sub-step 1, after the
              reveal. Uses werExpected (TIMIT eval split mean), matching
              the bar chart below. Width caps at 100% visually but the
              text shows the true value so M01's >100% reads as
              "fabricating extra words" without breaking the bar layout. */}
          {subStep === 1 && (
            <div
              key={`wer-bar-${speaker}`}
              className="mt-2 flex items-center gap-2 animate-[fadeUp_400ms_ease-out_both]"
            >
              <div className="relative w-28 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.min(100, werExpected * 100)}%`,
                    backgroundColor: finalAccent,
                  }}
                />
              </div>
              <span
                className="text-[10px] font-mono"
                style={{ color: finalAccent }}
              >
                ~{(werExpected * 100).toFixed(0)}% wrong
              </span>
            </div>
          )}
        </div>
        <div className="text-[10px] uppercase tracking-wider font-mono text-neutral-600">
          TORGO
        </div>
      </div>

      {/* Inner content — keyed on subStep so it remounts and re-fires the
          fade-in animation. */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div
          key={`card-${speaker}-${subStep}`}
          className="animate-[fadeUp_400ms_ease-out_both] space-y-4"
        >
          {clips.length === 0 ? (
            <div className="text-sm text-neutral-500">no clips available</div>
          ) : subStep === 0 ? (
            clips.map((c) => (
              <div key={c.id} className="flex items-start gap-3">
                <ClipPlayButton clipId={c.id} accent={finalAccent} />
                <div className="flex-1 text-base text-neutral-100 italic leading-relaxed">
                  “{c.intended_text}”
                </div>
              </div>
            ))
          ) : (
            clips.map((c, idx) => {
              const t = transcripts[c.id];
              return (
                <div
                  key={c.id}
                  className={
                    "space-y-1.5 " +
                    (idx < clips.length - 1
                      ? "pb-3 border-b border-neutral-800/60"
                      : "")
                  }
                >
                  <div className="text-xs italic text-neutral-500 leading-snug">
                    “{c.intended_text}”
                  </div>
                  <div className="text-sm leading-snug">
                    {t === undefined || t === null ? (
                      <span className="text-neutral-500 italic">
                        transcribing…
                      </span>
                    ) : (
                      <SeverityDiffText
                        hypothesis={t}
                        reference={c.intended_text}
                        accent={finalAccent}
                      />
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

// Compact circular play/pause button next to a single TIMIT sentence on
// the problem slide's sub-step 0. Each row owns its own <audio> so users
// can spot-check any sentence (especially the severe-dysarthria ones) and
// hear *why* off-the-shelf Whisper struggles before the transcripts are
// even revealed.
function ClipPlayButton({
  clipId,
  accent,
}: {
  clipId: number;
  accent: string;
}) {
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const ph = useAudioPlayhead(audioEl);
  return (
    <>
      <audio
        ref={setAudioEl}
        src={`${SERVER_HTTP}/torgo/audio/${clipId}`}
        preload="metadata"
        className="hidden"
      />
      <button
        onClick={() => {
          if (!audioEl) return;
          if (audioEl.paused) {
            audioEl.currentTime = 0;
            void audioEl.play().catch(() => {});
          } else audioEl.pause();
        }}
        className="shrink-0 mt-1 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono transition-colors"
        style={{
          backgroundColor: accent + (ph.playing ? "55" : "22"),
          border: `1px solid ${accent}`,
          color: accent,
        }}
        aria-label={ph.playing ? "Pause audio" : "Play audio"}
      >
        {ph.playing ? "❚❚" : "▶"}
      </button>
    </>
  );
}


function WerBarChart({
  rows,
}: {
  rows: { label: string; severity: string; wer: number }[];
}) {
  const max = Math.max(1.2, ...rows.map((r) => r.wer));
  // Bars animate in via staggered widths after mount.
  const [animated, setAnimated] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setAnimated(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 px-5 py-4">
      <div className="text-[11px] uppercase tracking-wider text-neutral-500 mb-3">
        WER by severity (vanilla whisper-base.en, TORGO eval split)
      </div>
      <div className="space-y-2.5">
        {rows.map((r, i) => {
          const sev = werSeverityStyle(r.wer);
          return (
            <div key={r.label} className="flex items-center gap-3">
              <div className="w-32 text-xs font-mono text-neutral-400">
                {r.label} · {r.severity}
              </div>
              <div className="flex-1 relative h-7 bg-neutral-800/60 rounded overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-1000 ease-out"
                  style={{
                    width: animated ? `${(r.wer / max) * 100}%` : "0%",
                    background: `linear-gradient(to right, ${sev.accent}66, ${sev.accent})`,
                    transitionDelay: `${i * 120}ms`,
                  }}
                />
                {/* 100% reference line */}
                <div
                  className="absolute top-0 h-full w-px bg-white/40"
                  style={{ left: `${(1 / max) * 100}%` }}
                  title="WER = 1.0"
                />
              </div>
              <div
                className="w-14 font-mono text-sm text-right"
                style={{ color: sev.accent }}
              >
                {(r.wer * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 text-[10px] font-mono text-neutral-500 text-right">
        ▎ white tick = 100% WER reference line
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────── Phase 2 — Architecture

// Tech-stack slide for hackathon judges. Two equal sections — platform
// stack on top, speech/ML model stack on bottom. Icon-driven tiles
// (logo-tile + name + one-line spec), no prose descriptions. Goal:
// ~10 s scan, judges recognize the stack from the icons alone.
function TechStackSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-12 py-8 max-w-7xl mx-auto w-full">
      <header className="mb-6">
        <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          What it&apos;s built with
        </div>
        <h1 className="text-4xl font-semibold tracking-tight">
          Open-source stack,{" "}
          <span className="text-cyan-300">all running locally</span>.
        </h1>
      </header>

      {/* Platform tech stack */}
      <section className="mb-8">
        <SectionLabel
          accent="#06b6d4"
          label="Platform"
          sub="streaming web app"
        />
        <div className="grid grid-cols-6 gap-3 mt-3">
          <TechTile
            icon={<NextIcon />}
            name="Next.js 15"
            description="renders the listener UI"
            meta="React 19 + Tailwind"
            accent="#e5e5e5"
          />
          <TechTile
            icon={<FastapiIcon />}
            name="FastAPI"
            description="serves the API"
            meta="REST + WebSocket"
            accent="#10b981"
          />
          <TechTile
            icon={<PythonIcon />}
            name="Python 3.14"
            description="runs the backend"
            meta="CPython · typed"
            accent="#fbbf24"
          />
          <TechTile
            icon={<PytorchIcon />}
            name="PyTorch"
            description="runs ML inference"
            meta="inference_mode · CPU"
            accent="#f97316"
          />
          <TechTile
            icon={<WebsocketIcon />}
            name="WebSocket"
            description="streams audio frames"
            meta="16 kHz · binary"
            accent="#22d3ee"
          />
          <TechTile
            icon={<SqliteIcon />}
            name="SQLite"
            description="stores patient data"
            meta="single-file, local"
            accent="#a78bfa"
          />
        </div>
      </section>

      {/* Speech & ML model stack */}
      <section className="flex-1 min-h-0 flex flex-col">
        <SectionLabel
          accent="#a78bfa"
          label="Speech & ML models"
          sub="six off-the-shelf weights"
        />
        <div className="grid grid-cols-6 gap-3 mt-3">
          <TechTile
            icon={<WhisperIcon />}
            name="Whisper"
            description="transcribes speech to text"
            meta="small.en · 242 M"
            accent="#a78bfa"
          />
          <TechTile
            icon={<LoraIcon />}
            name="PEFT · LoRA"
            description="adapts Whisper per patient"
            meta="r=16 · 6.7 M trainable"
            accent="#a78bfa"
          />
          <TechTile
            icon={<Wav2vecIcon />}
            name="wav2vec2"
            description="extracts phoneme probabilities"
            meta="50 Hz · 393 classes"
            accent="#22d3ee"
          />
          <TechTile
            icon={<SpeakerIcon />}
            name="ECAPA-TDNN"
            description="routes patient vs listener"
            meta="cosine sim ≥ 0.55"
            accent="#22d3ee"
          />
          <TechTile
            icon={<VadIcon />}
            name="Silero VAD"
            description="detects speech vs silence"
            meta="threshold 0.5"
            accent="#22d3ee"
          />
          <TechTile
            icon={<CorpusIcon />}
            name="TORGO + TIMIT"
            description="dysarthric speech corpus"
            meta="cal/eval data"
            accent="#22d3ee"
          />
        </div>
      </section>
    </div>
  );
}

// Vertical icon tile: big colored icon on top, name, optional one-line
// "what it does" description, then mono meta spec. Description is only
// used by the ML section — the platform tiles stay icon+name+meta
// because the names (Next.js, FastAPI, etc.) already say what they
// do. ML names (Whisper, ECAPA-TDNN, wav2vec2) need the verb.
function TechTile({
  icon,
  name,
  meta,
  description,
  accent,
}: {
  icon: React.ReactNode;
  name: string;
  meta: string;
  description?: string;
  accent: string;
}) {
  return (
    <div
      className="rounded-xl border bg-neutral-900/40 px-3 py-4 flex flex-col items-center gap-2.5"
      style={{
        borderColor: accent + "55",
        boxShadow: `0 0 16px ${accent}11`,
      }}
    >
      <div
        className="w-12 h-12 rounded-lg flex items-center justify-center"
        style={{
          backgroundColor: accent + "1a",
          border: `1px solid ${accent}55`,
          color: accent,
        }}
      >
        {icon}
      </div>
      <div className="text-center">
        <div className="text-sm font-semibold text-neutral-100 leading-tight">
          {name}
        </div>
        {description && (
          <div className="text-[11px] text-neutral-300 mt-1 leading-snug px-1">
            {description}
          </div>
        )}
        <div className="text-[10px] font-mono text-neutral-500 mt-1 leading-tight">
          {meta}
        </div>
      </div>
    </div>
  );
}

// Header strip for the two halves of the tech-stack slide. Small accent
// dot + label on the left, faded sub-caption on the right — sits inline
// in one row so each section starts crisp without burning vertical
// space the way a heading + paragraph would.
function SectionLabel({
  accent,
  label,
  sub,
}: {
  accent: string;
  label: string;
  sub: string;
}) {
  return (
    <div className="flex items-baseline justify-between border-b border-neutral-800/80 pb-1.5">
      <div className="flex items-center gap-2">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: accent }}
        />
        <span
          className="text-[11px] uppercase tracking-[0.2em] font-mono"
          style={{ color: accent }}
        >
          {label}
        </span>
      </div>
      <span className="text-[10px] uppercase tracking-wider font-mono text-neutral-600">
        {sub}
      </span>
    </div>
  );
}

// ── Platform-row monogram icons ─────────────────────────────────────
// Brand-stylized abstract glyphs (not real logos — avoids trademark
// concerns and keeps the visual language consistent with the hero
// icons above). Each evokes the brand's actual mark without copying it.

function NextIcon() {
  // Stylized "N" with the Next.js diagonal stroke
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" fill="none" strokeWidth="1.5" opacity="0.4" />
      <path d="M8 7 V17 M8 7 L16 17 M16 7 V17" stroke="currentColor" fill="none" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function FastapiIcon() {
  // Lightning bolt — FastAPI's brand association ("fast")
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <path d="M13 3 L5 13 H11 L9 21 L19 11 H13 L15 3 Z" fill="currentColor" />
    </svg>
  );
}

function PythonIcon() {
  // Two interlocking quarter-circles evoking the Python logo silhouette
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <path d="M8 5 Q12 5, 12 9 V13 H6 Q4 13, 4 15 V17 Q4 19, 6 19 H9" stroke="currentColor" fill="none" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M16 19 Q12 19, 12 15 V11 H18 Q20 11, 20 9 V7 Q20 5, 18 5 H15" stroke="currentColor" fill="none" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="8" cy="8" r="0.9" fill="currentColor" />
      <circle cx="16" cy="16" r="0.9" fill="currentColor" />
    </svg>
  );
}

function PytorchIcon() {
  // Flame — PyTorch's torch mark
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <path d="M12 3 L7 9 Q4 12, 5 16 Q6 20, 12 21 Q18 20, 19 16 Q20 12, 17 9 Z" fill="currentColor" opacity="0.85" />
      <circle cx="14.5" cy="8" r="1.4" fill="#0a0a0a" />
    </svg>
  );
}

function WebsocketIcon() {
  // Two-way arrow showing bidirectional stream
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <path d="M4 9 H18 M15 5.5 L18 9 L15 12.5" stroke="currentColor" fill="none" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M20 15 H6 M9 11.5 L6 15 L9 18.5" stroke="currentColor" fill="none" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SqliteIcon() {
  // Database cylinder
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <ellipse cx="12" cy="6" rx="7" ry="2.5" stroke="currentColor" fill="none" strokeWidth="1.4" />
      <path d="M5 6 V12 Q5 14.5, 12 14.5 Q19 14.5, 19 12 V6" stroke="currentColor" fill="none" strokeWidth="1.4" />
      <path d="M5 12 V18 Q5 20.5, 12 20.5 Q19 20.5, 19 18 V12" stroke="currentColor" fill="none" strokeWidth="1.4" />
    </svg>
  );
}

// ── Inline SVG icons for TechTile cards ─────────────────────────────
// Hand-rolled so the build doesn't take a dependency on lucide / heroicons
// for six icons. Each picks a glyph that visually echoes what the model
// does: stacked transformer layers, multiplied LoRA factors, waveform
// becoming phoneme dots, voiceprint bars, gated speech pulse, stacked
// records. `currentColor` is set by the card's color style so each icon
// inherits its tier accent.

function WhisperIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      <rect x="4" y="5" width="16" height="2" rx="0.6" fill="currentColor" opacity="0.95" />
      <rect x="4" y="9" width="16" height="2" rx="0.6" fill="currentColor" opacity="0.75" />
      <rect x="4" y="13" width="16" height="2" rx="0.6" fill="currentColor" opacity="0.55" />
      <rect x="4" y="17" width="16" height="2" rx="0.6" fill="currentColor" opacity="0.35" />
    </svg>
  );
}

function LoraIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      {/* B (tall, thin) */}
      <rect x="6" y="5" width="3.5" height="14" rx="0.6" fill="currentColor" opacity="0.85" />
      {/* multiply dot */}
      <circle cx="12" cy="12" r="0.8" fill="currentColor" />
      {/* A (wide, short) */}
      <rect x="14.5" y="10.25" width="5" height="3.5" rx="0.6" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

function Wav2vecIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      {/* Waveform on left */}
      <path
        d="M2.5 12 Q3.5 7, 4.5 12 T6.5 12 T8.5 12 T10.5 12"
        stroke="currentColor"
        fill="none"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      {/* Arrow → */}
      <path
        d="M12 12 H16 M14.5 10.4 L16 12 L14.5 13.6"
        stroke="currentColor"
        fill="none"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Phoneme-class dots (3 stacked, abstract symbols) */}
      <circle cx="18.5" cy="9" r="0.9" fill="currentColor" />
      <circle cx="20.7" cy="12" r="0.9" fill="currentColor" />
      <circle cx="18.5" cy="15" r="0.9" fill="currentColor" />
    </svg>
  );
}

function SpeakerIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      {/* Voiceprint — symmetric bars */}
      <rect x="3.5" y="10" width="1.8" height="4" rx="0.5" fill="currentColor" opacity="0.5" />
      <rect x="6.5" y="7" width="1.8" height="10" rx="0.5" fill="currentColor" opacity="0.75" />
      <rect x="9.5" y="4" width="1.8" height="16" rx="0.5" fill="currentColor" />
      <rect x="12.5" y="6" width="1.8" height="12" rx="0.5" fill="currentColor" opacity="0.9" />
      <rect x="15.5" y="9" width="1.8" height="6" rx="0.5" fill="currentColor" opacity="0.65" />
      <rect x="18.5" y="11" width="1.8" height="2" rx="0.5" fill="currentColor" opacity="0.4" />
    </svg>
  );
}

function VadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      {/* Gate brackets [ ] */}
      <path d="M6 5 H3.5 V19 H6" stroke="currentColor" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M18 5 H20.5 V19 H18" stroke="currentColor" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* Speech pulse inside */}
      <polyline
        points="6.5,12 8,12 9.5,7.5 11,16.5 12.5,7.5 14,16.5 15.5,12 17.5,12"
        stroke="currentColor"
        fill="none"
        strokeWidth="1.3"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function CorpusIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-7 h-7" aria-hidden>
      {/* Stacked record cards */}
      <rect x="3.5" y="5" width="17" height="3.5" rx="0.8" stroke="currentColor" fill="currentColor" fillOpacity="0.18" strokeWidth="1" />
      <rect x="3.5" y="10.25" width="17" height="3.5" rx="0.8" stroke="currentColor" fill="currentColor" fillOpacity="0.18" strokeWidth="1" />
      <rect x="3.5" y="15.5" width="17" height="3.5" rx="0.8" stroke="currentColor" fill="currentColor" fillOpacity="0.18" strokeWidth="1" />
      {/* Mini "row" indicators */}
      <rect x="5" y="6.3" width="3" height="0.9" rx="0.45" fill="currentColor" opacity="0.8" />
      <rect x="5" y="11.55" width="6" height="0.9" rx="0.45" fill="currentColor" opacity="0.8" />
      <rect x="5" y="16.8" width="4" height="0.9" rx="0.45" fill="currentColor" opacity="0.8" />
    </svg>
  );
}

function ArchitectureSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-10 py-6 max-w-[100rem] mx-auto w-full">
      <header className="mb-4">
        <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          The system
        </div>
        <h1 className="text-4xl font-semibold tracking-tight mb-2">
          The Phonemer pipeline
        </h1>
        <p className="text-base text-neutral-400 max-w-4xl">
          Audio enters at the left and forks into{" "}
          <span className="text-violet-300">Whisper + per-patient LoRA</span>{" "}
          (the calibrated path) and{" "}
          <span className="text-cyan-300">wav2vec2 phoneme extraction</span>{" "}
          (the safety check). The two paths converge at the gate, which
          decides whether to surface Whisper&apos;s text confidently or flag
          it as uncertain.
        </p>
      </header>
      <div className="flex-1 min-h-0">
        <PipelineDiagram />
      </div>
    </div>
  );
}

function PipelineDiagram() {
  return (
    <div className="relative h-full w-full rounded-2xl bg-gradient-to-br from-neutral-950 via-neutral-950 to-cyan-950/15 border border-cyan-900/30 overflow-hidden">
      <BackdropGrid />
      <FlowSvg />
      {/* Card centers and widths are *all* in % of the container so they
          share a coordinate space with the SVG flow paths below
          (viewBox 0..100 stretched to fill the container with
          preserveAspectRatio="none"). Edges are: regular cards 6% from
          center, wide cards 9% from center. The path endpoints in
          FlowSvg land *exactly* on these edges. Side-input nodes hang
          above the main row and feed downward into their consumers. */}
      <SideInput
        style={{ top: "38%", left: "36%" }}
        accent="#06b6d4"
        label="Enrolled voices"
        sub="from speakers DB"
      >
        <SideInputLine label="patient" value="centroid (F03)" labelColor="#67e8f9" />
        <SideInputLine label="listener" value="speaker_A" labelColor="#67e8f9" />
      </SideInput>
      <SideInput
        style={{ top: "8%", left: "58%" }}
        accent="#a78bfa"
        label="initial_prompt"
        sub="≤ 220 chars · ContextManager"
      >
        <SideInputLine label="last turn" value={"“Could you read this?”"} labelColor="#c4b5fd" />
        <SideInputLine label="topics" value="banks, money…" labelColor="#c4b5fd" />
        <SideInputLine label="vocab" value="Dr. Park, gabap…" labelColor="#c4b5fd" />
      </SideInput>
      <Stage style={{ top: "55%", left: "7%" }} accent="#94a3b8" label="Audio" sub="raw 16 kHz">
        <MiniWaveformVis />
      </Stage>
      <Stage style={{ top: "55%", left: "22%" }} accent="#06b6d4" label="VAD" sub="silero">
        <VadVis />
      </Stage>
      <Stage style={{ top: "55%", left: "36%" }} accent="#06b6d4" label="Speaker ID" sub="ECAPA-TDNN">
        <SpeakerVis />
      </Stage>
      <Stage
        style={{ top: "30%", left: "58%" }}
        accent="#8b5cf6"
        label="Whisper"
        sub="small.en + LoRA"
        wide
      >
        <WhisperVis />
      </Stage>
      <Stage
        style={{ top: "80%", left: "58%" }}
        accent="#06b6d4"
        label="Phoneme model"
        sub="wav2vec2"
        wide
      >
        <PhonemeVis />
      </Stage>
      <Stage style={{ top: "55%", left: "78%" }} accent="#06b6d4" label="Safety gate" sub="forced alignment">
        <GateVis />
      </Stage>
      <Stage style={{ top: "55%", left: "93%" }} accent="#10b981" label="Output" sub="text + flag">
        <OutputVis />
      </Stage>
      {/* Connector labels — what kind of data flows along each wire. The
          inner-card visualizations show what each stage *does*; these
          labels show what each stage *produces*. Positions are picked to
          land on the gap between cards rather than inside them. */}
      <EdgeLabel top="51%" left="14.5%">pcm · 16 kHz</EdgeLabel>
      <EdgeLabel top="51%" left="29%">speech</EdgeLabel>
      <EdgeLabel top="38%" left="47%">mel features</EdgeLabel>
      <EdgeLabel top="71%" left="44%">audio</EdgeLabel>
      <EdgeLabel top="40%" left="71%">text tokens</EdgeLabel>
      <EdgeLabel top="71%" left="71%">phoneme posterior</EdgeLabel>
      <EdgeLabel top="51%" left="85.5%">text + flag</EdgeLabel>
      <EdgeLabel top="46%" left="36%">cosine compare</EdgeLabel>
      <EdgeLabel top="17%" left="58%">decoder prefix</EdgeLabel>
      {/* Comparator badge at the gate's input — visually marks the spot
          where Whisper's text and wav2vec2's phoneme posterior are
          force-aligned against each other. */}
      <div
        className="absolute pointer-events-none"
        style={{ top: "55%", left: "72%", transform: "translate(-50%, -50%)" }}
      >
        <div className="w-5 h-5 rounded-full bg-neutral-950 border border-cyan-700/70 flex items-center justify-center text-[11px] font-mono text-cyan-300 shadow-[0_0_6px_rgba(6,182,212,0.4)]">
          ≟
        </div>
      </div>
      <div className="absolute bottom-4 left-6 right-6 flex items-center justify-between text-[11px] font-mono text-neutral-500 pointer-events-none">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-500" />
            frozen / pre-trained
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-violet-500" />
            calibrated per patient
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
            output
          </span>
        </div>
      </div>
    </div>
  );
}

function BackdropGrid() {
  return (
    <div
      aria-hidden
      className="absolute inset-0 opacity-15 pointer-events-none"
      style={{
        backgroundImage:
          "linear-gradient(to right, rgba(6,182,212,0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(6,182,212,0.15) 1px, transparent 1px)",
        backgroundSize: "48px 48px",
      }}
    />
  );
}

// SVG layer that draws the connecting paths between stages with animated
// "data" dots flowing along them. The fork into Whisper + wav2vec2 is the
// structural feature that the old card layout hid; here it's the first
// thing the audience sees.
function FlowSvg() {
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="cyanGrad" x1="0" x2="1">
          <stop offset="0" stopColor="#06b6d4" stopOpacity="0.5" />
          <stop offset="1" stopColor="#06b6d4" stopOpacity="0.9" />
        </linearGradient>
        <linearGradient id="violetGrad" x1="0" x2="1">
          <stop offset="0" stopColor="#06b6d4" stopOpacity="0.4" />
          <stop offset="1" stopColor="#8b5cf6" stopOpacity="0.9" />
        </linearGradient>
        <linearGradient id="violetToCyan" x1="0" x2="1">
          <stop offset="0" stopColor="#8b5cf6" stopOpacity="0.9" />
          <stop offset="1" stopColor="#06b6d4" stopOpacity="0.5" />
        </linearGradient>
        <linearGradient id="emeraldGrad" x1="0" x2="1">
          <stop offset="0" stopColor="#06b6d4" stopOpacity="0.5" />
          <stop offset="1" stopColor="#10b981" stopOpacity="0.9" />
        </linearGradient>
      </defs>
      {/* Edge coords (regular card halfWidth = 6%, wide halfWidth = 9%).
          Main row centered at y=55, Whisper at y=30, wav2vec2 at y=80.
          SideInput cards at y=8 (12% wide → halfWidth 6.5%):
              Audio       [1..13]    center  7,  y=55
              VAD         [16..28]   center 22,  y=55
              Speaker     [30..42]   center 36,  y=55  (top edge ≈ y=44)
              Whisper     [49..67]   center 58,  y=30  (left/right at y=30; top edge ≈ y=22)
              wav2vec2    [49..67]   center 58,  y=80  (top edge ≈ y=69)
              Gate        [72..84]   center 78,  y=55
              Output      [87..99]   center 93,  y=55
              Enrolled IN [29.5..42.5] center 36, y=38  (bottom edge ≈ y=46)
              Context IN  [51.5..64.5] center 58, y=8   (bottom edge ≈ y=15)         */}
      {/* Side inputs flowing DOWN into their consumers — short vertical
          comets so both side inputs feel visually balanced. */}
      <FlowLine
        d="M 36 46 V 50"
        gradientId="cyanGrad"
        color="#06b6d4"
        duration={1.6}
      />
      <FlowLine
        d="M 58 15 V 22"
        gradientId="violetGrad"
        color="#a78bfa"
        duration={1.6}
      />
      {/* Audio → VAD */}
      <FlowLine d="M 13 55 H 16" gradientId="cyanGrad" color="#06b6d4" duration={1.6} />
      {/* VAD → Speaker ID */}
      <FlowLine d="M 28 55 H 30" gradientId="cyanGrad" color="#06b6d4" duration={1.6} />
      {/* Speaker ID forks: up to Whisper, down to wav2vec2 */}
      <FlowLine
        d="M 42 55 C 46 55, 46 30, 49 30"
        gradientId="violetGrad"
        color="#8b5cf6"
        duration={2.4}
      />
      <FlowLine
        d="M 42 55 C 46 55, 46 80, 49 80"
        gradientId="cyanGrad"
        color="#06b6d4"
        duration={2.4}
      />
      {/* Whisper + wav2vec2 converge at the gate */}
      <FlowLine
        d="M 67 30 C 70 30, 70 55, 72 55"
        gradientId="violetToCyan"
        color="#8b5cf6"
        duration={2.4}
      />
      <FlowLine
        d="M 67 80 C 70 80, 70 55, 72 55"
        gradientId="cyanGrad"
        color="#06b6d4"
        duration={2.4}
      />
      {/* Gate → Output */}
      <FlowLine d="M 84 55 H 87" gradientId="emeraldGrad" color="#10b981" duration={1.6} />
    </svg>
  );
}

// One animated connector — a comet streak with a fading tail.
//
// Implementation: stack N paths over the same `d`. Each path uses
// pathLength=100 + stroke-dasharray="W (100-W)", which puts a single
// visible dash of width W on the path with the rest as gap. Animating
// stroke-dashoffset 0 → -100 sweeps that dash from start to end.
//
// To get a fading trail behind the head we render multiple paths with
// staggered animation-delays — each successive layer is offset slightly
// behind the head and rendered with smaller width and lower opacity.
// They all loop in lockstep, so the head + tail moves together as one
// comet. Only the head gets the drop-shadow glow.
//
// vector-effect="non-scaling-stroke" keeps the stroke width constant in
// device pixels even though the SVG is stretched non-uniformly by
// preserveAspectRatio="none" — without this the streaks would deform.
function FlowLine({
  d,
  gradientId,
  color,
  duration,
}: {
  d: string;
  gradientId: string;
  color: string;
  duration: number;
}) {
  // delay is a fraction of duration; multiplying by duration → seconds
  // offset behind the head. fraction × pathLength(100) → units behind.
  const tail = [
    { width: 1.8, opacity: 1.0, delay: 0 },
    { width: 1.6, opacity: 0.65, delay: 0.012 },
    { width: 1.4, opacity: 0.42, delay: 0.024 },
    { width: 1.2, opacity: 0.27, delay: 0.036 },
    { width: 0.9, opacity: 0.16, delay: 0.048 },
    { width: 0.6, opacity: 0.08, delay: 0.06 },
  ];
  return (
    <>
      <path
        d={d}
        stroke={`url(#${gradientId})`}
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        opacity="0.22"
        vectorEffect="non-scaling-stroke"
      />
      {tail.map((t, i) => (
        <path
          key={i}
          d={d}
          pathLength={100}
          stroke={color}
          strokeWidth="2"
          strokeDasharray={`${t.width} ${100 - t.width}`}
          strokeLinecap="round"
          fill="none"
          opacity={t.opacity}
          vectorEffect="non-scaling-stroke"
          style={{
            animation: `cometDash ${duration}s linear infinite`,
            animationDelay: `${t.delay * duration}s`,
            filter: i === 0 ? `drop-shadow(0 0 5px ${color})` : undefined,
          }}
        />
      ))}
    </>
  );
}

function Stage({
  style,
  accent,
  label,
  sub,
  wide = false,
  children,
}: {
  style: React.CSSProperties;
  accent: string;
  label: string;
  sub: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className="absolute"
      style={{
        ...style,
        // Widths are % of the diagram container so the SVG flow-path
        // endpoints (in viewBox units = container %) line up with the
        // actual card edges. Regular halfWidth = 6, wide halfWidth = 9.
        width: wide ? "18%" : "12%",
        minWidth: wide ? "180px" : "126px",
        transform: "translate(-50%, -50%)",
      }}
    >
      <div
        className="rounded-lg border bg-neutral-900/85 backdrop-blur p-2.5 transition-all overflow-hidden"
        style={{
          borderColor: accent + "66",
          boxShadow: `0 0 16px ${accent}22`,
        }}
      >
        <div
          className="text-[10px] uppercase tracking-wider font-mono mb-1.5"
          style={{ color: accent }}
        >
          {label}
        </div>
        {/* Inner viewport — overflow:hidden contains every visualization
            so 3D-translated children, oversized SVGs, etc. can never poke
            past the card's edges. */}
        <div className="rounded bg-neutral-950/60 p-1.5 overflow-hidden">
          {children}
        </div>
        <div className="text-[10px] font-mono text-neutral-500 mt-1.5 text-center">
          {sub}
        </div>
      </div>
    </div>
  );
}

// Side-input node — used for non-pipeline data sources that feed *into*
// a pipeline stage (e.g. the enrolled-voice centroids that Speaker-ID
// compares against, or the initial_prompt that ContextManager assembles
// for Whisper). Visually distinct from a Stage: smaller, with a dashed
// border + a small dot icon, lower elevation. Same coordinate system as
// Stage so SVG flow paths can connect to its edges.
function SideInput({
  style,
  accent,
  label,
  sub,
  children,
}: {
  style: React.CSSProperties;
  accent: string;
  label: string;
  sub: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="absolute"
      style={{
        ...style,
        width: "13%",
        minWidth: "140px",
        transform: "translate(-50%, -50%)",
      }}
    >
      <div
        className="rounded-md border border-dashed bg-neutral-950/70 backdrop-blur p-1.5 overflow-hidden"
        style={{
          borderColor: accent + "55",
          boxShadow: `0 0 8px ${accent}11`,
        }}
      >
        <div
          className="text-[8px] uppercase tracking-wider font-mono mb-1 flex items-center gap-1"
          style={{ color: accent }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: accent, opacity: 0.7 }}
          />
          {label}
        </div>
        <div className="space-y-0.5 px-0.5">{children}</div>
        <div className="text-[7px] font-mono text-neutral-600 mt-1 text-center">
          {sub}
        </div>
      </div>
    </div>
  );
}

// Small label that floats over a connector wire — names the data type
// flowing along it (e.g. "text tokens", "phoneme posterior"). Uses HTML
// (not SVG <text>) so it isn't stretched by the diagram's
// preserveAspectRatio="none". Positioned via top/left percentages that
// share the same coordinate space as the cards and FlowSvg viewBox.
//
// The label has a near-opaque pill background so it reads as a wire
// tag instead of bleeding into whatever card it happens to overlap.
// Without the backdrop, labels positioned in tight gaps between cards
// visually merged with card edges (e.g. "decoder prefix" looked like
// part of the WHISPER header, "cosine compare" looked like part of
// SPEAKER ID's top label).
function EdgeLabel({
  top,
  left,
  children,
}: {
  top: string;
  left: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="absolute font-mono text-[9px] text-neutral-400 pointer-events-none whitespace-nowrap select-none bg-neutral-950/85 px-1.5 py-0.5 rounded"
      style={{
        top,
        left,
        transform: "translate(-50%, -50%)",
      }}
    >
      {children}
    </div>
  );
}

// Realistic speech-shaped waveform from a deterministic envelope (not pure
// sine), with an animated playhead sweeping across — looks like live
// audio being captured. Static shape is meaningful (this is what real
// speech looks like in time), the cursor tells the audience "this is
// being processed in real time."
function MiniWaveformVis() {
  // Speech-like envelope: bursts of activity separated by pauses.
  const samples = useMemo(() => {
    const N = 60;
    const out: number[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / N;
      // Two speech bursts with quiet bookends + small in-utterance pauses.
      const env =
        Math.max(0, Math.sin((t - 0.1) * Math.PI * 1.3)) *
          (t > 0.08 && t < 0.55 ? 1 : 0) +
        Math.max(0, Math.sin((t - 0.6) * Math.PI * 2.3)) *
          (t > 0.6 && t < 0.95 ? 1 : 0);
      const carrier = 0.55 + 0.45 * Math.sin(i * 0.9 + Math.sin(i * 0.3) * 1.7);
      out.push(Math.min(1, env * carrier * 1.1));
    }
    return out;
  }, []);
  return (
    <div className="relative">
      <svg viewBox="0 0 60 24" className="w-full h-10 block" preserveAspectRatio="none">
        {samples.map((v, i) => {
          const h = Math.max(0.6, v * 22);
          return (
            <rect
              key={i}
              x={i}
              y={12 - h / 2}
              width="0.7"
              height={h}
              fill="#94a3b8"
              opacity={0.55 + v * 0.3}
            />
          );
        })}
        {/* Sweeping playhead */}
        <line x1="0" y1="0" x2="0" y2="24" stroke="#cbd5e1" strokeWidth="0.4">
          <animate attributeName="x1" from="0" to="60" dur="3.2s" repeatCount="indefinite" />
          <animate attributeName="x2" from="0" to="60" dur="3.2s" repeatCount="indefinite" />
        </line>
      </svg>
      <div className="text-[8px] font-mono text-neutral-500 text-center mt-1">
        16 kHz · float32
      </div>
    </div>
  );
}

// VAD — shows the SAME envelope from the audio stage, plus the model's
// per-frame speech probability curve overlaid, plus the threshold line
// at 0.5. The dashed emerald box marks where the segmenter merged
// frames into a single utterance. Teaches: "VAD outputs a probability,
// then thresholds it into utterance boundaries."
function VadVis() {
  const N = 60;
  // Two speech bursts get separately-detected by VAD initially, then
  // merged into one utterance because the gap is < silence_close_ms.
  const probs = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const burst1 = Math.max(0, Math.sin((t - 0.08) * Math.PI * 1.6)) * (t > 0.08 && t < 0.5 ? 1 : 0);
      const burst2 = Math.max(0, Math.sin((t - 0.55) * Math.PI * 2.4)) * (t > 0.55 && t < 0.92 ? 1 : 0);
      out.push(Math.min(1, (burst1 + burst2) * 1.1));
    }
    return out;
  }, []);
  const speechStart = 0.08;
  const speechEnd = 0.92;
  // Polyline path for the probability curve.
  const curve = probs
    .map((p, i) => `${(i / (N - 1)) * 60},${22 - p * 18}`)
    .join(" ");
  return (
    <div className="relative">
      <svg viewBox="0 0 60 24" className="w-full h-10 block" preserveAspectRatio="none">
        {/* faint waveform behind */}
        {probs.map((p, i) => {
          const isSpeech = i / N > speechStart && i / N < speechEnd;
          const h = Math.max(0.6, p * 18);
          return (
            <rect
              key={i}
              x={i}
              y={12 - h / 2}
              width="0.7"
              height={h}
              fill={isSpeech ? "#10b981" : "#525252"}
              opacity={isSpeech ? 0.55 : 0.3}
            />
          );
        })}
        {/* threshold line at 0.5 */}
        <line x1="0" y1="13" x2="60" y2="13" stroke="#525252" strokeWidth="0.2" strokeDasharray="1.5 1.5" />
        {/* probability curve */}
        <polyline points={curve} fill="none" stroke="#10b981" strokeWidth="0.5" />
        {/* utterance window */}
        <rect
          x={speechStart * 60}
          y="2"
          width={(speechEnd - speechStart) * 60}
          height="20"
          fill="none"
          stroke="#10b981"
          strokeWidth="0.4"
          strokeDasharray="2 1"
          opacity="0.9"
        />
      </svg>
      <div className="text-[8px] font-mono text-neutral-500 text-center mt-1">
        threshold 0.5 · merge &lt; 3.0 s
      </div>
    </div>
  );
}

// Speaker ID — a similarity meter, NOT random color noise. Shows the
// cosine similarity (0..1) with both decision thresholds drawn:
//   < 0.15 → reject (non-speech / unknown)
//   0.15–0.55 → listener (caregiver)
//   ≥ 0.55 → patient (route through adapter)
// Marker animates from 0 to 0.91 on each cycle so the audience watches
// the score "land" in the patient zone.
function SpeakerVis() {
  return (
    <div className="space-y-1.5 py-1">
      <div className="relative h-3 rounded-full bg-neutral-900 border border-neutral-800 overflow-hidden">
        {/* zone fills */}
        <div className="absolute top-0 left-0 h-full bg-red-900/30" style={{ width: "15%" }} />
        <div className="absolute top-0 h-full bg-amber-900/30" style={{ left: "15%", width: "40%" }} />
        <div className="absolute top-0 h-full bg-emerald-900/30" style={{ left: "55%", width: "45%" }} />
        {/* threshold ticks */}
        <div className="absolute top-0 h-full w-px bg-neutral-400/60" style={{ left: "15%" }} />
        <div className="absolute top-0 h-full w-px bg-neutral-400/60" style={{ left: "55%" }} />
        {/* marker — animates from 0 → 91% each cycle */}
        <div
          className="absolute top-0 h-full w-1 bg-white shadow-[0_0_6px_rgba(255,255,255,0.7)]"
          style={{ animation: "speakerMarker 3s ease-out infinite" }}
        />
      </div>
      <div className="flex justify-between text-[7px] font-mono text-neutral-500 px-0.5">
        <span className="text-red-400">reject</span>
        <span className="text-amber-400">listener</span>
        <span className="text-emerald-400">patient</span>
      </div>
      <div className="text-center text-[10px] font-mono text-emerald-300">
        cos sim 0.91 ✓
      </div>
    </div>
  );
}

// Whisper + LoRA — shows the LoRA decomposition as a math diagram:
//   y = W·x + (B·A)·x
// Whisper takes *two* inputs: audio (mel features from the encoder) **and**
// an initial_prompt — a free-form text prefix that biases the decoder.
// The audio comes from the main pipeline flow (Speaker-ID → here). The
// initial_prompt comes from a separate "context" side-input node above
// the card (see PipelineDiagram + SideInput). On top of those data
// inputs, the LoRA delta calibrates the model weights to the patient's
// voice. The math diagram below shows that calibration: W is frozen,
// B·A is the trainable add-on.
function WhisperVis() {
  return (
    <div className="space-y-1.5 py-1">
      <svg viewBox="0 0 100 56" className="w-full h-16 block">
        {/* Frozen W block */}
        <g>
          <rect x="6" y="14" width="42" height="36" rx="2" fill="#171717" stroke="#525252" strokeWidth="0.5" />
          {/* fake weight grid */}
          {Array.from({ length: 6 }).map((_, r) =>
            Array.from({ length: 7 }).map((_, c) => (
              <rect
                key={`${r}-${c}`}
                x={8 + c * 5.5}
                y={16 + r * 5.5}
                width="4.5"
                height="4.5"
                fill={`rgba(115,115,115,${0.12 + ((r * 7 + c) % 4) * 0.07})`}
              />
            )),
          )}
          <text x="27" y="35" textAnchor="middle" fontSize="6" fontFamily="monospace" fill="#a3a3a3">
            W
          </text>
          <text x="27" y="11" textAnchor="middle" fontSize="3.5" fontFamily="monospace" fill="#737373">
            frozen · 242 M
          </text>
        </g>
        <text x="52" y="34" textAnchor="middle" fontSize="6" fill="#a78bfa">+</text>
        {/* B (tall) */}
        <g>
          <rect x="58" y="20" width="8" height="24" rx="1" fill="#1e1b4b" stroke="#a78bfa" strokeWidth="0.5" />
          {Array.from({ length: 4 }).map((_, r) => (
            <rect
              key={r}
              x="60"
              y={22 + r * 5}
              width="4"
              height="4"
              fill={`rgba(167,139,250,${0.4 + r * 0.15})`}
            />
          ))}
          <text x="62" y="50" textAnchor="middle" fontSize="3.5" fontFamily="monospace" fill="#c4b5fd">B</text>
        </g>
        <text x="69" y="34" textAnchor="middle" fontSize="4" fill="#a78bfa">·</text>
        {/* A (wide) */}
        <g>
          <rect x="72" y="28" width="22" height="8" rx="1" fill="#1e1b4b" stroke="#a78bfa" strokeWidth="0.5" />
          {Array.from({ length: 4 }).map((_, c) => (
            <rect
              key={c}
              x={74 + c * 4.5}
              y="30"
              width="4"
              height="4"
              fill={`rgba(167,139,250,${0.4 + c * 0.12})`}
            />
          ))}
          <text x="83" y="42" textAnchor="middle" fontSize="3.5" fontFamily="monospace" fill="#c4b5fd">A</text>
        </g>
        {/* trainable highlight glow over B·A */}
        <rect
          x="56"
          y="18"
          width="40"
          height="28"
          rx="2"
          fill="none"
          stroke="#a78bfa"
          strokeWidth="0.4"
          opacity="0.5"
        >
          <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" />
        </rect>
        <text x="76" y="11" textAnchor="middle" fontSize="3.5" fontFamily="monospace" fill="#c4b5fd">
          trainable · 6.7 M
        </text>
      </svg>
      <div className="text-[9px] font-mono text-violet-200 text-center leading-tight">
        y = W<span className="text-neutral-500">x</span> + (B·A)
        <span className="text-neutral-500">x</span>
      </div>
      <div className="text-[8px] font-mono text-neutral-500 text-center">
        × 12 enc + 12 dec layers
      </div>
    </div>
  );
}

// Single line in a side-input panel — a faded label and a value that
// truncates rather than wraps so the panel stays compact. Used by the
// Context and Enrolled-voices nodes hanging above the main pipeline row.
function SideInputLine({
  label,
  value,
  labelColor,
}: {
  label: string;
  value: string;
  labelColor: string;
}) {
  return (
    <div className="flex items-baseline gap-1 text-[8px] font-mono leading-tight min-w-0">
      <span className="shrink-0" style={{ color: labelColor }}>
        {label}:
      </span>
      <span className="text-neutral-300 truncate">{value}</span>
    </div>
  );
}

// Wav2vec2 — a small phoneme-posterior heatmap. y-axis = top phoneme
// candidates, x-axis = time, brightness = probability. The argmax phoneme
// per timestep is shown as a label below. This is what wav2vec2 actually
// outputs — a soft probability over phonemes, not just one symbol.
function PhonemeVis() {
  // Hand-crafted illustrative posterior for "the [pause] can play [pause] this":
  // rows = top phonemes (þ ə k æ n p l eɪ ɪ s), cols = time bins.
  // High prob = bright cyan. Each column has one dominant phoneme.
  const rows = ["ð", "ə", "k", "æ", "n", "p", "l", "eɪ", "ɪ", "s"];
  const seq = ["ð", "ə", "_", "k", "æ", "n", "_", "p", "l", "eɪ", "_", "ð", "ɪ", "s"];
  const cols = seq.length;
  // Matrix: prob[r][c] = high if rows[r] is the dominant phoneme at col c.
  const matrix = useMemo(() => {
    const m: number[][] = [];
    for (const row of rows) {
      const r: number[] = [];
      for (const dominant of seq) {
        if (dominant === row) r.push(0.85);
        else if (dominant === "_") r.push(0.05);
        else r.push(0.05 + Math.random() * 0.18); // soft noise
      }
      m.push(r);
    }
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div className="space-y-1 py-0.5">
      <svg viewBox={`0 0 ${cols * 6} 24`} className="w-full h-12 block" preserveAspectRatio="none">
        {matrix.flatMap((row, r) =>
          row.map((p, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * 6}
              y={r * 2.4}
              width="5.6"
              height="2.2"
              fill={`rgba(34,211,238,${p})`}
            />
          )),
        )}
      </svg>
      <div className="flex justify-between font-mono text-[10px] text-cyan-200 px-px">
        {seq.map((s, i) => (
          <span key={i} className={s === "_" ? "text-neutral-600" : ""}>
            {s === "_" ? "·" : s}
          </span>
        ))}
      </div>
      <div className="text-[8px] font-mono text-neutral-500 text-center mt-0.5">
        50 Hz · 393 phoneme classes
      </div>
    </div>
  );
}

// Safety gate — shows the actual operation: Whisper's text gets phonemized
// (top row, "intended"), wav2vec2 reads the audio (bottom row, "observed"),
// and the gate scores how well they line up. Each column is one aligned
// pair so the connectors land exactly under their phonemes. The mean
// log-prob meter below is the gate's actual decision signal.
function GateVis() {
  const intended = ["ð", "ə", "k", "æ", "n"];
  const observed = ["ð", "ə", "k", "æ", "n"];
  const cols = `repeat(${intended.length}, minmax(0, 1fr))`;
  return (
    <div className="space-y-2 py-1">
      <div className="space-y-0">
        <div
          className="grid font-mono text-[11px] text-violet-200 text-center"
          style={{ gridTemplateColumns: cols }}
        >
          {intended.map((p, i) => (
            <div key={i}>{p}</div>
          ))}
        </div>
        <div className="grid" style={{ gridTemplateColumns: cols }}>
          {intended.map((_, i) => (
            <div key={i} className="flex justify-center py-0.5">
              <div
                className="w-px h-3 bg-emerald-400/70"
                style={{ animation: `pulseFade 2.4s ease-in-out ${i * 0.15}s infinite` }}
              />
            </div>
          ))}
        </div>
        <div
          className="grid font-mono text-[11px] text-cyan-200 text-center"
          style={{ gridTemplateColumns: cols }}
        >
          {observed.map((p, i) => (
            <div key={i}>{p}</div>
          ))}
        </div>
      </div>
      {/* Alignment score meter — log-prob, threshold at -0.85.
          Threshold position along the (-8 → 0) axis: (8-0.85)/8 ≈ 89.4% */}
      <div className="pt-0.5">
        <div className="relative h-2 rounded-full bg-neutral-900 border border-neutral-800 overflow-hidden">
          <div className="absolute top-0 h-full w-px bg-neutral-400/70" style={{ left: "89.4%" }} />
          <div
            className="absolute top-0 h-full bg-gradient-to-r from-emerald-700/40 to-emerald-500/80"
            style={{ width: "96.6%" }}
          />
          <div
            className="absolute top-0 h-full w-1 bg-white shadow-[0_0_4px_rgba(255,255,255,0.7)]"
            style={{ left: "calc(96.6% - 2px)" }}
          />
        </div>
        <div className="flex justify-between text-[7px] font-mono text-neutral-500 mt-0.5 px-0.5">
          <span>−8</span>
          <span className="text-neutral-400">thr −0.85</span>
          <span>0</span>
        </div>
      </div>
      <div className="text-[10px] font-mono text-emerald-300 text-center">
        align −0.27 ✓
      </div>
    </div>
  );
}

// Output — what the listener UI actually surfaces. Mock-up of a speech
// bubble with a green verdict pill. The text "types in" character by
// character on a loop so the audience sees output emerging.
function OutputVis() {
  const text = "If you destroy…";
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setShown((n) => (n >= text.length ? 0 : n + 1));
    }, 90);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="space-y-1 py-1.5">
      <div className="rounded border border-emerald-700/40 bg-emerald-950/30 px-1.5 py-1.5">
        <div className="text-[10px] text-emerald-100 leading-tight italic min-h-[14px]">
          “{text.slice(0, shown)}
          <span className="text-emerald-400/60">▎</span>”
        </div>
      </div>
      <div className="flex items-center gap-1 justify-center">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-[8px] font-mono text-emerald-300/80 uppercase tracking-wider">
          ok · confidence high
        </span>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────── Phase 3 — Two strengths

// Combined strengths slide. Replaces the older two-slide pair (LoraSlide
// + PhonemeGateSlide) — both ran too dense for a hackathon scan. Each
// strength gets its own card with: brief description, one concrete
// proof element (WER reduction chart / caught-hallucination example),
// and three compact stat chips. Designed for ~20 s read.
function StrengthsSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-12 py-7 max-w-7xl mx-auto w-full">
      <header className="mb-5">
        <div className="text-neutral-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          What makes Phonemer work
        </div>
        <h1 className="text-4xl font-semibold tracking-tight">
          <span className="text-violet-300">Two strengths</span>, each addressing
          a different Whisper failure mode.
        </h1>
      </header>

      <div className="grid lg:grid-cols-2 gap-5 flex-1 min-h-0">
        <StrengthCard
          n="1"
          title="Per-patient adaptation"
          accent="#a78bfa"
          description={
            <>
              Whisper learns this patient two ways: an offline{" "}
              <span className="text-violet-300">LoRA adapter</span> trained on
              their voice, plus{" "}
              <span className="text-violet-300">runtime context</span> (last
              listener turn, active topics, patient vocabulary) fed to Whisper
              at every call.
            </>
          }
          note={
            <>
              <span className="text-violet-300 font-semibold">LoRA</span>{" "}
              (Low-Rank Adaptation): keep Whisper&apos;s 242 M weights frozen,
              learn a tiny additive correction{" "}
              <code className="text-violet-200 font-mono">y = Wx + (B·A)x</code>{" "}
              where <code className="font-mono">B</code> and{" "}
              <code className="font-mono">A</code> are small matrices.
              Personalized output, same inference speed.
            </>
          }
          proofLabel="Measured WER reduction · TORGO eval split"
          proof={<WerProof />}
          stats={[
            { label: "adapter", value: "~27 MB" },
            { label: "trainable", value: "6.7 M" },
            { label: "frozen", value: "242 M" },
          ]}
        />
        <StrengthCard
          n="2"
          title="Phoneme safety gate"
          accent="#22d3ee"
          description={
            <>
              <span className="text-cyan-300">wav2vec2</span> reads the audio
              independently. If Whisper&apos;s text doesn&apos;t acoustically
              fit, we flag it as uncertain rather than surface a confident
              hallucination.
            </>
          }
          proofLabel="Caught hallucination · F03 reading TIMIT"
          proof={<CaughtProof />}
          stats={[
            { label: "align", value: "< −0.85" },
            { label: "over-prod", value: "> 25 ph/s" },
            { label: "under-prod", value: "< 4 ph/s" },
          ]}
        />
      </div>
    </div>
  );
}

// Card scaffold for the two strengths. Big tinted background, number
// badge, title, description, proof block, stat chips footer. Both
// cards share this so visual weight stays balanced.
function StrengthCard({
  n,
  title,
  accent,
  description,
  note,
  proofLabel,
  proof,
  stats,
}: {
  n: string;
  title: string;
  accent: string;
  description: React.ReactNode;
  note?: React.ReactNode;
  proofLabel: string;
  proof: React.ReactNode;
  stats: { label: string; value: string }[];
}) {
  return (
    <div
      className="rounded-2xl border bg-gradient-to-br from-neutral-900/60 to-neutral-950 p-6 flex flex-col gap-4 overflow-hidden"
      style={{
        borderColor: accent + "66",
        boxShadow: `0 0 24px ${accent}14`,
      }}
    >
      {/* Header */}
      <div>
        <div className="flex items-center gap-2.5 mb-2">
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold"
            style={{
              backgroundColor: accent + "22",
              border: `1px solid ${accent}88`,
              color: accent,
            }}
          >
            {n}
          </div>
          <div
            className="text-[11px] uppercase tracking-[0.2em] font-mono"
            style={{ color: accent }}
          >
            Strength {n}
          </div>
        </div>
        <h2
          className="text-2xl font-semibold tracking-tight mb-2"
          style={{ color: accent }}
        >
          {title}
        </h2>
        <p className="text-sm text-neutral-400 leading-relaxed">{description}</p>

        {/* Plain-language explainer for the bit of jargon in the
            description (e.g. "what's a LoRA?"). Styled as a subtle
            info-tile so it reads as supplementary, not part of the
            main pitch. */}
        {note && (
          <div
            className="mt-3 rounded-lg border bg-neutral-950/50 px-3 py-2 flex gap-2.5 items-start"
            style={{ borderColor: accent + "33" }}
          >
            <span
              className="shrink-0 mt-0.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-mono font-bold"
              style={{
                backgroundColor: accent + "33",
                color: accent,
              }}
            >
              i
            </span>
            <span className="text-[11.5px] text-neutral-400 leading-relaxed">
              {note}
            </span>
          </div>
        )}
      </div>

      {/* Proof block */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-mono mb-2">
          {proofLabel}
        </div>
        <div className="flex-1 min-h-0">{proof}</div>
      </div>

      {/* Stat chips */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-neutral-800/80">
        {stats.map((s) => (
          <div key={s.label}>
            <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-0.5">
              {s.label}
            </div>
            <div
              className="text-sm font-mono font-semibold"
              style={{ color: accent }}
            >
              {s.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Inline proof for Strength 1 — WER before/after for the three TORGO
// speakers in one tidy block. Reuses the ResultStat scaffold from the
// older slide for consistency.
function WerProof() {
  return (
    <div className="space-y-2">
      <ResultStat label="FC02 (control)" before="WER 0.27" after="WER 0.17" delta="−35%" />
      <ResultStat label="F03 (mild)" before="WER 0.70" after="WER 0.47" delta="−32%" />
      <ResultStat label="M01 (severe)" before="WER 1.02" after="WER 0.54" delta="−47%" highlight />
    </div>
  );
}

// Inline proof for Strength 2 — a single caught hallucination with the
// alignment score landing in the flag zone. Compact version of the
// older GateExampleCard, trimmed to fit alongside the WER chart.
function CaughtProof() {
  return (
    <div className="space-y-2.5">
      <div>
        <span className="text-[10px] text-neutral-500 italic">intended: </span>
        <span className="text-xs italic text-neutral-300">
          “A roll of wire lay near the wall”
        </span>
      </div>
      <div>
        <div className="flex items-center gap-1.5 mb-0.5">
          <span className="inline-block w-1 h-1 rounded-full bg-violet-400" />
          <span className="text-[10px] uppercase tracking-wider text-violet-400 font-mono">
            Whisper heard
          </span>
        </div>
        <div className="text-sm text-red-300 leading-snug">
          I need a little wired lay Diego mall
        </div>
      </div>
      <div>
        <div className="flex items-center gap-1.5 mb-0.5">
          <span className="inline-block w-1 h-1 rounded-full bg-cyan-400" />
          <span className="text-[10px] uppercase tracking-wider text-cyan-400 font-mono">
            wav2vec2 heard
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {["ə", "ɹ", "oʊ", "l", "ə", "v", "w", "aɪ", "ɚ", "l", "eɪ", "n"].map(
            (p, i) => (
              <span
                key={i}
                className="px-1 py-0 rounded text-[11px] font-mono bg-cyan-950/40 border border-cyan-800/40 text-cyan-200"
              >
                {p}
              </span>
            ),
          )}
          <span className="text-[10px] text-neutral-600 self-center">…</span>
        </div>
      </div>
      <div className="flex items-center justify-between pt-2 border-t border-neutral-800/80">
        <div className="text-[11px] text-neutral-400">
          align score{" "}
          <span className="font-mono text-red-300">−1.4</span>
          <span className="text-neutral-600"> (thr −0.85) →</span>
        </div>
        <div className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-amber-900/40 border border-amber-700/60 text-amber-300">
          ● flagged
        </div>
      </div>
    </div>
  );
}

function LoraSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-12 py-5 max-w-7xl mx-auto w-full">
      <header className="mb-3">
        <div className="text-violet-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          Strength 1 · Per-patient adaptation
        </div>
        <h1 className="text-3xl font-semibold tracking-tight mb-2">
          Two ways Whisper learns{" "}
          <span className="text-violet-300">this</span> patient.
        </h1>
        <p className="text-sm text-neutral-400 max-w-3xl">
          An <span className="text-violet-300">offline LoRA adapter</span>{" "}
          trained on the patient&apos;s voice, plus{" "}
          <span className="text-violet-300">runtime context</span> (last
          listener turn, active topics, patient vocab) fed to Whisper at
          every inference call.
        </p>
      </header>

      <div className="grid lg:grid-cols-5 gap-4 flex-1 min-h-0">
        {/* Mechanism 1 — offline LoRA */}
        <div className="lg:col-span-3 flex flex-col min-h-0">
          <SectionLabel
            accent="#a78bfa"
            label="Offline · LoRA adapter"
            sub="trained once per patient, lives on disk"
          />
          <div className="grid grid-cols-5 gap-3 mt-2.5 flex-1 min-h-0">
            <div className="col-span-3 min-h-0">
              <LoraVisualization />
            </div>
            <div className="col-span-2 flex flex-col gap-2.5">
              <Equation />
              <div className="grid grid-cols-2 gap-2">
                <FactBig label="trainable" value="6.7 M" sub="LoRA params" accent="#a78bfa" />
                <FactBig label="frozen" value="242 M" sub="Whisper" accent="#737373" />
                <FactBig label="size" value="~27 MB" sub="per patient" accent="#a78bfa" />
                <FactBig label="targets" value="q·k·v·o" sub="+ fc1·fc2" accent="#a78bfa" />
              </div>
            </div>
          </div>
        </div>

        {/* Mechanism 2 — runtime initial_prompt context */}
        <div className="lg:col-span-2 flex flex-col min-h-0">
          <SectionLabel
            accent="#a78bfa"
            label="Runtime · initial_prompt"
            sub="≤ 220 chars · ContextManager"
          />
          <ContextMechanism />
        </div>
      </div>

      {/* Measured improvement — concrete numbers for Strength 1 */}
      <div className="mt-4">
        <div className="flex items-baseline justify-between border-b border-neutral-800/80 pb-1.5 mb-2.5">
          <div className="flex items-center gap-2">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400" />
            <span className="text-[11px] uppercase tracking-[0.2em] text-violet-400 font-mono">
              Measured improvement
            </span>
          </div>
          <span className="text-[10px] uppercase tracking-wider font-mono text-neutral-600">
            vanilla → adapter · TORGO eval split
          </span>
        </div>
        <div className="grid md:grid-cols-3 gap-3">
          <ResultStat label="FC02 (control)" before="WER 0.27" after="WER 0.17" delta="−35%" />
          <ResultStat label="F03 (mild)" before="WER 0.70" after="WER 0.47" delta="−32%" />
          <ResultStat label="M01 (severe)" before="WER 1.02" after="WER 0.54" delta="−47%" highlight />
        </div>
      </div>
    </div>
  );
}

// Runtime-context panel for Strength 1. Shows what the ContextManager
// assembles into `initial_prompt` before every Whisper call — recent
// listener turn, conversation topics, patient-specific vocab. Mirrors
// the side-input node in the architecture diagram but explains *why*
// each channel matters. Sample values are illustrative; real ones come
// from `ContextManager.whisper_prompt()` at runtime.
function ContextMechanism() {
  return (
    <div className="mt-2.5 flex-1 min-h-0 flex flex-col gap-2.5">
      <div className="rounded-xl border border-violet-800/40 bg-neutral-950/60 px-4 py-3 font-mono text-[11px] space-y-2 leading-snug">
        <div className="text-[9px] uppercase tracking-wider text-violet-300 mb-1">
          initial_prompt =
        </div>
        <ContextLine
          label="last turn"
          value={"“Could you read this for me?”"}
          note="what the listener just said"
        />
        <ContextLine
          label="topics"
          value="banks, money, gabapentin…"
          note="active conversation subjects"
        />
        <ContextLine
          label="vocab"
          value="Dr. Park, gabapentin, valproate…"
          note="patient-specific terminology"
        />
      </div>

      <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 px-4 py-3 text-[12px] text-neutral-400 leading-relaxed">
        Whisper is a generative LM, not just an ASR. Priming the decoder
        with the patient&apos;s recent context biases borderline word choices
        toward what they&apos;ve actually been talking about —{" "}
        <span className="text-violet-300">&ldquo;gabapentin&rdquo;</span>{" "}
        instead of{" "}
        <span className="text-neutral-500">&ldquo;gabba pen tin&rdquo;</span>,{" "}
        <span className="text-violet-300">&ldquo;Dr. Park&rdquo;</span> instead
        of <span className="text-neutral-500">&ldquo;doctor parked&rdquo;</span>.
      </div>
    </div>
  );
}

// One row inside the mock initial_prompt panel. Channel label + value
// + grayed-out "what this is" caption underneath.
function ContextLine({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="flex flex-col">
      <div className="flex items-baseline gap-2 min-w-0">
        <span className="shrink-0 text-violet-300">{label}:</span>
        <span className="truncate text-neutral-200">{value}</span>
      </div>
      <span className="text-[9px] text-neutral-600 ml-[5.5rem] mt-0.5 leading-none">
        {note}
      </span>
    </div>
  );
}

function LoraVisualization() {
  return (
    <div className="rounded-2xl border border-violet-800/40 bg-neutral-900/40 p-6 flex items-center justify-center">
      <svg viewBox="0 0 400 320" className="w-full h-auto max-h-[480px]">
        {/* Big frozen W matrix */}
        <g>
          <rect x="20" y="20" width="180" height="180" fill="#171717" stroke="#525252" strokeWidth="1.5" rx="4" />
          {/* fake weight grid */}
          {Array.from({ length: 8 }).map((_, r) =>
            Array.from({ length: 8 }).map((_, c) => (
              <rect
                key={`${r}-${c}`}
                x={26 + c * 21.5}
                y={26 + r * 21.5}
                width="18"
                height="18"
                fill={`rgba(115,115,115,${0.18 + ((r * 7 + c) % 5) * 0.08})`}
              />
            )),
          )}
          <text x="110" y="220" textAnchor="middle" fontSize="14" fill="#d4d4d4" fontFamily="monospace">
            W (frozen)
          </text>
          <text x="110" y="240" textAnchor="middle" fontSize="11" fill="#737373" fontFamily="monospace">
            768 × 768
          </text>
        </g>
        <text x="218" y="115" fontSize="32" fill="#a78bfa" textAnchor="middle">+</text>
        {/* B matrix (tall) */}
        <g>
          <rect x="240" y="40" width="32" height="140" fill="#1e1b4b" stroke="#a78bfa" strokeWidth="1.5" rx="3" />
          {Array.from({ length: 7 }).map((_, r) =>
            Array.from({ length: 1 }).map((_, c) => (
              <rect
                key={`b-${r}-${c}`}
                x={244 + c * 12}
                y={44 + r * 19.5}
                width="24"
                height="16"
                fill={`rgba(167,139,250,${0.4 + (r % 3) * 0.2})`}
              />
            )),
          )}
          <text x="256" y="200" textAnchor="middle" fontSize="13" fill="#c4b5fd" fontFamily="monospace">
            B
          </text>
          <text x="256" y="216" textAnchor="middle" fontSize="10" fill="#737373" fontFamily="monospace">
            768 × 16
          </text>
        </g>
        <text x="290" y="115" fontSize="22" fill="#a78bfa" textAnchor="middle">·</text>
        {/* A matrix (wide) */}
        <g>
          <rect x="305" y="100" width="76" height="32" fill="#1e1b4b" stroke="#a78bfa" strokeWidth="1.5" rx="3" />
          {Array.from({ length: 1 }).map((_, r) =>
            Array.from({ length: 4 }).map((_, c) => (
              <rect
                key={`a-${r}-${c}`}
                x={309 + c * 18}
                y={104 + r * 12}
                width="14"
                height="22"
                fill={`rgba(167,139,250,${0.4 + (c % 3) * 0.2})`}
              />
            )),
          )}
          <text x="343" y="156" textAnchor="middle" fontSize="13" fill="#c4b5fd" fontFamily="monospace">
            A
          </text>
          <text x="343" y="172" textAnchor="middle" fontSize="10" fill="#737373" fontFamily="monospace">
            16 × 768
          </text>
        </g>
        {/* "Trainable" badge — points the audience at B and A without
            relying on backprop notation. Replaces the earlier
            ∂L/∂θ arrows that curved down to nothing visible. */}
        <g>
          <rect
            x="220"
            y="220"
            width="180"
            height="26"
            rx="13"
            fill="#1e1b4b"
            stroke="#a78bfa"
            strokeWidth="1"
            opacity="0.9"
          />
          <circle cx="238" cy="233" r="3.5" fill="#a78bfa">
            <animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite" />
          </circle>
          <text x="252" y="238" fontSize="12" fill="#c4b5fd" fontFamily="monospace">
            B and A — trainable
          </text>
        </g>
        {/* W frozen badge — symmetric callout below W so the
            "what updates / what stays" story reads at a glance. */}
        <g>
          <rect
            x="40"
            y="262"
            width="140"
            height="24"
            rx="12"
            fill="#171717"
            stroke="#525252"
            strokeWidth="1"
          />
          <circle cx="56" cy="274" r="3.5" fill="#737373" />
          <text x="70" y="279" fontSize="12" fill="#a3a3a3" fontFamily="monospace">
            W — frozen
          </text>
        </g>
      </svg>
    </div>
  );
}

function Equation() {
  return (
    <div className="rounded-lg bg-neutral-950 border border-violet-800/40 p-5 text-center">
      <div className="font-mono text-2xl text-neutral-200">
        y = W<span className="text-neutral-500">x</span> +{" "}
        <span className="text-violet-300">(B·A)</span>
        <span className="text-neutral-500">x</span>
      </div>
      <div className="mt-2 font-mono text-xs text-neutral-500">
        W frozen · B (768×16), A (16×768) trainable
      </div>
    </div>
  );
}

function FactBig({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent: string;
}) {
  return (
    <div
      className="rounded-lg border bg-neutral-900/40 p-3"
      style={{ borderColor: accent + "55" }}
    >
      <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
        {label}
      </div>
      <div className="font-mono text-2xl" style={{ color: accent }}>
        {value}
      </div>
      <div className="text-[10px] font-mono text-neutral-500 mt-0.5">{sub}</div>
    </div>
  );
}

// Strength 2 — Phoneme safety gate. Showcases the "hallucination check"
// half of Phonemer: a second model (wav2vec2) reads the audio
// independently, and we use forced alignment to score whether Whisper's
// proposed text acoustically fits. The slide is anchored by a concrete
// caught-hallucination example so judges see the strength in action,
// not just three threshold numbers.
function PhonemeGateSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-12 py-6 max-w-7xl mx-auto w-full">
      <header className="mb-4">
        <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          Strength 2 · Phoneme safety gate
        </div>
        <h1 className="text-4xl font-semibold tracking-tight mb-2">
          A <span className="text-cyan-300">second model</span> reads the audio.
          Hallucinations get flagged.
        </h1>
        <p className="text-sm text-neutral-400 max-w-3xl">
          Whisper&apos;s worst failure on impaired speech: fluent hallucinations
          that sound plausible. wav2vec2 gives an independent phoneme reading
          — if Whisper&apos;s text doesn&apos;t acoustically fit, we flag it.
        </p>
      </header>

      <div className="grid lg:grid-cols-5 gap-5 flex-1 min-h-0">
        {/* Concrete caught example */}
        <div className="lg:col-span-3 flex flex-col">
          <SectionLabel
            accent="#06b6d4"
            label="A caught hallucination"
            sub="F03 · TORGO eval"
          />
          <GateExampleCard
            intended="A roll of wire lay near the wall"
            whisperHeard="I need a little wired lay Diego mall"
            wav2vec2Phonemes={["ə", "ɹ", "oʊ", "l", "ə", "v", "w", "aɪ", "ɚ", "l", "eɪ", "n", "ɪ", "ɹ", "w", "ɔ", "l"]}
            alignScore={-1.4}
            threshold={-0.85}
          />
        </div>

        {/* The 3 gate checks */}
        <div className="lg:col-span-2 flex flex-col">
          <SectionLabel
            accent="#06b6d4"
            label="The three checks"
            sub="any one flags the clip"
          />
          <div className="grid grid-cols-1 gap-2.5 mt-3">
            <CheckCard
              title="Alignment"
              what="mean force-aligned log-prob"
              threshold="< −0.85"
              catches="fluent hallucination"
              icon={<AlignIcon />}
            />
            <CheckCard
              title="Phoneme density · high"
              what="more phonemes than acoustically fit"
              threshold="> 25 ph/s"
              catches="over-production"
              icon={<DensityHighIcon />}
            />
            <CheckCard
              title="Phoneme density · low"
              what="Whisper bailed mid-utterance (≥ 2 s clips)"
              threshold="< 4 ph/s voiced"
              catches="under-production"
              icon={<DensityLowIcon />}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Big "see the strength in action" card. Shows the input intended text,
// Whisper's (wrong) output, wav2vec2's parallel phoneme reading, and the
// alignment score landing below the threshold — verdict pill at the
// right. Designed so the judge can follow the catch left-to-right
// without reading the supporting prose.
function GateExampleCard({
  intended,
  whisperHeard,
  wav2vec2Phonemes,
  alignScore,
  threshold,
}: {
  intended: string;
  whisperHeard: string;
  wav2vec2Phonemes: string[];
  alignScore: number;
  threshold: number;
}) {
  // Threshold position on a [-8, 0] axis used by the score meter.
  const thresholdPct = ((8 + threshold) / 8) * 100;
  const scorePct = ((8 + alignScore) / 8) * 100;
  const flagged = alignScore < threshold;
  return (
    <div className="mt-3 rounded-2xl border border-cyan-800/40 bg-neutral-900/40 p-5 flex-1 flex flex-col gap-3.5 overflow-hidden">
      {/* Intended sentence (what the patient was actually trying to say) */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-mono mb-0.5">
          what they said
        </div>
        <div className="text-sm italic text-neutral-300">“{intended}”</div>
      </div>

      {/* Whisper's hallucinated output */}
      <div>
        <div className="flex items-center gap-2 mb-0.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400" />
          <span className="text-[10px] uppercase tracking-wider text-violet-400 font-mono">
            Whisper heard
          </span>
        </div>
        <div className="text-base text-red-300 leading-snug">{whisperHeard}</div>
      </div>

      {/* wav2vec2's independent phoneme reading */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400" />
          <span className="text-[10px] uppercase tracking-wider text-cyan-400 font-mono">
            wav2vec2 heard
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {wav2vec2Phonemes.map((p, i) => (
            <span
              key={i}
              className="px-1.5 py-0.5 rounded text-[12px] font-mono bg-cyan-950/40 border border-cyan-800/40 text-cyan-200"
            >
              {p}
            </span>
          ))}
        </div>
      </div>

      {/* Alignment score meter — log-prob axis [-8, 0], threshold at -0.85 */}
      <div className="mt-1">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-mono">
            forced-alignment score
          </span>
          <span className="text-[10px] font-mono text-neutral-500">
            {alignScore.toFixed(2)} log-prob · threshold {threshold}
          </span>
        </div>
        <div className="relative h-3 rounded-full bg-neutral-900 border border-neutral-800 overflow-hidden">
          {/* Pass zone */}
          <div
            className="absolute top-0 h-full bg-emerald-700/30"
            style={{ left: `${thresholdPct}%`, width: `${100 - thresholdPct}%` }}
          />
          {/* Fail zone */}
          <div
            className="absolute top-0 h-full bg-red-900/30"
            style={{ left: 0, width: `${thresholdPct}%` }}
          />
          {/* Threshold tick */}
          <div
            className="absolute top-0 h-full w-px bg-neutral-300/70"
            style={{ left: `${thresholdPct}%` }}
          />
          {/* Score marker */}
          <div
            className="absolute top-0 h-full w-1 bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.7)]"
            style={{ left: `calc(${scorePct}% - 2px)` }}
          />
        </div>
        <div className="flex justify-between text-[9px] font-mono text-neutral-600 mt-0.5">
          <span>−8 (worst)</span>
          <span style={{ marginLeft: `${thresholdPct - 8}%` }}>thr {threshold}</span>
          <span>0 (best)</span>
        </div>
      </div>

      {/* Verdict pill */}
      <div className="flex items-center justify-between pt-2 border-t border-neutral-800/80">
        <div className="text-xs text-neutral-400">
          {flagged ? (
            <>
              Text still surfaced — but marked{" "}
              <span className="text-amber-300">uncertain</span> so the listener
              knows not to act on it.
            </>
          ) : (
            <>Verdict accepted; text surfaced confidently.</>
          )}
        </div>
        <div
          className={`shrink-0 ml-3 px-2.5 py-1 rounded-md text-[11px] font-mono uppercase tracking-wider ${
            flagged
              ? "bg-amber-900/40 border border-amber-700/60 text-amber-300"
              : "bg-emerald-900/40 border border-emerald-700/60 text-emerald-300"
          }`}
        >
          {flagged ? "● flagged · low_alignment" : "● accepted"}
        </div>
      </div>
    </div>
  );
}

// One row in the "three checks" panel. Compact: icon + title, then
// what / threshold / catches. Designed to scan in ~3 s per row so the
// audience walks away with "any one check flags the clip."
function CheckCard({
  title,
  what,
  threshold,
  catches,
  icon,
}: {
  title: string;
  what: string;
  threshold: string;
  catches: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-cyan-800/40 bg-neutral-900/40 px-3.5 py-3 flex items-start gap-3">
      <div
        className="shrink-0 w-9 h-9 rounded-md flex items-center justify-center text-cyan-300"
        style={{
          backgroundColor: "#06b6d418",
          border: "1px solid #06b6d455",
        }}
      >
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-neutral-100 leading-tight">
          {title}
        </div>
        <div className="text-[11px] text-neutral-400 mt-0.5 leading-snug">
          {what}
        </div>
        <div className="flex items-baseline gap-2 mt-1.5 font-mono text-[10px]">
          <span className="text-cyan-300">{threshold}</span>
          <span className="text-neutral-600">·</span>
          <span className="text-neutral-500">catches {catches}</span>
        </div>
      </div>
    </div>
  );
}

// Inline icons for the three gate checks. Hand-rolled to echo the
// concept of each check (alignment crosses, dense bars, sparse bars).
function AlignIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <circle cx="6" cy="7" r="1.2" fill="currentColor" />
      <circle cx="6" cy="17" r="1.2" fill="currentColor" />
      <circle cx="18" cy="7" r="1.2" fill="currentColor" />
      <circle cx="18" cy="17" r="1.2" fill="currentColor" />
      <line x1="6" y1="7" x2="18" y2="7" stroke="currentColor" strokeWidth="1" />
      <line x1="6" y1="17" x2="18" y2="17" stroke="currentColor" strokeWidth="1" />
      <line x1="6" y1="7" x2="18" y2="17" stroke="currentColor" strokeWidth="1.4" opacity="0.85" />
      <line x1="18" y1="7" x2="6" y2="17" stroke="currentColor" strokeWidth="1.4" opacity="0.85" />
    </svg>
  );
}

function DensityHighIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <rect x="3" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="5.5" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="8" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="10.5" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="13" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="15.5" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="18" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="20.5" y="6" width="1.5" height="12" fill="currentColor" />
    </svg>
  );
}

function DensityLowIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" aria-hidden>
      <rect x="4" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="11" y="6" width="1.5" height="12" fill="currentColor" />
      <rect x="18" y="6" width="1.5" height="12" fill="currentColor" />
    </svg>
  );
}

function ResultStat({
  label,
  before,
  after,
  delta,
  highlight = false,
}: {
  label: string;
  before: string;
  after: string;
  delta: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border px-4 py-3 ${
        highlight
          ? "border-emerald-700/60 bg-emerald-950/15"
          : "border-neutral-800 bg-neutral-900/40"
      }`}
    >
      <div className="text-xs font-mono text-neutral-500 mb-1">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-sm text-amber-300/80 line-through">
          {before}
        </span>
        <span className="text-neutral-500">→</span>
        <span className="font-mono text-sm text-emerald-300">{after}</span>
        <span className="ml-auto font-mono text-base text-emerald-400">
          {delta}
        </span>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────── Phase 4 — Live demo

// Module-scope so the dedupe survives Next.js' dev StrictMode
// mount→unmount→remount cycle (a component-scope useRef gets a fresh
// instance on the remount and would let the auto-run fire twice,
// racing two parallel /torgo/audio fetches and surfacing one of them
// as "Failed to fetch"). Module state persists across the simulated
// remount; production builds run a single mount so this cache simply
// accumulates one entry per clip viewed during the session.
const _demoAutoRunClipIds = new Set<number>();

// The unified demo slide. One screen, one message: a "normal" speech
// model (off-the-shelf Whisper) confidently produces nonsense on the
// same audio that Phonemer (per-patient adapter + phoneme safety gate)
// either gets right or honestly flags as uncertain.
//
// Design priorities for a judge:
//   * One click → see the difference (auto-runs on patient change).
//   * The intended text is centerpiece — the question "did the model
//     hear what the patient was actually saying?" is what the audience
//     is judging.
//   * Two equal-weight side-by-side cards using DiffText so the wrong
//     words are visually obvious in both halves.
//   * Hero delta below makes the WER reduction quantitative.
//   * Pipeline internals (waveform / VAD / phoneme strip) tucked into
//     a collapsed details block — they're the proof, not the message.
function DemoSlide({
  clips,
  pipeline,
  setSelectedClipId,
  setSpeakerFilter,
  runPipeline,
  selectedClip,
  adapterPath,
  audio,
  isPlaying,
  playheadMs,
}: SlideProps) {
  const [picked, setPicked] = useState<string>("F03");
  // Index into the eligible-clips list for the current patient. Lets the
  // user (or judge) cycle through multiple examples per patient — important
  // for showing that severe-dysarthria patients have BOTH outcomes in their
  // adapter: some clips Phonemer flags ("honest uncertainty"), others
  // Phonemer transcribes correctly while vanilla fails ("better accuracy").
  // A single example doesn't tell that whole story.
  const [exampleIndex, setExampleIndex] = useState(0);

  // All eligible demo clips for the current patient: TIMIT sentences ≥4
  // words (real-life-shaped utterances, not one-word stimuli) in stable
  // id order so the cycler is deterministic across renders.
  const eligibleClips = useMemo(() => {
    return (clips ?? [])
      .filter(
        (c) =>
          c.speaker_id === picked &&
          c.prompt_set === "timit" &&
          (c.intended_text?.split(/\s+/).filter(Boolean).length ?? 0) >= 4,
      )
      .slice()
      .sort((a, b) => a.id - b.id);
  }, [clips, picked]);

  // Reset the example index when patient changes — the picker is per-patient.
  useEffect(() => {
    setExampleIndex(0);
  }, [picked]);

  useEffect(() => {
    setSpeakerFilter?.(picked);
    if (eligibleClips.length === 0) return;
    const i = ((exampleIndex % eligibleClips.length) + eligibleClips.length) %
      eligibleClips.length;
    setSelectedClipId?.(eligibleClips[i].id);
  }, [picked, exampleIndex, eligibleClips, setSelectedClipId, setSpeakerFilter]);

  // Auto-fire the pipeline when the patient (and therefore selectedClip)
  // changes. Lets a judge click a chip and see the comparison without an
  // intermediate "Run" click. The "have we already auto-run this clip?"
  // check uses ``_demoAutoRunClipIds`` (module-scope) so it survives
  // StrictMode's mount → unmount → remount cycle in dev — a component-
  // scoped useRef would be reset on the remount and let the effect
  // race two parallel pipelines. The pipeline.status check is a second
  // line of defense: even if the cache misses for some reason, we
  // never queue a second run while one is already in flight.
  useEffect(() => {
    if (!selectedClip || !runPipeline) return;
    if (pipeline.status === "running") return;
    if (_demoAutoRunClipIds.has(selectedClip.id)) return;
    _demoAutoRunClipIds.add(selectedClip.id);
    void runPipeline();
  }, [selectedClip, runPipeline, pipeline.status]);

  useAudioAutoplay(
    audio,
    pipeline.phonemeFramesByUtt &&
      Object.keys(pipeline.phonemeFramesByUtt).length > 0,
  );

  // Joined adapter output across utterances (VAD can split one clip
  // into N segments; concatenating preserves the full transcript).
  const adapted = pipeline.translations
    .slice()
    .sort((a, b) => a.utterance_id - b.utterance_id)
    .map((t) => t.primary)
    .filter(Boolean)
    .join(" ");

  const ref = selectedClip?.intended_text ?? null;
  const vw = ref ? wer(ref, pipeline.vanillaText) : null;
  const aw = ref ? wer(ref, adapted || null) : null;
  const anyFlagged = pipeline.translations.some((t) => t.flagged);
  const allFrames: PhonemeWindowMessage["frames"] = [];
  for (const u of pipeline.utterances) {
    const fr = pipeline.phonemeFramesByUtt[u.utterance_id] ?? [];
    allFrames.push(...fr);
  }
  allFrames.sort((a, b) => a.timestamp_ms - b.timestamp_ms);

  const running = pipeline.status === "running";
  const done = pipeline.status === "done";
  const reduction =
    vw !== null && aw !== null && vw > 0 && aw < vw
      ? Math.max(1, Math.round((1 - aw / vw) * 100))
      : null;

  const PATIENTS = [
    { id: "FC02", label: "healthy control" },
    { id: "F03", label: "mild dysarthria" },
    { id: "M01", label: "severe dysarthria" },
  ] as const;

  return (
    <div className="h-full flex flex-col px-10 py-6 max-w-7xl mx-auto w-full overflow-y-auto">
      {/* Top row: eyebrow + patient chips + Run button */}
      <header className="mb-5 flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="text-emerald-400 font-mono text-xs uppercase tracking-[0.25em] mb-1">
            Live demo
          </div>
          <h1 className="text-3xl font-semibold tracking-tight">
            Same audio, two transcribers.
          </h1>
          <p className="text-sm text-neutral-400 mt-1 max-w-2xl">
            Pick a patient. Phonemer and a stock speech model both listen to
            the same clip. The difference is what you should remember.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {PATIENTS.map((p) => {
            const active = picked === p.id;
            return (
              <button
                key={p.id}
                onClick={() => setPicked(p.id)}
                className={`text-left rounded-md border px-3 py-1.5 text-xs transition-colors ${
                  active
                    ? "border-emerald-500 bg-emerald-950/30 text-emerald-200"
                    : "border-neutral-700 bg-neutral-900/40 text-neutral-300 hover:border-neutral-600"
                }`}
              >
                <span className="font-mono font-semibold">{p.id}</span>
                <span className="opacity-60 ml-1.5">· {p.label}</span>
              </button>
            );
          })}
          <button
            onClick={runPipeline}
            disabled={!selectedClip || running}
            className="ml-1 rounded-md bg-emerald-600 px-4 py-1.5 text-xs font-medium hover:bg-emerald-500 disabled:opacity-40"
          >
            {running ? "Running…" : done ? "Re-run" : "Run pipeline"}
          </button>
          {eligibleClips.length > 1 && (
            <button
              onClick={() => {
                // The auto-run cache holds clip ids we've already kicked
                // off the pipeline for. If the user cycles back to a
                // seen example it would skip the run and leave the cards
                // showing stale data vs the new ground-truth text. We
                // clear the cache on every cycle so each step runs fresh.
                _demoAutoRunClipIds.clear();
                setExampleIndex((i) => i + 1);
              }}
              disabled={running}
              title={`Cycle through ${eligibleClips.length} demo clips for ${picked}`}
              className="rounded-md border border-neutral-700 bg-neutral-900/40 px-3 py-1.5 text-xs text-neutral-300 hover:border-neutral-600 disabled:opacity-40"
            >
              Try another clip →
            </button>
          )}
        </div>
      </header>

      {/* Ground-truth card — the question both models are trying to answer */}
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/40 px-5 py-4 mb-5 flex items-center gap-5">
        <div className="flex-1 min-w-0">
          <div className="text-[10px] uppercase tracking-[0.25em] font-mono text-neutral-500 mb-1 flex items-center gap-2">
            <span>patient says (ground truth)</span>
            {eligibleClips.length > 1 && (
              <span className="text-neutral-600 normal-case tracking-normal">
                · example{" "}
                <span className="tabular-nums text-neutral-400">
                  {(exampleIndex % eligibleClips.length) + 1}
                </span>{" "}
                of {eligibleClips.length}
              </span>
            )}
          </div>
          <div className="text-2xl text-neutral-100 italic leading-snug">
            “{selectedClip?.intended_text ?? "—"}”
          </div>
        </div>
        <div className="shrink-0">
          <PlayButton audio={audio} isPlaying={isPlaying} hint="Play this clip" />
        </div>
      </div>

      {/* The two-pipeline comparison — the actual message */}
      <div className="grid md:grid-cols-2 gap-4 mb-5">
        <ComparisonCard
          title="Off-the-shelf speech model"
          subtitle="Stock Whisper. Trained on healthy speech."
          tone="vanilla"
          running={running && pipeline.vanillaText === null}
          wer={vw}
          stamp={vanillaStamp(vw)}
          flagged={false}
        >
          {pipeline.vanillaText !== null ? (
            <DiffText
              hypothesis={pipeline.vanillaText}
              reference={ref}
              size="lg"
            />
          ) : pipeline.vanillaError ? (
            <span className="text-red-400 text-sm">
              error: {pipeline.vanillaError}
            </span>
          ) : null}
        </ComparisonCard>
        <ComparisonCard
          title="Phonemer"
          subtitle={
            adapterPath
              ? "Per-patient adapter + phoneme safety gate."
              : "(no adapter loaded — vanilla path)"
          }
          tone="phonemer"
          running={running && !adapted}
          wer={aw}
          stamp={phonemerStamp(aw, anyFlagged)}
          flagged={anyFlagged}
        >
          {adapted ? (
            <DiffText hypothesis={adapted} reference={ref} size="lg" />
          ) : null}
        </ComparisonCard>
      </div>

      {/* The takeaway — only render when we can compute a meaningful delta. */}
      {done && reduction !== null && (
        <div className="rounded-2xl border border-emerald-700/40 bg-emerald-950/20 px-6 py-4 mb-4 flex items-center justify-center gap-6">
          <div className="text-center">
            <div className="text-[10px] uppercase tracking-[0.25em] font-mono text-emerald-400 mb-1">
              the difference
            </div>
            <div className="flex items-baseline justify-center gap-3">
              <span className="font-mono tabular-nums text-5xl font-semibold text-emerald-300">
                −{reduction}%
              </span>
              <span className="text-sm text-neutral-400">
                word errors vs the stock model
              </span>
            </div>
          </div>
        </div>
      )}
      {done && reduction === null && anyFlagged && (
        <div className="rounded-2xl border border-rose-700/40 bg-rose-950/20 px-6 py-4 mb-4 text-sm text-rose-200">
          <span className="font-semibold">Phonemer flagged this clip.</span>{" "}
          The phoneme alignment doesn&apos;t support the words; instead of
          confidently outputting them, the system tells the listener to verify
          — that&apos;s the difference. A stock model has no such guardrail.
        </div>
      )}

      {/* Optional proof points for the technically curious. Collapsed by
          default so they don't compete with the message above. */}
      {(pipeline.waveform || allFrames.length > 0) && (
        <details className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-3">
          <summary className="cursor-pointer text-[10px] font-mono uppercase tracking-[0.25em] text-neutral-500 hover:text-neutral-300 list-none">
            under the hood · waveform + VAD + phoneme stream
          </summary>
          <div className="mt-3 space-y-3">
            {pipeline.waveform && pipeline.audioDurationMs && (
              <VadTimeline
                durationMs={pipeline.audioDurationMs}
                regions={pipeline.utterances.map((u) => ({
                  start: u.start_ms,
                  end: u.end_ms,
                }))}
                waveform={pipeline.waveform}
                tall
                playheadMs={playheadMs}
                showPlayhead
              />
            )}
            {allFrames.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
                  wav2vec2 phoneme stream (50 Hz)
                </div>
                <PhonemeStrip
                  frames={allFrames}
                  tall
                  playheadMs={playheadMs}
                  showPlayhead
                />
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}

// One-line qualitative stamp that summarizes how the stock model did
// on this clip. We can't say "vanilla wins" honestly — even when its
// WER is low, the point of the demo is that there's no safety net,
// so we frame the stamp around its *behavior* not its score.
function vanillaStamp(vw: number | null): {
  label: string;
  tone: "warn" | "neutral" | "good";
} | null {
  if (vw === null) return null;
  if (vw === 0)
    return { label: "matched ground truth · no safety net either way", tone: "good" };
  if (vw < 0.2)
    return { label: "mostly right · no warning when it slips", tone: "neutral" };
  if (vw < 0.5)
    return { label: "wrong — but confidently so", tone: "warn" };
  return { label: "confidently nonsense", tone: "warn" };
}

function phonemerStamp(
  aw: number | null,
  flagged: boolean,
): { label: string; tone: "warn" | "neutral" | "good" } | null {
  if (flagged)
    return {
      label: "flagged — phoneme alignment doesn't support these words",
      tone: "warn",
    };
  if (aw === null) return null;
  if (aw === 0) return { label: "matched ground truth", tone: "good" };
  if (aw < 0.2) return { label: "mostly right · gate let it through", tone: "good" };
  if (aw < 0.5) return { label: "imperfect · gate didn't catch it", tone: "neutral" };
  return { label: "wrong · gate should have flagged it", tone: "warn" };
}

// The comparison cards are styled so the two columns aren't equal: the
// stock model is framed in muted neutral (look at me cautiously), and
// Phonemer is framed emerald when transcription was clean or amber
// when the gate flagged. Both use DiffText against the same reference,
// so wrong words light up identically.
function ComparisonCard({
  title,
  subtitle,
  tone,
  running,
  wer,
  stamp,
  flagged,
  children,
}: {
  title: string;
  subtitle: string;
  tone: "vanilla" | "phonemer";
  running: boolean;
  wer: number | null;
  stamp: { label: string; tone: "warn" | "neutral" | "good" } | null;
  flagged: boolean;
  children: React.ReactNode;
}) {
  const borderCls =
    tone === "vanilla"
      ? "border-neutral-700 bg-neutral-900/40"
      : flagged
        ? "border-2 border-rose-600/60 bg-rose-950/15"
        : "border-2 border-emerald-600/60 bg-emerald-950/15";
  const stampCls =
    stamp?.tone === "warn"
      ? "text-rose-300"
      : stamp?.tone === "good"
        ? "text-emerald-300"
        : "text-neutral-400";

  return (
    <div className={`rounded-2xl p-4 ${borderCls}`}>
      <div className="flex items-baseline justify-between mb-2">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-neutral-100">{title}</div>
          <div className="text-[11px] text-neutral-500">{subtitle}</div>
        </div>
        {wer !== null && (
          <div className="text-right shrink-0">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              WER
            </div>
            <div className="font-mono text-base text-neutral-100 tabular-nums">
              {(wer * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
      <div className="min-h-[3rem]">
        {running ? (
          <div className="text-sm text-neutral-500 italic">
            transcribing…
          </div>
        ) : (
          children
        )}
      </div>
      {stamp && !running && (
        <div className={`mt-3 text-xs ${stampCls}`}>{stamp.label}</div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────── (legacy detail slides below)

function IntroSlide({ selectedClip, pipeline, adapterPath }: SlideProps) {
  return (
    <SlideShell
      number={null}
      title="How Phonemer transcribes patient speech"
      description={[
        "An audio clip from a brain-injury patient flows through 7 stages. The same code path runs in production; this presentation walks through each stage with a live visualization.",
        "Use the picker above to choose a TORGO clip, click Run pipeline, then step through the slides. Use ← / → or click the dots below to navigate.",
      ]}
    >
      <div className="grid md:grid-cols-2 gap-8 h-full content-center">
        <div className="space-y-3">
          <h2 className="text-base font-medium text-neutral-300">
            Pipeline architecture
          </h2>
          <ol className="space-y-2.5">
            {[
              { n: 1, t: "Audio input", s: "raw 16 kHz mono PCM" },
              { n: 2, t: "Voice Activity Detection", s: "Silero VAD splits speech from silence" },
              { n: 3, t: "Speaker identification", s: "ECAPA-TDNN embeddings → patient or listener" },
              { n: 4, t: "Phoneme extraction", s: "wav2vec2 posteriors at 50 Hz" },
              { n: 5, t: "Whisper + LoRA adapter", s: "patient-personalized transcription" },
              { n: 6, t: "Safety gate", s: "phoneme alignment flags hallucinations" },
              { n: 7, t: "Surfaced to listener", s: "with confidence indicator" },
            ].map(({ n, t, s }) => (
              <li
                key={n}
                className="flex items-start gap-3 rounded border border-neutral-800 bg-neutral-900/40 px-3 py-2"
              >
                <span className="text-emerald-500 font-mono text-xs mt-0.5">
                  {String(n).padStart(2, "0")}
                </span>
                <div>
                  <div className="text-sm font-medium">{t}</div>
                  <div className="text-xs text-neutral-500">{s}</div>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="space-y-4">
          <h2 className="text-base font-medium text-neutral-300">
            Today&apos;s example
          </h2>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 space-y-3">
            {selectedClip ? (
              <>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <Pill label="speaker" value={selectedClip.speaker_id} />
                  <Pill label="condition" value={selectedClip.condition} />
                  <Pill label="prompt set" value={selectedClip.prompt_set ?? "?"} />
                  <Pill
                    label="duration"
                    value={`${(selectedClip.duration_ms / 1000).toFixed(2)} s`}
                  />
                </div>
                {selectedClip.intended_text && (
                  <div className="rounded bg-neutral-950/50 border border-neutral-800 px-3 py-2 text-sm">
                    <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-0.5">
                      intended text
                    </div>
                    “{selectedClip.intended_text}”
                  </div>
                )}
                <audio
                  src={`${SERVER_HTTP}/torgo/audio/${selectedClip.id}`}
                  controls
                  preload="metadata"
                  className="w-full h-9"
                />
              </>
            ) : (
              <div className="text-sm text-neutral-500">
                Loading TORGO clips…
              </div>
            )}
          </div>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3 text-xs space-y-1">
            <div className="text-neutral-400 uppercase tracking-wide text-[10px]">
              system status
            </div>
            <div className="font-mono text-[11px] text-neutral-500">
              backend{" "}
              <span className="text-neutral-300">{SERVER_HTTP}</span>
            </div>
            <div className="font-mono text-[11px] text-neutral-500">
              patient adapter{" "}
              {adapterPath ? (
                <span className="text-emerald-400">{adapterPath}</span>
              ) : (
                <span className="text-amber-400">none — vanilla path</span>
              )}
            </div>
            <div className="font-mono text-[11px] text-neutral-500">
              pipeline status{" "}
              <span
                className={
                  pipeline.status === "done"
                    ? "text-emerald-400"
                    : pipeline.status === "running"
                      ? "text-amber-400"
                      : "text-neutral-400"
                }
              >
                {pipeline.status}
              </span>
            </div>
          </div>
        </div>
      </div>
    </SlideShell>
  );
}

function AudioSlide({
  selectedClip,
  pipeline,
  audio,
  isPlaying,
  playheadMs,
}: SlideProps) {
  // Auto-play on slide entry once the waveform is ready.
  useAudioAutoplay(audio, !!pipeline.waveform);
  return (
    <SlideShell
      number={1}
      title="Audio input"
      codeRef="server/app/main.py"
      description={[
        "The clip enters as 16 kHz mono PCM — the same format Whisper was trained on. In a live session this comes from the browser's microphone; here it comes from a TORGO recording on disk.",
        "No transformation at this stage. The waveform below is exactly what every downstream model will see. Press play to watch the audio scrub through.",
      ]}
    >
      {pipeline.waveform && pipeline.audioDurationMs ? (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <PlayButton audio={audio} isPlaying={isPlaying} />
            <div className="font-mono text-xs text-neutral-500">
              t = {(playheadMs / 1000).toFixed(2)} /{" "}
              {(pipeline.audioDurationMs / 1000).toFixed(2)} s
            </div>
          </div>
          <Waveform
            samples={pipeline.waveform}
            durationMs={pipeline.audioDurationMs}
            tall
            playheadMs={playheadMs}
            showPlayhead
          />
          <div className="grid grid-cols-4 gap-3">
            <BigStat
              label="duration"
              value={`${((pipeline.audioDurationMs ?? 0) / 1000).toFixed(2)}`}
              unit="s"
            />
            <BigStat label="sample rate" value="16,000" unit="Hz" />
            <BigStat label="channels" value="1" unit="mono" />
            <BigStat label="format" value="float32" />
          </div>
          <div className="text-xs text-neutral-500 max-w-3xl">
            <strong className="text-neutral-300">What you see:</strong> peaks in the
            waveform are voiced phonemes (vowels and voiced consonants). The
            silent stretches at the start and end are just room noise the
            patient is breathing through; they&apos;re not part of the spoken
            phrase.
          </div>
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function VadSlide({
  pipeline,
  selectedClip,
  audio,
  isPlaying,
  playheadMs,
}: SlideProps) {
  const us = pipeline.utterances;
  // Auto-play on slide entry once at least one utterance has been detected.
  useAudioAutoplay(audio, us.length > 0);
  const totalSpeechMs = us.reduce(
    (acc, u) => acc + (u.end_ms - u.start_ms),
    0,
  );
  // Which utterance is the playhead currently inside, if any?
  const inUttIdx = us.findIndex(
    (u) => playheadMs >= u.start_ms && playheadMs <= u.end_ms,
  );
  return (
    <SlideShell
      number={2}
      title="Voice Activity Detection"
      codeRef="server/app/silero_vad.py + vad.py"
      description={[
        "A small neural network (Silero VAD) scores every 32 ms of audio for the probability it contains speech vs silence/breath/noise.",
        "Speech regions are merged into utterances using a 3000 ms silence-close threshold — long enough to bridge the multi-second intra-sentence pauses severe dysarthric speakers take (M01 reading a TIMIT sentence routinely produces 2–3 s pauses between clauses), short enough to still separate distinct speaker turns in a clinical conversation (those typically have >3 s gaps because the listener naturally waits for slow speech). The segmenter also enforces a 400 ms minimum and 15 s maximum utterance length. Tunable per session via `PHONEMER_SILENCE_CLOSE_MS`.",
      ]}
    >
      {us.length > 0 && pipeline.audioDurationMs && pipeline.waveform ? (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <PlayButton audio={audio} isPlaying={isPlaying} />
            <div className="font-mono text-xs text-neutral-500">
              {inUttIdx >= 0 ? (
                <span className="text-emerald-300">
                  ● in utterance {inUttIdx + 1} of {us.length}
                </span>
              ) : (
                <span>○ in silence between utterances</span>
              )}
            </div>
          </div>
          <VadTimeline
            durationMs={pipeline.audioDurationMs}
            regions={us.map((u) => ({ start: u.start_ms, end: u.end_ms }))}
            waveform={pipeline.waveform}
            tall
            playheadMs={playheadMs}
            showPlayhead
          />
          <div className="grid grid-cols-4 gap-3">
            <BigStat
              label="utterances detected"
              value={`${us.length}`}
              unit={us.length === 1 ? "segment" : "segments"}
            />
            <BigStat
              label="total speech"
              value={`${(totalSpeechMs / 1000).toFixed(2)}`}
              unit="s"
            />
            <BigStat
              label="silence trimmed"
              value={`${(
                (pipeline.audioDurationMs - totalSpeechMs) /
                1000
              ).toFixed(2)}`}
              unit="s"
            />
            <BigStat
              label="speech ratio"
              value={`${Math.round(
                (totalSpeechMs / pipeline.audioDurationMs) * 100,
              )}`}
              unit="%"
            />
          </div>
          {us.length > 1 && (
            <div className="rounded-lg border border-amber-700/40 bg-amber-950/15 px-4 py-3 text-xs">
              <div className="text-amber-300 font-semibold mb-1">
                ▲ This clip was split into {us.length} utterances.
              </div>
              <div className="text-neutral-400">
                The patient paused between phrases for long enough that VAD
                treated them as separate speech events. Each utterance below
                gets its own phoneme extraction and transcription downstream;
                the final transcript is the concatenation in order.
              </div>
              <div className="mt-2 space-y-0.5 font-mono text-[11px] text-neutral-500">
                {us.map((u, i) => (
                  <div key={u.utterance_id}>
                    {i + 1}. {u.start_ms}–{u.end_ms} ms (
                    {((u.end_ms - u.start_ms) / 1000).toFixed(2)}s)
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="text-xs text-neutral-500 max-w-3xl">
            <strong className="text-neutral-300">Why this matters:</strong> Whisper
            tends to hallucinate confident-sounding text from silence
            (&quot;Thank you.&quot; or &quot;Bye.&quot; are notorious). Trimming
            non-speech regions prevents this whole class of failure at the
            cheapest possible layer.
          </div>
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function SpeakerSlide({ pipeline, selectedClip }: SlideProps) {
  // Speaker classification is per-utterance but in practice all utterances
  // from one clip get the same label. Show the first one.
  const u = pipeline.utterances[0] ?? null;
  return (
    <SlideShell
      number={3}
      title="Speaker identification"
      codeRef="server/app/speaker_id.py"
      description={[
        "A 192-dimensional voice embedding from SpeechBrain's ECAPA-TDNN model is computed for the utterance and compared (cosine similarity) against the patient's enrolled embedding.",
        "Above the match threshold (0.55 by default) → patient pipeline (with the per-patient adapter, if loaded). Below it → listener pipeline (plain Whisper, no adapter). On this page we use force_patient=1 so every clip routes through the patient path.",
      ]}
    >
      {u ? (
        <div className="space-y-6">
          <SimilarityMeter
            value={u.speaker_similarity}
            matchThreshold={0.55}
            rejectThreshold={0.15}
            label={u.speaker}
            isPatient={u.is_patient}
          />
          <div className="grid md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-2">
                Routing decision
              </div>
              <div
                className={`text-2xl font-semibold ${
                  u.is_patient ? "text-emerald-400" : "text-neutral-300"
                }`}
              >
                {u.is_patient ? "→ PATIENT pipeline" : "→ LISTENER pipeline"}
              </div>
              <div className="text-xs text-neutral-500 mt-2">
                {u.is_patient
                  ? "uses Whisper + per-patient LoRA adapter, then phoneme-alignment safety gate"
                  : "uses plain faster-whisper directly (no personalization)"}
              </div>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-2">
                Why two pipelines?
              </div>
              <div className="text-sm text-neutral-300 space-y-2">
                <p>
                  The listener (caregiver, doctor, family) speaks normally —
                  faster-whisper handles their speech well at low latency.
                </p>
                <p>
                  The patient&apos;s speech is impaired in patient-specific
                  ways. Their pipeline runs through the LoRA adapter trained
                  on their calibration data.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function PhonemesSlide({
  pipeline,
  selectedClip,
  audio,
  isPlaying,
  playheadMs,
}: SlideProps) {
  // Concatenate phoneme frames from all utterances in time order so the
  // strip displays the full clip's phoneme stream, not just the latest
  // utterance's. Each frame already has an absolute timestamp_ms (offset
  // by the utterance's start_ms in the server), so flat-merge is correct.
  const allFrames = useMemo(() => {
    const flat: PhonemeWindowMessage["frames"] = [];
    for (const u of pipeline.utterances) {
      const frames = pipeline.phonemeFramesByUtt[u.utterance_id] ?? [];
      flat.push(...frames);
    }
    return flat.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  }, [pipeline.utterances, pipeline.phonemeFramesByUtt]);
  useAudioAutoplay(audio, allFrames.length > 0);
  return (
    <SlideShell
      number={4}
      title="Phoneme extraction (wav2vec2)"
      codeRef="server/app/phonemes.py"
      description={[
        "The audio is fed to facebook/wav2vec2-lv-60-espeak-cv-ft, a 300M-parameter model trained to label every 20 ms frame with its most likely phoneme (espeak inventory, ~50 distinct symbols).",
        "These per-frame guesses don't drive transcription directly — Whisper handles that. They're used (a) for the safety gate in stage 6 to detect Whisper hallucinations, and (b) as the teacher signal in the patient distortion calibration that produces each adapter.",
      ]}
    >
      {allFrames.length > 0 ? (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <PlayButton
              audio={audio}
              isPlaying={isPlaying}
              hint="Press play to watch each phoneme light up as it's heard"
            />
            <div className="font-mono text-xs text-neutral-500">
              {allFrames.length} frames · 50 Hz · {pipeline.utterances.length}{" "}
              utterance{pipeline.utterances.length === 1 ? "" : "s"}
            </div>
          </div>
          <PhonemeStrip
            frames={allFrames}
            tall
            playheadMs={playheadMs}
            showPlayhead
          />
          <div className="grid grid-cols-4 gap-3">
            <BigStat
              label="frames"
              value={`${allFrames.length}`}
              unit="@ 20 ms"
            />
            <BigStat label="frame rate" value="50" unit="Hz" />
            <BigStat
              label="phoneme inventory"
              value="393"
              unit="tokens"
            />
            <BigStat
              label="speech frames"
              value={`${allFrames.filter((f) => f.phonemes[0]?.symbol !== "<pad>").length}`}
              unit="of total"
            />
          </div>
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function WhisperSlide({ pipeline, selectedClip, adapterPath }: SlideProps) {
  const ref = selectedClip?.intended_text ?? null;
  // Concatenate every utterance's translation in time order so the user
  // sees the full transcript, not just the last utterance to arrive.
  const adaptedFull = pipeline.translations
    .slice()
    .sort((a, b) => a.utterance_id - b.utterance_id)
    .map((t) => t.primary)
    .filter(Boolean)
    .join(" ");
  const vw = ref ? wer(ref, pipeline.vanillaText) : null;
  const aw = ref ? wer(ref, adaptedFull || null) : null;
  const winner =
    vw !== null && aw !== null ? (aw < vw ? "adapted" : aw > vw ? "vanilla" : "tie") : null;

  return (
    <SlideShell
      number={5}
      title="Whisper transcription — vanilla vs patient adapter"
      codeRef="server/app/stt.py"
      description={[
        "The same audio is passed to two Whisper models: plain Whisper (vanilla) and Whisper with the patient's LoRA adapter loaded.",
        "Words highlighted in green match the reference; amber words don't. The WER score below quantifies the difference. The adapter is calibrated on this patient's calibration data — its job is to learn how this specific patient distorts the language.",
      ]}
    >
      {pipeline.status === "done" || pipeline.translations.length > 0 ? (
        <div className="space-y-5">
          {ref && (
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">
                Reference (what the patient was asked to say)
              </div>
              <div className="text-base">“{ref}”</div>
            </div>
          )}
          <div className="grid md:grid-cols-2 gap-4">
            <BranchCard
              title="Vanilla Whisper"
              subtitle="no adapter — baseline"
              winner={winner === "vanilla"}
              wer={vw}
              error={pipeline.vanillaError}
            >
              {pipeline.vanillaText !== null && (
                <DiffText
                  hypothesis={pipeline.vanillaText}
                  reference={ref}
                  size="lg"
                />
              )}
            </BranchCard>
            <BranchCard
              title="Patient-adapted Whisper"
              subtitle={
                adapterPath
                  ? `LoRA adapter at ${adapterPath}${
                      pipeline.translations.length > 1
                        ? ` · ${pipeline.translations.length} utterances joined`
                        : ""
                    }`
                  : "no adapter loaded — falls back to vanilla"
              }
              winner={winner === "adapted"}
              wer={aw}
              error={null}
            >
              {adaptedFull ? (
                <DiffText
                  hypothesis={adaptedFull}
                  reference={ref}
                  size="lg"
                />
              ) : null}
            </BranchCard>
          </div>
          {winner && winner !== "tie" && vw !== null && aw !== null && (
            <div className="text-sm text-neutral-400">
              {winner === "adapted" ? (
                <>
                  Adapter wins by{" "}
                  <span className="text-emerald-400 font-semibold">
                    {((vw - aw) * 100).toFixed(0)} percentage points
                  </span>{" "}
                  on this clip.
                </>
              ) : (
                <>
                  Vanilla wins on this clip. (Adapters average ≥30% relative
                  WER reduction on TORGO dysarthric speakers — individual
                  clips vary, especially short isolated-word stimuli.)
                </>
              )}
            </div>
          )}
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function GateSlide({ pipeline, selectedClip }: SlideProps) {
  const ts = pipeline.translations;
  // For the meter, show the worst (most-likely-flagged) utterance — that's
  // what determines whether the listener UI surfaces a warning.
  const t = ts.length > 0
    ? ts.reduce((worst, cur) => {
        const ws = worst.align_score ?? 0;
        const cs = cur.align_score ?? 0;
        return cs < ws ? cur : worst;
      })
    : null;
  return (
    <SlideShell
      number={6}
      title="Phoneme-alignment safety gate"
      codeRef="server/app/stt.py · GatedTranscriber"
      description={[
        "Whisper emits 3 sampled candidates; each is phonemized via espeak and force-aligned against the wav2vec2 distribution from stage 4. We pick the candidate that fits the audio best, not the one Whisper's LM head liked most — that's the structural fix for fluent hallucinations like 'I need a little wired lay Diego mall' on 'A roll of wire lay near the wall'.",
        "Four signals can flag a clip: alignment score (mean log-prob, threshold −0.85; too low = audio doesn't fit), phoneme density too high (>25 ph/s = hallucinated extras), phoneme density too low (<4 ph/s on clips ≥ 2 s, measured over VOICED duration via energy-based pause detection in `features._pause_stats` so heavy-pause fluent speech doesn't false-flag), and overall confidence (exp of align score) below 55% — that last one catches cases where the gate would otherwise pass but the per-word confidences are uniformly mediocre. If any trips, the output is flagged 'low confidence' but still surfaced — hiding it would defeat the medical-safety story.",
      ]}
    >
      {t ? (
        <div className="space-y-6">
          <ThresholdMeter
            label="Alignment log-probability"
            value={t.align_score ?? null}
            min={-8}
            max={0}
            threshold={-0.85}
            thresholdSide="below"
            suffix=""
            helpAbove="audio supports the transcription"
            helpBelow="audio doesn't support → likely hallucination"
            tall
          />
          <ThresholdMeter
            label="Phoneme density"
            value={t.phoneme_density ?? null}
            min={0}
            max={35}
            threshold={25}
            thresholdSide="above"
            suffix=" ph/s"
            helpAbove="too many phonemes per second → hallucinated extra words"
            helpBelow="natural speech: 10–18 ph/s (voiced-duration) · under-production threshold: 4 ph/s on clips ≥ 2 s"
            tall
          />
          <div className="grid md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-2">
                Verdict
              </div>
              <div
                className={`text-2xl font-semibold ${
                  t.flagged ? "text-red-400" : "text-emerald-400"
                }`}
              >
                {t.flagged ? `FLAGGED · ${t.reason ?? "low_alignment"}` : "OK"}
              </div>
              <div className="text-xs text-neutral-500 mt-2">
                {t.flagged
                  ? "Will render with a red border and 'low confidence' hint in the listener UI."
                  : "Will render as a confident transcription in the listener UI."}
              </div>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-2">
                Why surface flagged text at all?
              </div>
              <div className="text-sm text-neutral-300 leading-relaxed">
                In a medical setting, a confidently-wrong transcription is
                worse than an honestly uncertain one. The gate&apos;s job is
                not to hide; it&apos;s to{" "}
                <span className="text-emerald-300">label</span> uncertainty so
                the doctor or family member knows when to ask for a repeat.
              </div>
            </div>
          </div>
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

function OutputSlide({ pipeline, selectedClip }: SlideProps) {
  const ts = pipeline.translations
    .slice()
    .sort((a, b) => a.utterance_id - b.utterance_id);
  // Combined transcript across utterances; flagged if any utterance is
  // flagged (worst-case is what the listener sees).
  const combined = ts.map((t) => t.primary).filter(Boolean).join(" ");
  const anyFlagged = ts.some((t) => t.flagged);
  const elapsed =
    pipeline.startedAt && pipeline.finishedAt
      ? ((pipeline.finishedAt - pipeline.startedAt) / 1000).toFixed(2)
      : null;
  return (
    <SlideShell
      number={7}
      title="Surfaced to the listener"
      description={[
        "What the doctor / caregiver / family member actually sees in the listener UI. The text is always shown, but flagged outputs are visually marked as low-confidence so the reader knows to verify or ask for a repeat.",
        ts.length > 1
          ? `This clip produced ${ts.length} utterances; the listener sees them as separate but consecutive transcripts. Below they're shown joined for clarity.`
          : "",
      ].filter(Boolean)}
    >
      {ts.length > 0 ? (
        <div className="space-y-6">
          <div
            className={`rounded-2xl p-8 ${
              anyFlagged
                ? "border-2 border-red-700/60 bg-red-950/20"
                : "border border-emerald-700/40 bg-emerald-950/10"
            }`}
          >
            <div
              className={`text-[10px] uppercase tracking-wide font-mono mb-3 ${
                anyFlagged ? "text-red-400" : "text-emerald-400"
              }`}
            >
              {anyFlagged
                ? "low confidence — verify with the patient"
                : "transcribed"}
            </div>
            <div className="text-3xl font-medium leading-tight">
              {combined || (
                <span className="text-neutral-500 italic">
                  (empty transcription)
                </span>
              )}
            </div>
          </div>
          {ts.length > 1 && (
            <div className="space-y-2">
              <div className="text-xs text-neutral-500 uppercase tracking-wide">
                Per-utterance breakdown
              </div>
              {ts.map((t, i) => (
                <div
                  key={t.utterance_id}
                  className={`rounded border px-3 py-2 text-sm ${
                    t.flagged
                      ? "border-red-700/40 bg-red-950/15"
                      : "border-neutral-800 bg-neutral-900/40"
                  }`}
                >
                  <span className="font-mono text-[10px] text-neutral-500 mr-2">
                    {i + 1}.
                  </span>
                  {t.primary || <span className="italic text-neutral-500">(empty)</span>}
                  {t.flagged && (
                    <span className="ml-2 text-[10px] uppercase text-red-400 font-mono">
                      flagged
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
          <div className="grid grid-cols-3 gap-3">
            <BigStat
              label="utterances"
              value={`${ts.length}`}
              unit={ts.length === 1 ? "segment" : "segments"}
            />
            <BigStat
              label="any flagged"
              value={anyFlagged ? "yes" : "no"}
            />
            <BigStat
              label="pipeline time"
              value={elapsed ?? "—"}
              unit="s"
            />
          </div>
          {selectedClip?.intended_text && (
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
              <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">
                Reference
              </div>
              <div className="text-sm text-neutral-300">
                “{selectedClip.intended_text}”
              </div>
            </div>
          )}
        </div>
      ) : (
        <RunPrompt status={pipeline.status} clip={selectedClip} />
      )}
    </SlideShell>
  );
}

// ────────────────────────────────────────────────────────────── primitive UI

function Pill({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
}) {
  const toneCls =
    tone === "good"
      ? "border-emerald-700/40 bg-emerald-900/20"
      : tone === "warn"
        ? "border-amber-700/40 bg-amber-900/20"
        : "border-neutral-800 bg-neutral-900/40";
  return (
    <div className={`rounded border ${toneCls} px-3 py-2`}>
      <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">
        {label}
      </div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  );
}

function BigStat({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">
        {label}
      </div>
      <div className="font-mono">
        <span className="text-2xl text-neutral-100">{value}</span>
        {unit && (
          <span className="text-xs text-neutral-500 ml-1.5">{unit}</span>
        )}
      </div>
    </div>
  );
}

function RunPrompt({
  status,
  clip,
}: {
  status: PipelineState["status"];
  clip: TorgoClip | null;
}) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center space-y-2">
        <div className="text-neutral-500 text-sm">
          {status === "running"
            ? "Processing the clip…"
            : status === "error"
              ? "An error occurred — see top right."
              : !clip
                ? "Loading clips…"
                : "Click Run pipeline at the top to populate this slide."}
        </div>
        {clip && status !== "running" && (
          <div className="text-xs text-neutral-600">
            {clip.speaker_id} · {clip.prompt_set ?? "?"} ·{" "}
            {(clip.duration_ms / 1000).toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  );
}

function BranchCard({
  title,
  subtitle,
  winner,
  wer,
  error,
  children,
}: {
  title: string;
  subtitle: string;
  winner: boolean;
  wer: number | null;
  error: string | null;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-lg p-4 ${
        winner
          ? "border-2 border-emerald-600/60 bg-emerald-950/15"
          : "border border-neutral-800 bg-neutral-900/40"
      }`}
    >
      <div className="flex items-baseline justify-between mb-2">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          <div className="text-[11px] font-mono text-neutral-500">
            {subtitle}
          </div>
        </div>
        {wer !== null && (
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-wide text-neutral-500">
              WER
            </div>
            <div className="font-mono text-base">
              {(wer * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
      {error ? (
        <div className="text-sm text-red-400">error: {error}</div>
      ) : (
        children
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────── visualizations

function Waveform({
  samples,
  durationMs,
  tall = false,
  playheadMs = 0,
  showPlayhead = false,
}: {
  samples: Float32Array;
  durationMs: number;
  tall?: boolean;
  playheadMs?: number;
  showPlayhead?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const height = tall ? 200 : 80;
  // Render the static waveform once per (samples, height) — playhead is a
  // separate DOM overlay so we don't have to redraw 1000s of bars per frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    canvas.width = cssWidth * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, cssWidth, height);
    const bars = cssWidth;
    const samplesPerBar = Math.max(1, Math.floor(samples.length / bars));
    const mid = height / 2;
    ctx.fillStyle = "#10b981";
    for (let x = 0; x < bars; x++) {
      const start = x * samplesPerBar;
      const end = Math.min(samples.length, start + samplesPerBar);
      let max = 0;
      for (let i = start; i < end; i++) {
        const a = Math.abs(samples[i]);
        if (a > max) max = a;
      }
      const h = Math.max(1, Math.min(mid - 1, max * mid * 1.3));
      ctx.fillRect(x, mid - h, 1, h * 2);
    }
  }, [samples, height]);

  const tickStepMs = durationMs > 5000 ? 1000 : 500;
  const ticks: number[] = [];
  for (let t = 0; t <= durationMs; t += tickStepMs) ticks.push(t);

  const playheadPct =
    durationMs > 0 ? Math.max(0, Math.min(1, playheadMs / durationMs)) * 100 : 0;

  return (
    <div className="space-y-1">
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full rounded bg-neutral-950/80 block"
          style={{ width: "100%", height }}
        />
        {showPlayhead && (
          <>
            {/* Soft trail behind the playhead — fades the past with white,
                makes the moving line read as "where we are now". */}
            <div
              className="absolute top-0 h-full bg-emerald-400/10 pointer-events-none"
              style={{ left: 0, width: `${playheadPct}%` }}
            />
            <div
              className="absolute top-0 h-full w-0.5 bg-white/90 shadow-[0_0_8px_rgba(255,255,255,0.6)] pointer-events-none transition-[left] duration-75 ease-linear"
              style={{ left: `${playheadPct}%` }}
            />
          </>
        )}
      </div>
      <div className="relative h-4">
        {ticks.map((t) => (
          <div
            key={t}
            className="absolute top-0 text-[10px] text-neutral-500 font-mono"
            style={{
              left: `${(t / durationMs) * 100}%`,
              transform: "translateX(-50%)",
            }}
          >
            <div className="border-l border-neutral-700 h-1 mx-auto" />
            <div className="mt-0.5">{(t / 1000).toFixed(1)}s</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Play / pause button that drives the shared audio element. When playing,
// the playhead state updates via rAF and the visualizations animate.
function PlayButton({
  audio,
  isPlaying,
  hint = "Play with the pipeline visualization",
}: {
  audio: HTMLAudioElement | null;
  isPlaying: boolean;
  hint?: string;
}) {
  const onClick = () => {
    if (!audio) return;
    if (audio.paused) {
      // If at end, rewind to start so user can replay.
      if (audio.currentTime >= (audio.duration || 0) - 0.01) {
        audio.currentTime = 0;
      }
      void audio.play();
    } else {
      audio.pause();
    }
  };
  return (
    <button
      onClick={onClick}
      disabled={!audio}
      className="inline-flex items-center gap-2 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium hover:bg-emerald-500 disabled:opacity-40"
      title={hint}
    >
      <span aria-hidden>{isPlaying ? "❚❚" : "▶"}</span>
      <span>{isPlaying ? "Pause" : "Play audio"}</span>
    </button>
  );
}

function VadTimeline({
  durationMs,
  regions,
  waveform,
  tall = false,
  playheadMs = 0,
  showPlayhead = false,
}: {
  durationMs: number;
  regions: { start: number; end: number }[];
  waveform: Float32Array | null;
  tall?: boolean;
  playheadMs?: number;
  showPlayhead?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const height = tall ? 180 : 64;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform) return;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    canvas.width = cssWidth * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, height);

    // Pre-compute speech-vs-silence sample-index ranges for each region.
    const ranges = regions.map((r) => ({
      start: Math.floor((r.start / durationMs) * waveform.length),
      end: Math.ceil((r.end / durationMs) * waveform.length),
    }));
    const inSpeech = (sampleIdx: number) =>
      ranges.some((r) => sampleIdx >= r.start && sampleIdx < r.end);

    const bars = cssWidth;
    const samplesPerBar = Math.max(1, Math.floor(waveform.length / bars));
    const mid = height / 2;

    for (let x = 0; x < bars; x++) {
      const start = x * samplesPerBar;
      const end = Math.min(waveform.length, start + samplesPerBar);
      let max = 0;
      for (let i = start; i < end; i++) {
        const a = Math.abs(waveform[i]);
        if (a > max) max = a;
      }
      ctx.fillStyle = inSpeech(start) ? "#10b981" : "#525252";
      const h = Math.max(1, Math.min(mid - 1, max * mid * 1.3));
      ctx.fillRect(x, mid - h, 1, h * 2);
    }
  }, [waveform, durationMs, regions, height]);

  const playheadPct =
    durationMs > 0 ? Math.max(0, Math.min(1, playheadMs / durationMs)) * 100 : 0;
  const inSpeechRegion = regions.some(
    (r) => playheadMs >= r.start && playheadMs <= r.end,
  );

  return (
    <div className="space-y-2">
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full rounded bg-neutral-950/80 block"
          style={{ width: "100%", height }}
        />
        {/* Translucent rect overlay per detected region. */}
        {regions.map((r, i) => {
          const startPct = (r.start / durationMs) * 100;
          const endPct = (r.end / durationMs) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 h-full border-l-2 border-r-2 border-emerald-500/60 pointer-events-none"
              style={{
                left: `${startPct}%`,
                width: `${endPct - startPct}%`,
                background: "rgba(16,185,129,0.08)",
              }}
            />
          );
        })}
        {showPlayhead && (
          <div
            className={`absolute top-0 h-full w-0.5 pointer-events-none transition-[left] duration-75 ease-linear shadow-[0_0_8px_rgba(255,255,255,0.6)] ${
              inSpeechRegion ? "bg-emerald-300" : "bg-white/90"
            }`}
            style={{ left: `${playheadPct}%` }}
          />
        )}
      </div>
      <div className="flex justify-between text-xs text-neutral-500 font-mono">
        <span>0 ms</span>
        <span className="text-emerald-400">
          ▼ {regions.length} speech region
          {regions.length === 1 ? "" : "s"}
        </span>
        <span>{durationMs} ms</span>
      </div>
    </div>
  );
}

function PhonemeStrip({
  frames,
  tall = false,
  playheadMs = 0,
  showPlayhead = false,
}: {
  frames: PhonemeWindowMessage["frames"];
  tall?: boolean;
  playheadMs?: number;
  showPlayhead?: boolean;
}) {
  const target = tall ? 180 : 120;
  const stride = Math.max(1, Math.ceil(frames.length / target));
  const sampled = frames.filter((_, i) => i % stride === 0);
  const cells = sampled.map((f) => {
    const top = f.phonemes[0];
    return {
      ts: f.timestamp_ms,
      symbol: top?.symbol ?? "·",
      prob: top?.prob ?? 0,
    };
  });

  const firstTs = sampled[0]?.timestamp_ms ?? 0;
  const lastTs = sampled[sampled.length - 1]?.timestamp_ms ?? 0;
  const totalDurationMs = sampled.length > 1 ? lastTs - firstTs : 0;
  const tickStepMs = totalDurationMs > 5000 ? 1000 : 500;
  const ticks: number[] = [];
  for (
    let t = Math.ceil(firstTs / tickStepMs) * tickStepMs;
    t <= lastTs;
    t += tickStepMs
  ) {
    ticks.push(t);
  }

  // Find the cell whose timestamp is the latest <= the current playhead.
  let activeIdx = -1;
  if (showPlayhead && cells.length > 0) {
    if (playheadMs <= cells[0].ts) activeIdx = 0;
    else if (playheadMs >= cells[cells.length - 1].ts)
      activeIdx = cells.length - 1;
    else {
      let lo = 0;
      let hi = cells.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (cells[mid].ts <= playheadMs) lo = mid;
        else hi = mid;
      }
      activeIdx = lo;
    }
  }
  const activeCell = activeIdx >= 0 ? cells[activeIdx] : null;
  const activePhoneme =
    activeCell && activeCell.symbol !== "<pad>" ? activeCell.symbol : null;

  return (
    <div className="space-y-3">
      <div className="rounded bg-neutral-950/50 p-4">
        {showPlayhead && (
          <div className="flex items-baseline justify-between mb-3 px-1">
            <div className="text-xs text-neutral-500 font-mono">
              t = {(playheadMs / 1000).toFixed(2)} s
            </div>
            <div className="font-mono">
              <span className="text-[10px] uppercase tracking-wide text-neutral-500 mr-2">
                hearing now:
              </span>
              <span
                className={`text-2xl ${
                  activePhoneme ? "text-emerald-300" : "text-neutral-600"
                }`}
              >
                {activePhoneme ?? "—"}
              </span>
              {activeCell && activePhoneme && (
                <span className="text-xs text-neutral-500 ml-2">
                  {(activeCell.prob * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>
        )}
        <div
          className={`flex flex-nowrap gap-px font-mono overflow-x-auto pb-2 ${
            tall ? "text-base" : "text-[11px]"
          }`}
        >
          {cells.map((c, i) => {
            const isBlank = c.symbol === "<pad>";
            const opacity = Math.min(1, 0.2 + c.prob * 0.8);
            const isActive = i === activeIdx;
            const isPast = activeIdx >= 0 && i < activeIdx;
            return (
              <span
                key={i}
                title={`t=${c.ts} ms · ${c.symbol} (${(c.prob * 100).toFixed(0)}%)`}
                className={`flex-shrink-0 text-center rounded transition-all duration-75 ${
                  tall ? "min-w-[24px] py-2 px-1.5" : "min-w-[14px] py-1 px-1"
                } ${
                  isActive
                    ? "bg-emerald-400 text-neutral-950 font-bold scale-125 z-10 shadow-[0_0_12px_rgba(52,211,153,0.7)]"
                    : isBlank
                      ? isPast
                        ? "bg-neutral-700/40 text-neutral-500"
                        : "bg-neutral-800/40 text-neutral-600"
                      : isPast
                        ? "bg-emerald-900/40 text-emerald-300/70"
                        : "bg-emerald-900/60 text-emerald-100"
                }`}
                style={{ opacity: isActive ? 1 : opacity }}
              >
                {isBlank ? "·" : c.symbol}
              </span>
            );
          })}
        </div>
        <div className="relative h-4 mt-2 border-t border-neutral-800">
          {ticks.map((t) => (
            <div
              key={t}
              className="absolute top-0 text-[10px] text-neutral-500 font-mono"
              style={{
                left:
                  totalDurationMs > 0
                    ? `${((t - firstTs) / totalDurationMs) * 100}%`
                    : "0%",
                transform: "translateX(-50%)",
              }}
            >
              <div className="border-l border-neutral-700 h-1 mx-auto" />
              <div className="mt-0.5">{(t / 1000).toFixed(1)}s</div>
            </div>
          ))}
          {showPlayhead && totalDurationMs > 0 && (
            <div
              className="absolute -top-1 h-1 w-0.5 bg-white/80 pointer-events-none transition-[left] duration-75 ease-linear"
              style={{
                left: `${
                  Math.max(
                    0,
                    Math.min(1, (playheadMs - firstTs) / totalDurationMs),
                  ) * 100
                }%`,
              }}
            />
          )}
        </div>
      </div>
      <div className="text-xs text-neutral-500 leading-relaxed max-w-3xl">
        <strong className="text-neutral-300">Reading this:</strong> each cell is
        the most-likely phoneme at that 20 ms frame. Brightness reflects model
        confidence; faded gray cells are CTC blanks (silence between phonemes).
        {showPlayhead &&
          " Press play to watch the active phoneme scrub in sync with the audio."}
      </div>
    </div>
  );
}

function SimilarityMeter({
  value,
  matchThreshold,
  rejectThreshold,
  label,
  isPatient,
}: {
  value: number;
  matchThreshold: number;
  rejectThreshold: number;
  label: string;
  isPatient: boolean;
}) {
  // Animate the marker from 0 → value when the value first arrives, so
  // the user sees the score "land" rather than just appear.
  const [animPct, setAnimPct] = useState(0);
  useEffect(() => {
    setAnimPct(0);
    const id = requestAnimationFrame(() => {
      setAnimPct(Math.max(0, Math.min(1, value)) * 100);
    });
    return () => cancelAnimationFrame(id);
  }, [value]);
  const pct = animPct;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-xs text-neutral-400 uppercase tracking-wide">
          Cosine similarity to enrolled patient embedding
        </div>
        <div className="font-mono text-base">{value.toFixed(3)}</div>
      </div>
      <div className="relative h-10 rounded-full bg-neutral-900 overflow-hidden border border-neutral-800">
        <div className="absolute top-0 left-0 h-full bg-red-900/30" style={{ width: `${rejectThreshold * 100}%` }} />
        <div
          className="absolute top-0 h-full bg-amber-900/30"
          style={{
            left: `${rejectThreshold * 100}%`,
            width: `${(matchThreshold - rejectThreshold) * 100}%`,
          }}
        />
        <div
          className="absolute top-0 h-full bg-emerald-900/30"
          style={{
            left: `${matchThreshold * 100}%`,
            width: `${(1 - matchThreshold) * 100}%`,
          }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-neutral-300/70"
          style={{ left: `${rejectThreshold * 100}%` }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-neutral-300/70"
          style={{ left: `${matchThreshold * 100}%` }}
        />
        <div
          className="absolute top-0 h-full w-1.5 bg-white shadow-[0_0_10px_rgba(255,255,255,0.7)] transition-[left] duration-700 ease-out"
          style={{ left: `calc(${pct}% - 3px)` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-neutral-500 font-mono mt-1">
        <span className="text-red-400">
          ≤ {rejectThreshold} reject
        </span>
        <span className="text-amber-400">
          {rejectThreshold}–{matchThreshold} listener
        </span>
        <span className="text-emerald-400">
          ≥ {matchThreshold} patient
        </span>
      </div>
      <div className="text-sm mt-2">
        <span className="text-neutral-400">→ matched as </span>
        <span
          className={`font-semibold ${isPatient ? "text-emerald-300" : "text-neutral-300"}`}
        >
          {label}
        </span>
        {isPatient && (
          <span className="text-emerald-500/70 text-xs ml-2">
            (patient pipeline activated)
          </span>
        )}
      </div>
    </div>
  );
}

function ThresholdMeter({
  label,
  value,
  min,
  max,
  threshold,
  thresholdSide,
  suffix,
  helpAbove,
  helpBelow,
  tall = false,
}: {
  label: string;
  value: number | null;
  min: number;
  max: number;
  threshold: number;
  thresholdSide: "above" | "below";
  suffix: string;
  helpAbove: string;
  helpBelow: string;
  tall?: boolean;
}) {
  const range = max - min;
  const targetPct =
    value === null ? 0 : Math.max(0, Math.min(1, (value - min) / range)) * 100;
  const threshPct = ((threshold - min) / range) * 100;
  const ok =
    value !== null &&
    (thresholdSide === "above" ? value < threshold : value > threshold);
  const h = tall ? "h-10" : "h-5";

  // Animate fill from 0 → target on value arrival.
  const [animPct, setAnimPct] = useState(0);
  useEffect(() => {
    if (value === null) {
      setAnimPct(0);
      return;
    }
    setAnimPct(0);
    const id = requestAnimationFrame(() => setAnimPct(targetPct));
    return () => cancelAnimationFrame(id);
  }, [value, targetPct]);

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <div className="text-xs text-neutral-400 uppercase tracking-wide">
          {label}
        </div>
        <div className="font-mono text-base">
          {value === null ? "—" : value.toFixed(2)}
          {suffix}
        </div>
      </div>
      <div className={`relative ${h} rounded-full bg-neutral-900 overflow-hidden border border-neutral-800`}>
        <div
          className={`absolute top-0 left-0 h-full transition-[width] duration-1000 ease-out ${
            ok
              ? "bg-gradient-to-r from-emerald-700/30 to-emerald-500/60"
              : "bg-gradient-to-r from-amber-700/30 to-amber-500/70"
          }`}
          style={{ width: `${animPct}%` }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-neutral-300/70"
          style={{ left: `${threshPct}%` }}
        />
        {value !== null && (
          <div
            className="absolute top-0 h-full w-1 bg-white shadow-[0_0_10px_rgba(255,255,255,0.7)] transition-[left] duration-1000 ease-out"
            style={{ left: `calc(${animPct}% - 2px)` }}
          />
        )}
      </div>
      <div className="flex justify-between text-[10px] text-neutral-500 font-mono mt-1">
        <span>
          {min}
          {suffix}
        </span>
        <span className="text-neutral-400">
          threshold {threshold}
          {suffix}
        </span>
        <span>
          {max}
          {suffix}
        </span>
      </div>
      <div className="text-xs text-neutral-500 mt-1">
        {ok ? helpBelow : helpAbove}
      </div>
    </div>
  );
}

function DiffText({
  hypothesis,
  reference,
  size = "md",
}: {
  hypothesis: string;
  reference: string | null;
  size?: "md" | "lg";
}) {
  if (!hypothesis) {
    return (
      <span className="text-neutral-500 italic">(empty transcription)</span>
    );
  }
  if (!reference) {
    return <span className="text-neutral-100">{hypothesis}</span>;
  }
  const refTokens = tokenize(reference);
  const refSet = new Set(refTokens);
  const hypTokens = hypothesis.split(/\s+/);
  const sizeCls = size === "lg" ? "text-lg" : "text-base";
  return (
    <div className={`text-neutral-100 leading-relaxed ${sizeCls}`}>
      {hypTokens.map((tok, i) => {
        const norm = normalizeWord(tok);
        const match = norm.length > 0 && refSet.has(norm);
        return (
          <span key={i}>
            <span
              className={
                match
                  ? "text-emerald-300"
                  : norm.length === 0
                    ? "text-neutral-100"
                    : "text-amber-300 underline decoration-amber-700/50 decoration-1"
              }
            >
              {tok}
            </span>
            {i < hypTokens.length - 1 ? " " : ""}
          </span>
        );
      })}
    </div>
  );
}

// ────────────────────────────────────────────────────────────── pure helpers

// Like DiffText but uses a passed accent color for wrong words instead of
// the soft amber default. The Problem slide passes the WER-severity color
// so wrong words are visually as alarming as the WER number itself.
function SeverityDiffText({
  hypothesis,
  reference,
  accent,
}: {
  hypothesis: string;
  reference: string | null;
  accent: string;
}) {
  if (!hypothesis) {
    return (
      <span className="text-neutral-500 italic">(empty transcription)</span>
    );
  }
  if (!reference) return <span className="text-neutral-100">{hypothesis}</span>;
  const refSet = new Set(tokenize(reference));
  const hypTokens = hypothesis.split(/\s+/);
  return (
    <div className="text-neutral-100 leading-relaxed text-base">
      {hypTokens.map((tok, i) => {
        const norm = normalizeWord(tok);
        const match = norm.length > 0 && refSet.has(norm);
        return (
          <span key={i}>
            <span
              style={
                match
                  ? { color: "#86efac" }
                  : norm.length === 0
                    ? undefined
                    : { color: accent }
              }
            >
              {tok}
            </span>
            {i < hypTokens.length - 1 ? " " : ""}
          </span>
        );
      })}
    </div>
  );
}

// Map a WER value (0..>1) to a color so the audience can see severity at
// a glance — but resist the urge to editorialize with alarmist labels.
// Whisper isn't "failing"; it's out of distribution. The numbers and the
// hue do the work; we don't need exclamation points.
function werSeverityStyle(w: number): {
  accent: string;
  bgRgb: string;
  label: string;
} {
  if (w >= 1.0) {
    return { accent: "#dc2626", bgRgb: "220,38,38", label: "very high" };
  }
  if (w >= 0.5) {
    return { accent: "#f97316", bgRgb: "249,115,22", label: "high" };
  }
  if (w >= 0.2) {
    return { accent: "#f59e0b", bgRgb: "245,158,11", label: "moderate" };
  }
  return { accent: "#84cc16", bgRgb: "132,204,22", label: "low" };
}

// Approximation of Whisper's official EnglishTextNormalizer (the one used
// in evaluate_adapter.py and find_demo_clips.py). The analysis script and
// the page need to agree on WER, otherwise the slide numbers undersell
// the reduction the adapter actually delivers.
//
// Whisper expands contractions ("don't" → "do not") before tokenizing,
// which counts as 2 tokens instead of 1. Without this, common Whisper
// outputs like "I don't think..." understate WER vs an analysis that
// matches the python reference. Comprehensive list: only the common
// English contractions Whisper actually emits — letters-only tokens.
const CONTRACTIONS: Record<string, string> = {
  "ain't": "is not",
  "aren't": "are not",
  "can't": "cannot",
  "could've": "could have",
  "couldn't": "could not",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'll": "he will",
  "he's": "he is",
  "i'd": "i would",
  "i'll": "i will",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it would",
  "it'll": "it will",
  "it's": "it is",
  "let's": "let us",
  "she'd": "she would",
  "she'll": "she will",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "that's": "that is",
  "there's": "there is",
  "they'd": "they would",
  "they'll": "they will",
  "they're": "they are",
  "they've": "they have",
  "wasn't": "was not",
  "we'd": "we would",
  "we'll": "we will",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what's": "what is",
  "where's": "where is",
  "who's": "who is",
  "won't": "will not",
  "would've": "would have",
  "wouldn't": "would not",
  "you'd": "you would",
  "you'll": "you will",
  "you're": "you are",
  "you've": "you have",
};

function tokenize(s: string): string[] {
  // Normalize curly apostrophes → straight, lowercase, then expand
  // contractions before stripping punctuation. The order matters: if we
  // strip apostrophes first, "don't" becomes "dont" and won't match.
  let normalized = s.toLowerCase().replace(/[‘’ʼ]/g, "'");
  // Expand contractions word-by-word (whole-word match only).
  normalized = normalized.replace(/\b[a-z]+'[a-z]+\b/g, (m) =>
    CONTRACTIONS[m] ?? m,
  );
  return normalized
    .replace(/[^a-z0-9'\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function normalizeWord(w: string): string {
  return w.toLowerCase().replace(/[^a-z0-9']/g, "");
}

function levenshteinTokens(a: string[], b: string[]): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = new Array(b.length + 1).fill(0);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    const curr = new Array(b.length + 1).fill(0);
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return prev[b.length];
}

function wer(reference: string, hypothesis: string | null): number | null {
  if (hypothesis === null) return null;
  const ref = tokenize(reference);
  const hyp = tokenize(hypothesis);
  if (ref.length === 0) return null;
  return levenshteinTokens(ref, hyp) / ref.length;
}
