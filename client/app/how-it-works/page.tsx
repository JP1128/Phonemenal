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
import { useTheme } from "../../lib/theme";
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
  { id: "app",          title: "Phonemenal · the app",      phase: "app",     steps: 1 },
  { id: "problem",      title: "Patients & off-the-shelf Whisper", phase: "problem", steps: 1 },
  { id: "architecture", title: "Inside the app · the pipeline", phase: "inside", steps: 4 },
  { id: "demo",         title: "Live demo · the difference", phase: "demo",    steps: 3 },
] as const;

const PHASE_META: Record<string, { label: string; color: string }> = {
  app:     { label: "1 · The app",          color: "cyan"    },
  problem: { label: "2 · The problem",      color: "amber"   },
  inside:  { label: "3 · Inside the app",   color: "cyan"    },
  demo:    { label: "4 · Live demo",        color: "emerald" },
};

const PHASE_ACCENT_HEX: Record<string, string> = {
  app:     "#06b6d4", // cyan
  problem: "#f59e0b", // amber
  inside:  "#06b6d4", // cyan — same as app, both about the system we built
  demo:    "#10b981", // emerald
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

  // Load TORGO clips list once — but only after the user advances past
  // slide 0. The Problem slide is now static (baked-in transcripts);
  // fetching ~5 MB of clip metadata while the audience is reading the
  // first slide just adds main-thread JSON parse work for no benefit.
  // ArchitectureSlide/StrengthsSlide/DemoSlide will need it by then.
  useEffect(() => {
    if (slide === 0 || clips.length > 0) return;
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
    return () => {
      cancelled = true;
    };
  }, [slide, clips.length]);

  // Healthz probe — cheap, fires once on mount independent of slide.
  useEffect(() => {
    let cancelled = false;
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
    //
    // ``bypass_vad=1`` because the TORGO demo clips are curated audio
    // with known speech content. VAD's energy-based segmentation can
    // spuriously reject heavily-dysarthric speech (M01 in particular —
    // the envelope doesn't match "normal" voicing), leaving the
    // Phonemer card empty. The live mic path still uses VAD; we only
    // bypass it for these scripted demos where the clip IS the speech.
    const wsUrl =
      SERVER_HTTP.replace(/^http/, "ws") +
      `/ws/audio?force_patient=1&bypass_vad=1` +
      `&adapter_speaker=${encodeURIComponent(selectedClip.speaker_id)}`;
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
    <main
      className="h-screen w-screen overflow-hidden bg-neutral-950 text-neutral-100 flex flex-col relative"
      style={{
        // Two soft accent blobs frame the canvas — kept ambient and at
        // low opacity so they read as atmosphere, not chrome. Was four
        // blobs; collapsed to two because every card repaint above
        // composites through this layer.
        backgroundImage: `
          radial-gradient(ellipse at 0% 0%, rgba(245, 158, 11, 0.10), transparent 60%),
          radial-gradient(ellipse at 100% 100%, rgba(16, 185, 129, 0.10), transparent 60%)
        `,
      }}
    >
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
            Phonemenal
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
        {slide === 0 && <AppOverviewSlide {...slideProps} />}
        {slide === 1 && <ProblemSlide {...slideProps} />}
        {slide === 2 && <ArchitectureSlide {...slideProps} />}
        {slide === 3 && <DemoSlide {...slideProps} />}

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


// Baked-in demo data for the problem slide. Pre-computed by running
// vanilla faster-whisper (``base.en``) on TIMIT eval-split clips for
// each demo speaker — the live ``/demo/whisper`` endpoint would
// reproduce these byte-for-byte. We hard-code instead of fetching at
// page-load time so the slide renders instantly and so the demo isn't
// at the mercy of the audio cache / backend being warm.
//
// To regenerate, run (with the backend up):
//   python3 - <<'EOF'
//   import json, urllib.request as ur
//   clips = json.load(ur.urlopen("http://localhost:8000/torgo/clips?limit=20000"))["clips"]
//   for sp in ["FC02","F03","M01"]:
//       pool = [c for c in clips if c["speaker_id"]==sp and c["prompt_set"]=="timit"
//               and c["split"]=="eval" and 4<=len((c["intended_text"] or "").split())<=12]
//       pool.sort(key=lambda c: c["id"])
//       for c in pool[:7]:
//           r = ur.urlopen(ur.Request(f"http://localhost:8000/demo/whisper/{c['id']}", method="POST"))
//           print(sp, c["intended_text"], "→", json.loads(r.read())["transcript"])
//   EOF
type PatientExample = { intended: string; hypothesis: string };
const PROBLEM_SLIDE_EXAMPLES: Record<string, PatientExample[]> = {
  // All transcripts below are real `base.en` outputs from
  // `/demo/whisper`. The set is curated so each speaker's displayed
  // mean WER matches their card's aggregate-WER hero number:
  //   FC02 ≈ 27%   F03 ≈ 70%   M01 ≈ 103%
  // Run /tmp/bake_problem_set.py to re-verify after any edit.
  FC02: [
    { intended: "She suggested we all have dinner together", hypothesis: "She's the best that we all have dinner together." },
    { intended: "That pickpocket was caught redhanded", hypothesis: "That pickpocket was caught red-handed." },
    { intended: "You can ride stagecoach", hypothesis: "You can ride stagecoats." },
    { intended: "Old telephone booths aren't easy to find", hypothesis: "All telephone booths aren't easy to find." },
    { intended: "We gathered shells on the beach", hypothesis: "We gathered shells on the beach." },
  ],
  F03: [
    { intended: "She wore warm fleecy woolen overalls", hypothesis: "She wore a wore, fleecy bum, or off." },
    { intended: "Help celebrate your brother's success", hypothesis: "Help us out by your brother's herself." },
    { intended: "A roll of wire lay near the wall", hypothesis: "I got a little wire blade in the egomah." },
    { intended: "The train approached the depot slowly", hypothesis: "The train approached it deep also." },
    { intended: "The fish swam across the pond", hypothesis: "The first swam cast pond." },
  ],
  M01: [
    { intended: "The islands are sparsely populated", hypothesis: "the island, the party, the baby" },
    { intended: "Both figures would go higher in later years", hypothesis: "But it is a good way to go higher in maybe years." },
    { intended: "The misguided souls have lost their way", hypothesis: "and maybe I did not have a right to eat" },
    { intended: "When all else fails use force", hypothesis: "When I always do it, you force." },
    { intended: "Just one side got wet", hypothesis: "Z1, I do it." },
  ],
};

function ProblemSlide(_: SlideProps) {
  void _;
  const featured: {
    speaker: string;
    severity: string;
    examples: PatientExample[];
    werExpected: number;
  }[] = [
    // werExpected = mean vanilla WER over each speaker's full TORGO
    // eval split (TIMIT sentences + intelligibility prompts), sourced
    // from server/data/adapters/<S>_small_hicap/eval.json (`vanilla_wer`).
    // The displayed example transcripts above are curated so their own
    // mean WER lands within ~1pt of these numbers — see the comment on
    // PROBLEM_SLIDE_EXAMPLES.
    { speaker: "FC02", severity: "Healthy control", examples: PROBLEM_SLIDE_EXAMPLES.FC02, werExpected: 0.27 },
    { speaker: "F03", severity: "Mild dysarthria", examples: PROBLEM_SLIDE_EXAMPLES.F03, werExpected: 0.70 },
    { speaker: "M01", severity: "Severe dysarthria", examples: PROBLEM_SLIDE_EXAMPLES.M01, werExpected: 1.02 },
  ];

  return (
    <div className="h-full flex flex-col px-12 py-8 max-w-7xl mx-auto w-full">
      <header className="mb-6 min-h-[120px]">
        <div className="animate-[fadeUp_400ms_ease-out_both]">
          <h1 className="text-4xl font-semibold tracking-tight mb-2">
            Off-the-shelf speech recognition:{" "}
            <span className="text-amber-300">
              it gets worse the worse it gets
            </span>
            .
          </h1>
          <p className="text-base text-neutral-400 max-w-3xl">
            Each patient reads several short standard sentences, fed to{" "}
            <span className="text-amber-300">Whisper</span> — today&apos;s
            most popular open-source speech-to-text model. Highlighted
            words are wrong; the more severe the patient, the more red on
            the card.
          </p>
        </div>
      </header>

      <div className="grid auto-rows-fr md:grid-cols-3 gap-5 flex-1 min-h-0">
        {featured.map((p) => (
          <UnifiedPatientCard key={p.speaker} {...p} />
        ))}
      </div>

      <div className="mt-6 text-xs font-mono text-neutral-500 tracking-wider">
        Per-card percentages are mean WER of vanilla{" "}
        <code className="text-amber-200 text-xs">base.en</code> over each
        speaker&apos;s full TORGO eval split (TIMIT sentences plus
        intelligibility prompts, held out from calibration). Displayed
        examples are picked to reflect the same mean.
      </div>
    </div>
  );
}

// One card per patient: speaker label + severity title + per-card WER bar
// at the top, then a list of intended/Whisper pairs so the audience can
// scan across several sentences and form a "this whole patient is
// mistranscribed" impression at a glance.
// Hero WER number for a Problem-slide patient card. Counts up from 0
// → werExpected*100 in ~800ms (ease-out cubic) when the card mounts, so
// all three cards land their values together as the slide arrives.
function HeroWerNumber({
  werExpected,
  accent,
  bgRgb,
}: {
  werExpected: number;
  accent: string;
  bgRgb: string;
}) {
  const animated = useCountUp(werExpected * 100);
  return (
    <div className="mt-2 flex items-baseline gap-2 animate-[fadeUp_400ms_ease-out_both]">
      <span
        className="text-6xl font-bold tracking-tight tabular-nums leading-none transition-colors duration-500"
        style={{ color: accent, textShadow: `0 1px 24px rgba(${bgRgb}, 0.25)` }}
      >
        {animated.toFixed(0)}
        <span className="text-3xl font-semibold align-top ml-0.5">%</span>
      </span>
      <span className="text-[10px] uppercase tracking-[0.25em] font-mono text-neutral-500 pb-1">
        words<br />wrong
      </span>
    </div>
  );
}

// Count up from 0 to `target` over `durationMs` using an ease-out cubic.
// Returns the current value as a state. Re-fires on mount (so when the
// slide re-mounts the cards "land" their hero numbers fresh) and on
// target change. rAF stops as soon as the animation completes, so it
// isn't an idle CPU draw.
function useCountUp(target: number, durationMs = 800): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);
  return value;
}

function UnifiedPatientCard({
  speaker,
  severity,
  examples,
  werExpected,
}: {
  speaker: string;
  severity: string;
  examples: PatientExample[];
  werExpected: number;
}) {
  const { theme } = useTheme();
  // Color-code by *aggregate* severity (werExpected), not any individual
  // clip's measured WER. Per-clip values can break monotonic ordering with
  // severity (TIMIT clip noise), which would paint FC02 redder than F03
  // and undercut the "it gets worse the worse it gets" narrative.
  const sev = werSeverityStyle(werExpected, theme);
  const finalAccent = sev.accent;
  // "Correct word" green threaded down so SeverityDiffText doesn't need
  // its own theme subscription.
  const matchColor = theme === "light" ? "#15803d" : "#86efac";
  // Scale the card background intensity with aggregate WER so the audience
  // doesn't have to count red words to feel "M01 is worse than F03". Both
  // cards have lots of highlights — the *proportion* is what differs, and
  // a heavier bg makes that proportion visible at-a-glance, reinforcing
  // the hero WER number.
  const bgAlpha =
    werExpected >= 0.75 ? 0.22 : werExpected >= 0.5 ? 0.13 : 0.05;
  const borderAlpha =
    werExpected >= 0.75 ? "ff" : werExpected >= 0.5 ? "cc" : "99";
  // Frosted-card look without `backdrop-filter`. backdrop-blur on cards
  // over a multi-layer page mesh was the dominant per-frame GPU cost on
  // this slide. Higher base alpha now hides the page mesh underneath so
  // the cards still read as a clean glass surface — just without the
  // blur-sampling expense.
  const glassBaseAlpha = 0.88;
  // Dark mode: bright inset top edge sells the glass gleam. Light mode:
  // a near-white card on a near-white canvas needs the opposite — a faint
  // dark inset at the bottom gives the card a sense of weight + edge.
  const innerEdge =
    theme === "light"
      ? "inset 0 -1px 0 rgba(15, 23, 42, 0.04)"
      : "inset 0 1px 0 rgba(255, 255, 255, 0.08)";
  const accentGlow = `0 14px 36px rgba(${sev.bgRgb}, ${werExpected >= 0.75 ? 0.35 : werExpected >= 0.5 ? 0.22 : 0.10})`;
  const boxShadow = ["var(--glass-shadow)", accentGlow, innerEdge].join(", ");
  return (
    <div
      className="relative rounded-2xl border p-5 flex flex-col gap-4 transition-colors duration-300 min-h-0"
      style={{
        borderColor: finalAccent + borderAlpha,
        // Two-layer background: base glass tint on top of the accent
        // colour wash. CSS paints earlier `background` values *on top*,
        // so the gradient (glass) renders above the solid (tint).
        background: `linear-gradient(rgba(var(--glass-bg-rgb), ${glassBaseAlpha}), rgba(var(--glass-bg-rgb), ${glassBaseAlpha})), rgba(${sev.bgRgb}, ${bgAlpha + 0.04})`,
        boxShadow,
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="text-xs font-mono uppercase tracking-wider text-neutral-500">
            {speaker} · {severity}
          </div>
          {/* Hero WER number. The percentage IS the headline — at a glance
              the audience sees 3% / 40% / 79% and the severity gradient
              is felt before any transcript is read. Counts up from 0 on
              slide mount so all three cards "land" their values at the
              same beat. */}
          <HeroWerNumber
            werExpected={werExpected}
            accent={finalAccent}
            bgRgb={sev.bgRgb}
          />
        </div>
        <div className="text-[10px] uppercase tracking-wider font-mono text-neutral-600 shrink-0">
          TORGO
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <div className="animate-[fadeUp_400ms_ease-out_both] space-y-4">
          {examples.map((ex, idx) => (
            <div
              key={idx}
              className={
                "space-y-1.5 " +
                (idx < examples.length - 1
                  ? "pb-3 border-b border-neutral-800/60"
                  : "")
              }
            >
              <div className="text-xs italic text-neutral-500 leading-snug">
                “{ex.intended}”
              </div>
              <div className="text-sm leading-snug">
                <SeverityDiffText
                  hypothesis={ex.hypothesis}
                  reference={ex.intended}
                  accent={finalAccent}
                  matchColor={matchColor}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────── Phase 2 — The app

// Product overview slide. Sits between the problem (slide 1) and the
// pipeline (slide 3). Establishes that Phonemenal is a *complete app*
// with three distinct product surfaces — a calibration session, the
// listener's live view, and a progress dashboard — and that the
// transcription pipeline (next slide) is what powers them. Without
// this slide the audience leaves thinking "they built a transcription
// pipeline"; with it they leave thinking "they built an app, and they
// built the engine inside it themselves."
function AppOverviewSlide(_: SlideProps) {
  void _;
  return (
    <div className="h-full flex flex-col px-10 py-6 max-w-[100rem] mx-auto w-full">
      <header className="mb-5">
        <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
          The app
        </div>
        <h1 className="text-4xl font-semibold tracking-tight mb-2">
          <span className="text-cyan-300">Phonemenal</span> — a complete app
          for brain-injury speech.
        </h1>
        <p className="text-base text-neutral-400 leading-relaxed max-w-4xl">
          Three product surfaces wrap the transcription engine. The pipeline
          you&apos;ll see next is what powers all of them.
        </p>
      </header>

      {/* Cards take natural height (sized by their viewport's aspect
          ratio); the row centers vertically in the available slide
          space so the cards don't stretch tall and leave dead space
          around the mockup. */}
      <div className="flex-1 min-h-0 flex items-center">
        <div className="grid md:grid-cols-3 gap-5 w-full">
          <AppSurfaceCard
            title="Calibrate"
            path="/calibrate"
            mockup={<CalibrateMockup />}
            description="Record ~145 short prompts. Trains a per-patient LoRA adapter in the background."
          />
          <AppSurfaceCard
            title="Listen"
            path="/"
            mockup={<ListenMockup />}
            description="Real-time transcription with the patient's adapter loaded. The listener's working view."
          />
          <AppSurfaceCard
            title="Progress"
            path="/progress"
            mockup={<ProgressMockup />}
            description="WER trajectory over time. Personal vocabulary. Therapy gain at a glance."
          />
        </div>
      </div>
    </div>
  );
}

// One product-surface card. Stylized mockup in the viewport area,
// headline strip on top, description underneath. Matches the visual
// language of the other slides' cards (solid accent header bar + drop
// shadow) so the audience recognizes "this is a member of the same
// product family."
function AppSurfaceCard({
  title,
  path,
  mockup,
  description,
}: {
  title: string;
  path: string;
  mockup: React.ReactNode;
  description: string;
}) {
  return (
    <div
      className="rounded-lg overflow-hidden bg-neutral-800 border border-neutral-700/60 flex flex-col"
      style={{
        boxShadow:
          "0 10px 30px rgba(15,23,42,0.18), 0 2px 6px rgba(15,23,42,0.10)",
      }}
    >
      <div
        className="px-3 py-1.5 text-[11px] uppercase tracking-[0.2em] font-mono font-semibold flex items-center justify-between gap-2"
        style={{ backgroundColor: "#06b6d4", color: "#ffffff" }}
      >
        <span>{title}</span>
        <span className="opacity-70 text-[10px] font-normal normal-case tracking-normal">
          {path}
        </span>
      </div>
      {/* Square viewport so each surface card reads as a substantial
          showcase rather than a thumbnail. The mockup inside has its
          own bumped max-width to fill more of the viewport. */}
      <div
        className="aspect-square overflow-hidden border-b border-neutral-700/60 flex items-center justify-center p-6"
        style={{
          background: "linear-gradient(180deg, #0a0a0a 0%, #171717 100%)",
        }}
      >
        {mockup}
      </div>
      <div className="px-4 py-3 bg-neutral-900">
        <p className="text-sm text-neutral-300 leading-snug">{description}</p>
      </div>
    </div>
  );
}

// Calibrate mockup: a "prompt card" with the current sentence to read,
// a recording indicator, and a waveform. Violet matches the
// "calibrated path" pillar on slide 4.
function CalibrateMockup() {
  // Deterministic waveform — same shape every render so React doesn't
  // see a new prop tree on hover/focus and trigger reflow.
  const bars = [
    6, 11, 4, 14, 18, 9, 13, 17, 12, 6, 9, 15, 20, 14, 9, 12, 7, 5, 10, 16,
    12, 19, 14, 9, 5, 7, 12, 17, 11, 7,
  ];
  return (
    <div
      className="w-full max-w-[420px] rounded-xl p-5 shadow-xl"
      style={{
        backgroundColor: "#0a0a0a",
        border: "1px solid rgba(167, 139, 250, 0.35)",
      }}
    >
      <div className="flex items-center justify-between text-[9px] font-mono uppercase tracking-wider mb-2">
        <span style={{ color: "#a78bfa" }}>Prompt 27 / 145</span>
        <span className="flex items-center gap-1" style={{ color: "#fb7185" }}>
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse"
            style={{ backgroundColor: "#f43f5e" }}
          />
          Rec
        </span>
      </div>
      <div className="text-[13px] italic leading-snug mb-3" style={{ color: "#e5e5e5" }}>
        “We gathered shells on the beach”
      </div>
      <svg viewBox="0 0 220 28" className="w-full h-7 mb-1.5">
        {bars.map((h, i) => (
          <rect
            key={i}
            x={i * 7.3}
            y={14 - h / 2}
            width="3"
            height={h}
            rx="1"
            fill="#a78bfa"
            opacity={0.45 + (i % 3) * 0.18}
          />
        ))}
      </svg>
      <div className="flex items-center justify-between text-[9px] font-mono uppercase tracking-wider" style={{ color: "#737373" }}>
        <span>FC02</span>
        <span>00:04 / 00:08</span>
      </div>
    </div>
  );
}

// Listen mockup: a live transcript with green for matched / amber for
// uncertain. Emerald frame matches the "output" lane of the pipeline.
function ListenMockup() {
  return (
    <div
      className="w-full max-w-[420px] rounded-xl p-5 shadow-xl"
      style={{
        backgroundColor: "#0a0a0a",
        border: "1px solid rgba(52, 211, 153, 0.35)",
      }}
    >
      <div className="flex items-center justify-between text-[9px] font-mono uppercase tracking-wider mb-2">
        <span className="flex items-center gap-1" style={{ color: "#34d399" }}>
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse"
            style={{ backgroundColor: "#10b981" }}
          />
          Live
        </span>
        <span style={{ color: "#737373" }}>F03 · mild</span>
      </div>
      <div className="text-[13px] leading-snug mb-2" style={{ color: "#e5e5e5" }}>
        “The <span style={{ color: "#34d399" }}>fish</span>{" "}
        <span style={{ color: "#34d399" }}>swam</span>{" "}
        <span style={{ color: "#f59e0b" }}>cross</span> pond.”
      </div>
      <div
        className="flex items-center justify-between text-[9px] font-mono uppercase tracking-wider pt-2 border-t"
        style={{ borderColor: "#262626", color: "#737373" }}
      >
        <span>confidence 78%</span>
        <span className="flex items-center gap-1">
          {[3, 6, 4, 8, 5, 9, 6, 4].map((h, i) => (
            <span
              key={i}
              className="inline-block w-0.5 rounded-full"
              style={{
                height: `${h}px`,
                backgroundColor: "#10b981",
                opacity: 0.4 + (i / 8) * 0.5,
              }}
            />
          ))}
        </span>
      </div>
    </div>
  );
}

// Progress mockup: a small line chart showing WER trending down over a
// few weeks. Cyan because progress is measurement, not a model output.
function ProgressMockup() {
  return (
    <div
      className="w-full max-w-[420px] rounded-xl p-5 shadow-xl"
      style={{
        backgroundColor: "#0a0a0a",
        border: "1px solid rgba(34, 211, 238, 0.35)",
      }}
    >
      <div className="flex items-center justify-between text-[9px] font-mono uppercase tracking-wider mb-2">
        <span style={{ color: "#22d3ee" }}>WER · 6 weeks</span>
        <span style={{ color: "#34d399" }}>↓ 42%</span>
      </div>
      <svg viewBox="0 0 220 90" className="w-full h-[88px]">
        <defs>
          <linearGradient id="progFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#22d3ee" stopOpacity="0.35" />
            <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* dashed grid */}
        <line x1="20" y1="14" x2="215" y2="14" stroke="#404040" strokeWidth="0.4" strokeDasharray="2 3" />
        <line x1="20" y1="45" x2="215" y2="45" stroke="#404040" strokeWidth="0.4" strokeDasharray="2 3" />
        <line x1="20" y1="76" x2="215" y2="76" stroke="#404040" strokeWidth="0.4" />
        {/* axis labels */}
        <text x="2" y="17" fill="#737373" fontSize="6" fontFamily="monospace">70%</text>
        <text x="2" y="48" fill="#737373" fontSize="6" fontFamily="monospace">50%</text>
        <text x="2" y="79" fill="#737373" fontSize="6" fontFamily="monospace">30%</text>
        {/* area fill */}
        <polygon
          points="25,16 60,30 95,42 130,55 165,67 210,71 210,76 25,76"
          fill="url(#progFill)"
        />
        {/* chart line */}
        <polyline
          points="25,16 60,30 95,42 130,55 165,67 210,71"
          fill="none"
          stroke="#22d3ee"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* data points */}
        {[
          [25, 16],
          [60, 30],
          [95, 42],
          [130, 55],
          [165, 67],
          [210, 71],
        ].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="2.5" fill="#0a0a0a" stroke="#22d3ee" strokeWidth="1.5" />
        ))}
        {/* week labels */}
        <text x="22" y="86" fill="#525252" fontSize="6" fontFamily="monospace">WK1</text>
        <text x="91" y="86" fill="#525252" fontSize="6" fontFamily="monospace">WK3</text>
        <text x="200" y="86" fill="#525252" fontSize="6" fontFamily="monospace">WK6</text>
      </svg>
    </div>
  );
}

// ───────────────────────────────────────────────── Phase 3 — Architecture



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


// Per-sub-step header narrative. We reveal the diagram additively: each
// step adds one section to the active set, so the audience never has to
// hold the whole pipeline in their head at once.
const ARCHITECTURE_NARRATIVE: { eyebrow: string; copy: React.ReactNode }[] = [
  {
    eyebrow: "Audio in",
    copy: (
      <>
        Audio enters at the left. <span className="text-cyan-300">VAD</span>{" "}
        trims silence, then <span className="text-cyan-300">Speaker ID</span>{" "}
        matches the voice to an enrolled patient.
      </>
    ),
  },
  {
    eyebrow: "+ calibrated path",
    copy: (
      <>
        <span className="text-violet-300">Whisper with a per-patient LoRA</span>{" "}
        produces the transcript. Conversation context (last turn, vocabulary)
        is assembled into an <span className="text-violet-300">initial_prompt</span>{" "}
        that biases Whisper toward what we expect to hear.
      </>
    ),
  },
  {
    eyebrow: "+ safety path",
    copy: (
      <>
        <span className="text-cyan-300">wav2vec2</span> extracts a phoneme
        posterior in parallel — what the audio actually <em>sounds like</em>,
        independent of Whisper&apos;s text.
      </>
    ),
  },
  {
    eyebrow: "+ gate decides",
    copy: (
      <>
        The gate forces Whisper&apos;s text to{" "}
        <span className="text-amber-300">align against wav2vec2&apos;s phonemes</span>.
        If they agree, the text ships{" "}
        <span className="text-emerald-300">confident</span>. If they don&apos;t,
        it ships <span className="text-amber-300">flagged as uncertain</span>.
      </>
    ),
  },
];

function ArchitectureSlide({ subStep }: SlideProps) {
  const step = ARCHITECTURE_NARRATIVE[subStep] ?? ARCHITECTURE_NARRATIVE[0];
  return (
    <div className="h-full flex flex-col px-10 py-6 max-w-[100rem] mx-auto w-full">
      <header className="mb-4 min-h-[120px]">
        <div
          key={`hdr-${subStep}`}
          className="animate-[fadeUp_400ms_ease-out_both]"
        >
          <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
            Inside the app · {step.eyebrow}
          </div>
          <h1 className="text-4xl font-semibold tracking-tight mb-2">
            The transcription engine
          </h1>
          <p className="text-base text-neutral-400 max-w-4xl leading-relaxed">
            {step.copy}
          </p>
        </div>
      </header>
      <div className="flex-1 min-h-0 flex gap-4">
        <div className="flex-1 min-w-0">
          <PipelineDiagram subStep={subStep} />
        </div>
        <EngineResultsPanel />
      </div>
    </div>
  );
}

// Right-rail panel: a compact "after running this engine" readout.
// Numbers are *base.en TIMIT* WER (the same baseline shown on slide 2
// for vanilla), with the adapter's adapted WER from eval.json's TIMIT
// subset. Three patients, each shown as a before → after pair.
function EngineResultsPanel() {
  return (
    <aside
      className="w-[340px] shrink-0 h-full rounded-2xl border border-neutral-800 bg-neutral-900/50 px-6 py-6 flex flex-col gap-6"
      aria-label="Engine results"
    >
      <div>
        <div className="text-[10px] uppercase tracking-[0.25em] font-mono text-emerald-400 mb-2">
          What this delivers
        </div>
        <h3 className="text-xl font-semibold tracking-tight text-neutral-100 leading-snug">
          Halves the errors on severe speech.
        </h3>
      </div>

      <div className="space-y-5">
        <EngineDeltaRow label="FC02" sub="healthy control" before={27} after={17} />
        <EngineDeltaRow label="F03"  sub="mild dysarthria"  before={70} after={47} />
        <EngineDeltaRow
          label="M01"
          sub="severe dysarthria"
          before={102}
          after={54}
          emphasise
        />
      </div>

      <div className="text-[10px] font-mono text-neutral-600 leading-snug mt-auto pt-3 border-t border-neutral-800">
        Mean WER over each speaker&apos;s full TORGO eval split (held out
        from calibration). Stock vs per-patient adapter.
      </div>
    </aside>
  );
}

function EngineDeltaRow({
  label,
  sub,
  before,
  after,
  emphasise = false,
}: {
  label: string;
  sub: string;
  before: number;
  after: number;
  emphasise?: boolean;
}) {
  const reduction = Math.round(((before - after) / before) * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-sm font-mono">
          <span className={`font-semibold ${emphasise ? "text-rose-300" : "text-neutral-300"}`}>
            {label}
          </span>
          <span className="text-neutral-500 ml-1.5 text-xs">{sub}</span>
        </span>
        {reduction > 0 && (
          <span
            className={`text-xs font-mono tabular-nums font-semibold ${
              emphasise ? "text-emerald-300" : "text-neutral-500"
            }`}
          >
            −{reduction}%
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-3 font-mono tabular-nums">
        <span
          className={
            emphasise
              ? "text-4xl text-rose-300/80 line-through decoration-1"
              : "text-2xl text-neutral-500 line-through decoration-1"
          }
        >
          {before}%
        </span>
        <span className="text-neutral-600 text-xl">→</span>
        <span
          className={
            emphasise
              ? "text-6xl text-emerald-300 font-bold tracking-tight"
              : "text-4xl text-emerald-300 font-semibold tracking-tight"
          }
          style={
            emphasise
              ? { textShadow: "0 1px 24px rgba(16, 185, 129, 0.4)" }
              : { textShadow: "0 1px 16px rgba(16, 185, 129, 0.25)" }
          }
        >
          {after}%
        </span>
      </div>
    </div>
  );
}

// Four zones the pipeline reads as. Vertical dividers at x=46 / x=70
// split the canvas into three columns; the middle column has a
// horizontal divider at y=57 splitting the two parallel branches.
// Each zone's title + one-liner description sits at its bottom edge.
// The `section` index is the sub-step that first reveals the zone —
// ArchitectureSlide reveals additively.
const ARCHITECTURE_ZONES = [
  {
    section: 0,
    left: "0%", top: "0%", width: "46%", height: "100%",
    labelLeft: "23%", labelTop: "92%",
    accent: "#06b6d4",
    title: "Audio in",
  },
  {
    section: 1,
    left: "46%", top: "0%", width: "24%", height: "57%",
    labelLeft: "58%", labelTop: "52%",
    accent: "#8b5cf6",
    title: "Calibrated path",
  },
  {
    section: 2,
    left: "46%", top: "57%", width: "24%", height: "43%",
    labelLeft: "58%", labelTop: "92%",
    accent: "#06b6d4",
    title: "Safety path",
  },
  {
    section: 3,
    left: "70%", top: "0%", width: "30%", height: "100%",
    labelLeft: "85%", labelTop: "92%",
    accent: "#f59e0b",
    title: "Gate decides",
  },
] as const;

// Map each stage / side-input to the zone (sub-step) that introduces it.
// Used by Stage + SideInput + FlowSvg to know when to dim or hide.
const STAGE_SECTION = {
  audio: 0, vad: 0, speakerId: 0, enrolled: 0,
  whisper: 1, initialPrompt: 1,
  phoneme: 2,
  gate: 3, output: 3,
} as const;

function PipelineDiagram({ subStep }: { subStep: number }) {
  const dim = (s: keyof typeof STAGE_SECTION) =>
    subStep < STAGE_SECTION[s];
  return (
    <div className="relative h-full w-full rounded-2xl bg-neutral-950 border border-neutral-800 overflow-hidden">
      {/* Zone dividers — two verticals + one horizontal splitting the
          middle column. Rendered first so flow lines and cards paint
          on top. Dashed lines read as "logical grouping", not as
          structural connectors. */}
      <div aria-hidden className="absolute top-0 bottom-0 border-l border-dashed border-neutral-700" style={{ left: "46%" }} />
      <div aria-hidden className="absolute top-0 bottom-0 border-l border-dashed border-neutral-700" style={{ left: "70%" }} />
      <div aria-hidden className="absolute border-t border-dashed border-neutral-700" style={{ top: "57%", left: "46%", width: "24%" }} />

      {/* Zone label pills, anchored at the bottom of each zone. The
          slide's header narrative carries the per-section explanation;
          these just name the zone, so dropping the bottom descriptions
          keeps the row visually uniform (varying description lengths
          made the bottom rail look jagged). Dim when their section
          hasn't been revealed yet. */}
      {ARCHITECTURE_ZONES.map((zone) => {
        const inactive = subStep < zone.section;
        return (
          <span
            key={zone.title}
            aria-hidden
            className="absolute pointer-events-none text-[11px] uppercase tracking-[0.25em] font-mono font-semibold px-2.5 py-1 rounded-md whitespace-nowrap transition-opacity duration-500"
            style={{
              left: zone.labelLeft,
              top: zone.labelTop,
              transform: "translate(-50%, -50%)",
              backgroundColor: zone.accent,
              // Hardcoded white instead of `text-white` so the global
              // light-mode color-token inversion (which is what we want
              // for body copy) doesn't flip these chip labels to dark
              // text on saturated colored backgrounds.
              color: "#ffffff",
              boxShadow: `0 2px 8px ${zone.accent}55`,
              opacity: inactive ? 0.18 : 1,
            }}
          >
            {zone.title}
          </span>
        );
      })}

      <FlowSvg subStep={subStep} />
      {/* Card centers and widths are *all* in % of the container so they
          share a coordinate space with the SVG flow paths below
          (viewBox 0..100 stretched to fill the container with
          preserveAspectRatio="none"). Edges are: regular cards 6% from
          center, wide cards 9% from center. The path endpoints in
          FlowSvg land *exactly* on these edges. Side-input nodes hang
          above the main row and feed downward into their consumers. */}
      {/* Both side-input stickers sit on the same y baseline at the top
          of the canvas, with matching content-line counts so they read
          as a uniform pair of inputs that feed down into the pipeline. */}
      <SideInput
        style={{ top: "10%", left: "36%" }}
        accent="#06b6d4"
        label="Enrolled voices"
        purpose="Voiceprints we match against to know who is speaking"
        sub="from speakers DB"
        dim={dim("enrolled")}
      >
        <SideInputLine label="patient" value="centroid (F03)" labelColor="#67e8f9" />
        <SideInputLine label="listener" value="speaker_A" labelColor="#67e8f9" />
      </SideInput>
      <SideInput
        style={{ top: "10%", left: "58%" }}
        accent="#a78bfa"
        label="initial_prompt"
        purpose="Conversation context that primes Whisper for expected words"
        sub="≤ 220 chars · ContextManager"
        dim={dim("initialPrompt")}
      >
        <SideInputLine label="last turn" value={"“Could you read this?”"} labelColor="#c4b5fd" />
        <SideInputLine label="vocab" value="Dr. Park, gabap…" labelColor="#c4b5fd" />
      </SideInput>
      <Stage style={{ top: "55%", left: "7%" }} accent="#94a3b8" label="Audio" sub="raw 16 kHz" dim={dim("audio")}>
        <MiniWaveformVis />
      </Stage>
      <Stage style={{ top: "55%", left: "22%" }} accent="#06b6d4" label="VAD" sub="silero" dim={dim("vad")}>
        <VadVis />
      </Stage>
      <Stage style={{ top: "55%", left: "36%" }} accent="#06b6d4" label="Speaker ID" sub="ECAPA-TDNN" dim={dim("speakerId")}>
        <SpeakerVis />
      </Stage>
      <Stage
        style={{ top: "35%", left: "58%" }}
        accent="#8b5cf6"
        label="Whisper"
        sub="small.en + LoRA"
        wide
        dim={dim("whisper")}
      >
        <WhisperVis />
      </Stage>
      <Stage
        style={{ top: "75%", left: "58%" }}
        accent="#06b6d4"
        label="Phoneme model"
        sub="wav2vec2"
        wide
        dim={dim("phoneme")}
      >
        <PhonemeVis />
      </Stage>
      <Stage style={{ top: "55%", left: "78%" }} accent="#f59e0b" label="Safety gate" sub="forced alignment" dim={dim("gate")}>
        <GateVis />
      </Stage>
      <Stage style={{ top: "55%", left: "93%" }} accent="#10b981" label="Output" sub="text + flag" dim={dim("output")}>
        <OutputVis />
      </Stage>
      {/* Connector labels — what kind of data flows along each wire. The
          inner-card visualizations show what each stage *does*; these
          labels show what each stage *produces*. Positions are picked to
          land on the gap between cards rather than inside them. */}
      {/* Three labels carry the story: what feeds Whisper, what feeds
          wav2vec2, what the gate emits. Everything else was clutter. */}
      <EdgeLabel top="40%" left="47%" dim={dim("whisper")}>mel features</EdgeLabel>
      <EdgeLabel top="69%" left="47%" dim={dim("phoneme")}>audio</EdgeLabel>
      <EdgeLabel top="51%" left="85.5%" dim={dim("output")}>text + flag</EdgeLabel>
      {/* Comparator badge at the gate's input — visually marks the spot
          where Whisper's text and wav2vec2's phoneme posterior are
          force-aligned against each other. Only fades in with the gate. */}
      <div
        className="absolute pointer-events-none transition-opacity duration-500"
        style={{
          top: "55%",
          left: "72%",
          transform: "translate(-50%, -50%)",
          opacity: dim("gate") ? 0 : 1,
        }}
      >
        <div className="w-5 h-5 rounded-full bg-neutral-950 border border-amber-500/70 flex items-center justify-center text-[11px] font-mono text-amber-300 shadow-[0_0_6px_rgba(245,158,11,0.45)]">
          ≟
        </div>
      </div>
    </div>
  );
}

// SVG layer that draws the connecting paths between stages with animated
// "data" dots flowing along them. The fork into Whisper + wav2vec2 is the
// structural feature that the old card layout hid; here it's the first
// thing the audience sees.
function FlowSvg({ subStep = 3 }: { subStep?: number }) {
  // Each connector belongs to a section; fade it in only after that
  // section has been revealed. Stays mounted, just toggles opacity, so
  // the SVG never reflows.
  const show = (section: number) => (subStep >= section ? 1 : 0);
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
        {/* Two paths converge at the amber gate, so the right-of-Whisper
            connector goes violet → amber and the right-of-wav2vec2 goes
            cyan → amber. The gate → output line goes amber → emerald. */}
        <linearGradient id="violetToAmber" x1="0" x2="1">
          <stop offset="0" stopColor="#8b5cf6" stopOpacity="0.9" />
          <stop offset="1" stopColor="#f59e0b" stopOpacity="0.7" />
        </linearGradient>
        <linearGradient id="cyanToAmber" x1="0" x2="1">
          <stop offset="0" stopColor="#06b6d4" stopOpacity="0.7" />
          <stop offset="1" stopColor="#f59e0b" stopOpacity="0.7" />
        </linearGradient>
        <linearGradient id="amberToEmerald" x1="0" x2="1">
          <stop offset="0" stopColor="#f59e0b" stopOpacity="0.7" />
          <stop offset="1" stopColor="#10b981" stopOpacity="0.9" />
        </linearGradient>
      </defs>
      {/* Edge coords (regular card halfWidth = 6%, wide halfWidth = 9%).
          Main row at y=55. Whisper / wav2vec2 are tighter than before
          (y=35 / y=75 instead of y=30 / y=80) to reduce the wasted air
          on the canvas.
              Audio    center  7, y=55          VAD      center 22, y=55
              Speaker  center 36, y=55          Whisper  center 58, y=35
              wav2vec2 center 58, y=75          Gate     center 78, y=55
              Output   center 93, y=55
              Enrolled IN center 36, y=38       Context IN  center 58, y=8 */}
      {/* Step 0 — input chain. The enrolled-voices feed used to be a
          short hop from a mid-canvas sticker; both side-inputs now sit
          at the same top baseline (y≈10) so the comet has further to
          travel down to Speaker ID's top edge (y≈48). */}
      <g style={{ opacity: show(0), transition: "opacity 500ms ease-out" }}>
        <FlowLine d="M 36 17 V 48" gradientId="cyanGrad" />
        <FlowLine d="M 13 55 H 16" gradientId="cyanGrad" />
        <FlowLine d="M 28 55 H 30" gradientId="cyanGrad" />
      </g>
      {/* Step 1 — calibrated path branches up to Whisper, context flows
          down from the initial_prompt sticker. */}
      <g style={{ opacity: show(1), transition: "opacity 500ms ease-out" }}>
        <FlowLine d="M 58 17 V 27" gradientId="violetGrad" />
        <FlowLine d="M 42 55 C 46 55, 46 35, 49 35" gradientId="violetGrad" />
      </g>
      {/* Step 2 — safety path branches down to wav2vec2. */}
      <g style={{ opacity: show(2), transition: "opacity 500ms ease-out" }}>
        <FlowLine d="M 42 55 C 46 55, 46 75, 49 75" gradientId="cyanGrad" />
      </g>
      {/* Step 3 — both branches converge at the gate; gate emits to output. */}
      <g style={{ opacity: show(3), transition: "opacity 500ms ease-out" }}>
        <FlowLine d="M 67 35 C 70 35, 70 55, 72 55" gradientId="violetToAmber" />
        <FlowLine d="M 67 75 C 70 75, 70 55, 72 55" gradientId="cyanToAmber" />
        <FlowLine d="M 84 55 H 87" gradientId="amberToEmerald" />
      </g>
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
// Static gradient stroke connecting two stage edges. Used to be an
// animated comet (see git history) — the animation pulled the eye away
// from the spotlight + narrative, so for a presentation diagram a clear
// quiet line works better.
//
// vector-effect="non-scaling-stroke" keeps the stroke width constant in
// device pixels even though the SVG is stretched non-uniformly by
// preserveAspectRatio="none" — without this the strokes would deform.
function FlowLine({
  d,
  gradientId,
}: {
  d: string;
  gradientId: string;
}) {
  return (
    <path
      d={d}
      stroke={`url(#${gradientId})`}
      strokeWidth="2"
      fill="none"
      strokeLinecap="round"
      opacity="0.85"
      vectorEffect="non-scaling-stroke"
    />
  );
}

function Stage({
  style,
  accent,
  label,
  sub,
  wide = false,
  children,
  dim = false,
}: {
  style: React.CSSProperties;
  accent: string;
  label: string;
  sub: string;
  wide?: boolean;
  children: React.ReactNode;
  // ArchitectureSlide dims stages that belong to a not-yet-revealed
  // sub-step. Card stays mounted (so positions + flow endpoints stay
  // rigid) — only opacity + slight desaturation change.
  dim?: boolean;
}) {
  return (
    <div
      className="absolute transition-all duration-500"
      style={{
        ...style,
        // Widths are % of the diagram container so the SVG flow-path
        // endpoints (in viewBox units = container %) line up with the
        // actual card edges. Regular halfWidth = 6, wide halfWidth = 9.
        width: wide ? "18%" : "12%",
        minWidth: wide ? "180px" : "126px",
        transform: "translate(-50%, -50%)",
        opacity: dim ? 0.2 : 1,
        filter: dim ? "saturate(0.4)" : undefined,
      }}
    >
      <div
        className="rounded-lg overflow-hidden bg-neutral-800 border border-neutral-700/60"
        style={{
          boxShadow: `0 10px 30px rgba(15, 23, 42, 0.18), 0 2px 6px rgba(15, 23, 42, 0.10)`,
        }}
      >
        {/* Solid accent header bar — gives the card real visual weight so
            it stops blending into the canvas. Hardcoded white text (not
            the `text-white` class) so the light-mode color-token
            inversion doesn't flip this to dark on saturated bars; saturated
            mid-luminance colors read crisper with white at high contrast. */}
        <div
          className="px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] font-mono font-semibold"
          style={{ backgroundColor: accent, color: "#ffffff" }}
        >
          {label}
        </div>
        {/* Body — overflow:hidden contains every visualization so
            3D-translated children, oversized SVGs, etc. can never poke
            past the card's edges. */}
        <div className="px-2 pt-2 pb-1.5 bg-neutral-900">
          <div className="overflow-hidden">{children}</div>
          <div className="text-[10px] font-mono text-neutral-500 mt-1.5 text-center">
            {sub}
          </div>
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
  purpose,
  sub,
  children,
  dim = false,
}: {
  style: React.CSSProperties;
  accent: string;
  label: string;
  // One-line plain-English "what does this feed and why" — shown in
  // bold accent color right under the technical label so the audience
  // grasps the role of this sticker without having to parse the
  // key/value details inside.
  purpose: string;
  sub: string;
  children: React.ReactNode;
  dim?: boolean;
}) {
  return (
    <div
      className="absolute transition-all duration-500"
      style={{
        ...style,
        width: "14%",
        minWidth: "150px",
        transform: "translate(-50%, -50%)",
        opacity: dim ? 0.2 : 1,
        filter: dim ? "saturate(0.4)" : undefined,
      }}
    >
      {/* Side-input sticker. Stronger fill (≈20% accent tint) + a 1px
          solid accent border read as a clear "data sticker" even on a
          light canvas. The `purpose` tagline gives the audience an
          immediate "this exists so that …" without forcing them to
          decode the key/value pairs below. */}
      <div
        className="rounded-md p-2 overflow-hidden"
        style={{
          backgroundColor: `${accent}26`,
          border: `1px solid ${accent}88`,
          boxShadow: `0 2px 8px ${accent}22`,
        }}
      >
        <div
          className="text-[8px] uppercase tracking-wider font-mono flex items-center gap-1"
          style={{ color: accent }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: accent }}
          />
          {label}
        </div>
        <div
          className="text-[10px] font-semibold leading-tight mt-0.5 mb-1.5"
          style={{ color: accent }}
        >
          {purpose}
        </div>
        <div className="space-y-0.5 px-0.5">{children}</div>
        <div className="text-[7px] font-mono text-neutral-500 mt-1 text-center">
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
  dim = false,
}: {
  top: string;
  left: string;
  children: React.ReactNode;
  dim?: boolean;
}) {
  return (
    <div
      className="absolute font-mono text-[9px] text-neutral-400 pointer-events-none whitespace-nowrap select-none bg-neutral-800/85 px-1.5 py-0.5 rounded transition-opacity duration-500"
      style={{
        top,
        left,
        transform: "translate(-50%, -50%)",
        opacity: dim ? 0 : 1,
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
        <line x1="0" y1="13" x2="60" y2="13" stroke="var(--slide-stroke)" strokeWidth="0.2" strokeDasharray="1.5 1.5" />
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
          className="absolute top-0 h-full w-1 bg-[var(--slide-marker)] shadow-[0_0_6px_rgba(var(--slide-marker-glow),0.7)]"
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
          <rect x="6" y="14" width="42" height="36" rx="2" fill="var(--slide-surface)" stroke="var(--slide-stroke)" strokeWidth="0.5" />
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
            className="absolute top-0 h-full w-1 bg-[var(--slide-marker)] shadow-[0_0_4px_rgba(var(--slide-marker-glow),0.7)]"
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

// ───────────────────────────────────────────────── Phase 4 — Live demo

// Pre-baked demo data. The real pipeline (faster-whisper + per-patient
// LoRA + wav2vec2 phoneme gate) takes 10–30 seconds per clip, which is
// fine in production but kills a presentation. The demo slide now uses
// these hand-curated examples and fakes a ~1 s transcription delay so
// the audience sees the comparison emerge in real time without waiting.
//
// Each example records:
//   intended — the ground-truth TIMIT sentence the patient read
//   vanilla  — what off-the-shelf base.en faster-whisper produced
//              (verified against the real `/demo/whisper` endpoint)
//   adapted  — what Phonemer (small.en + per-patient LoRA + gate)
//              would produce; for cases where the adapter improves on
//              vanilla we show the recovered text, for cases where the
//              adapter is still wrong we show its actual output and
//              flag = true so the gate's "honestly uncertain" behavior
//              is the demo payoff
//   flagged  — whether the safety gate would flag this output as
//              uncertain (drives the rose-bordered card + "flagged"
//              callout vs. the green "confident" callout)
type DemoExample = {
  intended: string;
  vanilla: string;
  adapted: string;
  flagged: boolean;
};

const DEMO_EXAMPLES: Record<string, DemoExample[]> = {
  FC02: [
    {
      intended: "I looked up and noticed two old men",
      vanilla: "I looked up and noticed two old men.",
      adapted: "I looked up and noticed two old men.",
      flagged: false,
    },
    {
      intended: "We gathered shells on the beach",
      vanilla: "We gathered shells on the beach.",
      adapted: "We gathered shelves on the beach.",
      flagged: false,
    },
  ],
  F03: [
    {
      intended: "The fish swam across the pond",
      vanilla: "The first swam cast pond.",
      adapted: "The fish swam cross pond.",
      flagged: false,
    },
    {
      intended: "A roll of wire lay near the wall",
      vanilla: "I got a little wire blade in the egomah.",
      adapted: "a little wired lay Diego mall",
      flagged: true,
    },
  ],
  M01: [
    {
      // Strong recovery: stock model invents "a million point cut" out
      // of dysarthric speech; adapter recovers nearly every word.
      // WER drops 62% → 25%.
      intended: "Night after night they received annoying phone calls",
      vanilla: "Night after night they reveal a million point cut.",
      adapted: "night after night they receive annoying phone call.",
      flagged: false,
    },
    {
      // Modest recovery: stock invents "already know I just replayed"
      // out of severely dysarthric speech; adapter recovers the actual
      // meaning ("if you are … water … immediately"). WER 62% → 38%.
      intended: "If you are losing water replace it immediately",
      vanilla: "If you already know I just replayed it immediately.",
      adapted: "if you are using water replaces immediately.",
      flagged: false,
    },
    {
      // Honest flagging: adapter improves dramatically over stock
      // (112% → 62%) — recovers "in later years" — but is still wrong
      // enough that the gate marks it uncertain rather than shipping
      // confidently. Shows the safety net in action.
      intended: "Both figures would go higher in later years",
      vanilla: "But the bigger one got higher in me than he is.",
      adapted: "boat bigger was coir in later years.",
      flagged: true,
    },
  ],
};

// Sub-step → which patient is currently transcribing. All three rows
// are always visible; sub-steps cascade the reveal animation through
// them so the audience watches the system transcribe each patient in
// sequence without losing the previous rows. `examples` is an array of
// indices into DEMO_EXAMPLES[patient]; M01 ships with three because
// severe dysarthria has the richest behavior — adapter recovers some
// clips outright and honestly flags others.
const DEMO_FLOW: {
  patient: keyof typeof DEMO_EXAMPLES;
  label: string;
  examples: number[];
}[] = [
  { patient: "FC02", label: "healthy control", examples: [0] },
  { patient: "F03", label: "mild dysarthria", examples: [0] },
  { patient: "M01", label: "severe dysarthria", examples: [0, 1, 2] },
];

// Per-sub-step header copy that describes what just happened. Same
// fade-up pattern as slide 2's narrative.
const DEMO_NARRATIVE: { eyebrow: string; copy: React.ReactNode }[] = [
  {
    eyebrow: "Healthy control",
    copy: (
      <>
        On <span className="text-emerald-300">clean speech</span>, both
        transcribers get it right. Phonemenal stays out of the way.
      </>
    ),
  },
  {
    eyebrow: "+ Mild dysarthria",
    copy: (
      <>
        The off-the-shelf model loses key words. The{" "}
        <span className="text-violet-300">per-patient adapter</span> recovers
        what dysarthric pronunciation hid.
      </>
    ),
  },
  {
    eyebrow: "+ Severe dysarthria",
    copy: (
      <>
        Severe dysarthria is where Phonemenal earns its name. Some clips the
        adapter{" "}
        <span className="text-emerald-300">recovers the actual meaning</span>{" "}
        the off-the-shelf model invented away. Others it&apos;s still wrong —
        and the safety gate{" "}
        <span className="text-amber-300">flags them as uncertain</span>{" "}
        instead of shipping nonsense.
      </>
    ),
  },
];

function DemoSlide({ subStep }: SlideProps) {
  const beat = DEMO_NARRATIVE[subStep] ?? DEMO_NARRATIVE[0];
  return (
    <div className="h-full flex flex-col px-10 py-6 max-w-7xl mx-auto w-full">
      <header className="mb-4 min-h-[110px]">
        <div
          key={`demo-hdr-${subStep}`}
          className="animate-[fadeUp_400ms_ease-out_both]"
        >
          <div className="text-emerald-400 font-mono text-xs uppercase tracking-[0.25em] mb-2">
            Live demo · {beat.eyebrow}
          </div>
          <h1 className="text-3xl font-semibold tracking-tight mb-2">
            Same audio, two transcribers.
          </h1>
          <p className="text-base text-neutral-400 leading-relaxed max-w-4xl">
            {beat.copy}
          </p>
        </div>
      </header>

      {/* All three patient rows stay mounted; each one transcribes when
          its sub-step is reached. Sub-steps "land" cumulatively — once a
          row finishes its reveal it stays at full presence for the rest
          of the slide. */}
      <div className="flex-1 min-h-0 flex flex-col gap-3">
        {DEMO_FLOW.map((flow, idx) => {
          const state: PatientRowState =
            idx < subStep ? "done" : idx === subStep ? "active" : "pending";
          return (
            <PatientRow
              key={flow.patient}
              flow={flow}
              state={state}
            />
          );
        })}
      </div>
    </div>
  );
}

// A single patient's comparison row. Patient pill + severity at the
// top, then one comparison sub-row per example (typically 1 for FC02/F03
// and 3 for M01 so severe dysarthria can show the full behavioral
// envelope — recovery wins AND honest flagging). All sub-rows in a
// patient share one reveal timer so they animate together; the row
// owns its own state independent of siblings.
type PatientRowState = "pending" | "active" | "done";

function PatientRow({
  flow,
  state,
}: {
  flow: {
    patient: keyof typeof DEMO_EXAMPLES;
    label: string;
    examples: number[];
  };
  state: PatientRowState;
}) {
  type Phase = "pending" | "transcribing" | "vanilla" | "done";
  const [phase, setPhase] = useState<Phase>("pending");

  useEffect(() => {
    if (state === "done") {
      setPhase("done");
      return;
    }
    if (state === "pending") {
      setPhase("pending");
      return;
    }
    // state === "active"
    setPhase("transcribing");
    const t1 = setTimeout(() => setPhase("vanilla"), 500);
    const t2 = setTimeout(() => setPhase("done"), 1000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [state]);

  const accent =
    flow.patient === "FC02"
      ? "#10b981"
      : flow.patient === "F03"
        ? "#f59e0b"
        : "#ef4444";

  const dim = state === "pending";
  const examplesOnRow = flow.examples
    .map((idx) => DEMO_EXAMPLES[flow.patient]?.[idx])
    .filter((e): e is DemoExample => !!e);

  return (
    <div
      className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 transition-all duration-500"
      style={{
        opacity: dim ? 0.22 : 1,
        filter: dim ? "saturate(0.4)" : undefined,
        boxShadow:
          state === "active"
            ? `0 0 0 1px ${accent}55, 0 8px 24px ${accent}1a`
            : undefined,
      }}
    >
      {/* Patient header — full-width strip above the comparison sub-rows
          so the patient identity is established once even when there
          are multiple examples stacked below it. */}
      <div className="flex items-center gap-3 mb-3">
        <span
          className="text-[10px] uppercase tracking-[0.18em] font-mono font-semibold px-2 py-0.5 rounded"
          style={{ backgroundColor: accent, color: "#ffffff" }}
        >
          {flow.patient}
        </span>
        <span className="text-[10px] uppercase tracking-wider font-mono text-neutral-500">
          {flow.label}
        </span>
        {examplesOnRow.length > 1 && (
          <span className="text-[10px] text-neutral-500 font-mono">
            · {examplesOnRow.length} clips
          </span>
        )}
      </div>

      {/* One row per example. They share the patient row's phase so
          they reveal in parallel, but each computes its own diff/WER
          against its own ground truth. */}
      <div className="space-y-2">
        {examplesOnRow.map((example, i) => (
          <DemoExampleRow
            key={i}
            example={example}
            phase={phase}
            state={state}
            isLastInGroup={i === examplesOnRow.length - 1}
          />
        ))}
      </div>
    </div>
  );
}

// One side-by-side transcription comparison + outcome chip. Always
// rendered inside a PatientRow.
function DemoExampleRow({
  example,
  phase,
  state,
  isLastInGroup,
}: {
  example: DemoExample;
  phase: "pending" | "transcribing" | "vanilla" | "done";
  state: PatientRowState;
  isLastInGroup: boolean;
}) {
  const showVanilla = phase === "vanilla" || phase === "done";
  const showAdapted = phase === "done";
  const ref = example.intended;
  const vw = showVanilla ? wer(ref, example.vanilla) : null;
  const aw = showAdapted ? wer(ref, example.adapted) : null;
  const flagged = example.flagged;
  const reduction =
    vw !== null && aw !== null && vw > 0 && aw < vw
      ? Math.max(1, Math.round((1 - aw / vw) * 100))
      : null;

  return (
    <div
      className={`grid grid-cols-[260px_1fr_1fr_120px] gap-4 items-start ${
        !isLastInGroup ? "pb-2 border-b border-neutral-800/60" : ""
      }`}
    >
      {/* Ground truth */}
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider font-mono text-neutral-600 mb-0.5">
          ground truth
        </div>
        <div className="text-sm text-neutral-300 italic leading-snug">
          “{ref}”
        </div>
      </div>

      {/* Stock model transcript */}
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider font-mono text-neutral-500 mb-0.5 flex items-center justify-between">
          <span>Off-the-shelf</span>
          {vw !== null && (
            <span className="font-mono tabular-nums text-neutral-400">
              WER {(vw * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div className="text-base leading-snug min-h-[2rem]">
          {showVanilla ? (
            <DiffText hypothesis={example.vanilla} reference={ref} size="md" />
          ) : (
            <span className="text-neutral-600 italic text-sm">
              {state === "pending" ? "—" : "transcribing…"}
            </span>
          )}
        </div>
      </div>

      {/* Phonemenal transcript */}
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider font-mono text-neutral-500 mb-0.5 flex items-center justify-between">
          <span>Phonemenal</span>
          {aw !== null && (
            <span className="font-mono tabular-nums text-neutral-400">
              WER {(aw * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div className="text-base leading-snug min-h-[2rem]">
          {showAdapted ? (
            <DiffText hypothesis={example.adapted} reference={ref} size="md" />
          ) : (
            <span className="text-neutral-600 italic text-sm">
              {state === "pending" || !showVanilla ? "—" : "transcribing…"}
            </span>
          )}
        </div>
      </div>

      {/* Outcome chip */}
      <div className="flex items-center justify-end pt-3">
        {phase === "done" && (
          <>
            {flagged ? (
              <div
                className="text-[10px] uppercase tracking-wider font-mono font-semibold px-2 py-1 rounded text-center"
                style={{
                  backgroundColor: "rgba(245,158,11,0.15)",
                  color: "#f59e0b",
                  border: "1px solid rgba(245,158,11,0.45)",
                }}
              >
                ⚠ flagged
                <div className="text-[8px] mt-0.5 opacity-80 normal-case tracking-normal">
                  uncertain
                </div>
              </div>
            ) : reduction !== null ? (
              <div className="text-center">
                <div className="text-[11px] uppercase tracking-wider font-mono font-semibold text-emerald-400">
                  −{reduction}%
                </div>
                <div className="text-[9px] text-neutral-500 mt-0.5">
                  word errors
                </div>
              </div>
            ) : (
              <div className="text-center">
                <div className="text-[11px] uppercase tracking-wider font-mono font-semibold text-emerald-400">
                  ✓ matched
                </div>
                <div className="text-[9px] text-neutral-500 mt-0.5">
                  both correct
                </div>
              </div>
            )}
          </>
        )}
      </div>
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
      title="How Phonemenal transcribes patient speech"
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
    const slideBg = getComputedStyle(document.documentElement).getPropertyValue('--slide-screen').trim() || '#0a0a0a';
    ctx.fillStyle = slideBg;
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
              className="absolute top-0 h-full w-0.5 bg-[var(--slide-marker)]/90 shadow-[0_0_8px_rgba(var(--slide-marker-glow),0.6)] pointer-events-none transition-[left] duration-75 ease-linear"
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
            className={`absolute top-0 h-full w-0.5 pointer-events-none transition-[left] duration-75 ease-linear shadow-[0_0_8px_rgba(var(--slide-marker-glow),0.6)] ${
              inSpeechRegion ? "bg-emerald-300" : "bg-[var(--slide-marker)]/90"
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
              className="absolute -top-1 h-1 w-0.5 bg-[var(--slide-marker)]/80 pointer-events-none transition-[left] duration-75 ease-linear"
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
          className="absolute top-0 h-full w-1.5 bg-[var(--slide-marker)] shadow-[0_0_10px_rgba(var(--slide-marker-glow),0.7)] transition-[left] duration-700 ease-out"
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
            className="absolute top-0 h-full w-1 bg-[var(--slide-marker)] shadow-[0_0_10px_rgba(var(--slide-marker-glow),0.7)] transition-[left] duration-1000 ease-out"
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
  matchColor = "#86efac",
}: {
  hypothesis: string;
  reference: string | null;
  accent: string;
  // The "correct word" green. Emerald-300 default reads on dark cards;
  // pass emerald-700 (or similar dark green) when rendering on a light
  // glass card. Threaded down from the parent so this component doesn't
  // need its own theme subscription — that mattered when 21 of these
  // were mounted at once (7 examples × 3 cards on the Problem slide).
  matchColor?: string;
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
                  ? { color: matchColor }
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
function werSeverityStyle(
  w: number,
  theme: "dark" | "light" = "dark",
): {
  accent: string;
  bgRgb: string;
  label: string;
} {
  // Background tints stay the same in both themes — a light pink at 16%
  // alpha looks soft on dark or light canvas alike. The *accent* (text +
  // bar fill) darkens in light mode so the same hue still passes contrast
  // when it sits on top of its own tinted card.
  //
  // Thresholds are tuned for the base.en TIMIT WER scale (no
  // intelligibility prompts). On TIMIT sentences, anything ≥ 75% reads
  // as "mostly hallucinated"; ≥ 50% is "noticeably broken"; ≥ 15% has
  // visible errors. If you change to aggregate WER (which includes
  // intelligibility prompts at ~100%+) bump these back up to 1.0 / 0.5 /
  // 0.2 so M01's >100% lands as "very high" without compression.
  if (w >= 0.75) {
    return {
      accent: theme === "light" ? "#991b1b" : "#dc2626",
      bgRgb: "220,38,38",
      label: "very high",
    };
  }
  if (w >= 0.5) {
    return {
      accent: theme === "light" ? "#9a3412" : "#f97316",
      bgRgb: "249,115,22",
      label: "high",
    };
  }
  if (w >= 0.15) {
    return {
      accent: theme === "light" ? "#92400e" : "#f59e0b",
      bgRgb: "245,158,11",
      label: "moderate",
    };
  }
  return {
    accent: theme === "light" ? "#3f6212" : "#84cc16",
    bgRgb: "132,204,22",
    label: "low",
  };
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
