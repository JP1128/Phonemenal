"use client";

// 4-phase pipeline animation, decoupled from the /viz demo page so it can
// be re-used on the home page driven by live translation events. Pass a
// VizUtterance + a unique `runKey` to trigger a fresh play-through.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  VizPhoneme,
  VizProbDistribution,
  VizUtterance,
  VizWord,
} from "../lib/viz-types";

type Phase =
  | "idle"
  | "scanning"
  | "phonemes"
  | "probabilities"
  | "translation"
  | "done";

const TEAL = "#4ecdc4";
const RED = "#ef4444";
const BLUE = "#6c8cff";
const GREEN = "#10b981";

function scanDurationFor(durationMs: number): number {
  return Math.max(2500, Math.min(durationMs, 5500));
}

export type PipelineVizProps = {
  utterance: VizUtterance | null;
  /** Increment to retrigger the animation for the same utterance. */
  runKey?: number;
  /** Auto-start when the utterance prop changes. Default true. */
  autoPlay?: boolean;
  /** Hide section labels for an embedded look. */
  compact?: boolean;
};

export function PipelineViz({
  utterance,
  runKey,
  autoPlay = true,
  compact = false,
}: PipelineVizProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [scanProgress, setScanProgress] = useState(0);
  const [revealedProbs, setRevealedProbs] = useState(0);
  const [resolvedProbs, setResolvedProbs] = useState<Set<number>>(new Set());
  const [revealedWords, setRevealedWords] = useState(0);
  const animRef = useRef<number | null>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const reset = useCallback(() => {
    setPhase("idle");
    setScanProgress(0);
    setRevealedProbs(0);
    setResolvedProbs(new Set());
    setRevealedWords(0);
    if (animRef.current !== null) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    timersRef.current.forEach((t) => clearTimeout(t));
    timersRef.current = [];
  }, []);

  const run = useCallback(() => {
    if (!utterance) return;
    reset();
    setPhase("scanning");
    const scanDuration = scanDurationFor(utterance.durationMs);
    const start = performance.now();

    const tick = () => {
      const t = (performance.now() - start) / scanDuration;
      if (t >= 1) {
        setScanProgress(1);
        setPhase("phonemes");
        timersRef.current.push(
          setTimeout(() => {
            setPhase("probabilities");
            utterance.probDistributions.forEach((_, i) => {
              timersRef.current.push(
                setTimeout(() => {
                  setRevealedProbs((n) => Math.max(n, i + 1));
                  timersRef.current.push(
                    setTimeout(() => {
                      setResolvedProbs((prev) => {
                        const next = new Set(prev);
                        next.add(i);
                        return next;
                      });
                    }, 700),
                  );
                }, i * 1500),
              );
            });
            const probsTotalMs =
              utterance.probDistributions.length * 1500 + 700;
            timersRef.current.push(
              setTimeout(() => {
                setPhase("translation");
                utterance.words.forEach((_, i) => {
                  timersRef.current.push(
                    setTimeout(() => {
                      setRevealedWords((n) => Math.max(n, i + 1));
                      if (i === utterance.words.length - 1) {
                        timersRef.current.push(
                          setTimeout(() => setPhase("done"), 600),
                        );
                      }
                    }, i * 280),
                  );
                });
              }, probsTotalMs),
            );
          }, 600),
        );
        return;
      }
      setScanProgress(t);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
  }, [reset, utterance]);

  useEffect(() => () => reset(), [reset]);

  // Auto-run on a fresh utterance / runKey.
  useEffect(() => {
    if (!autoPlay) return;
    if (!utterance) {
      reset();
      return;
    }
    run();
  }, [utterance, runKey, autoPlay, run, reset]);

  if (!utterance) {
    return (
      <div className="rounded-lg border border-white/10 p-12 text-center opacity-60 text-sm">
        no translation yet — talk into the mic or stream a TORGO clip
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {!compact && <ContextBar utterance={utterance} />}

      <Section
        title="01 · Acoustic scan"
        active={phase === "scanning" || phase === "phonemes"}
        compact={compact}
      >
        <Waveform utterance={utterance} progress={scanProgress} phase={phase} />
        <Spectrogram utterance={utterance} progress={scanProgress} />
        <Legend
          items={[
            { color: TEAL, label: "normal" },
            { color: RED, label: "distortion" },
            { color: "#374151", label: "unscanned" },
          ]}
        />
      </Section>

      <Section
        title="02 · Phoneme alignment"
        active={
          phase === "phonemes" ||
          phase === "probabilities" ||
          phase === "translation" ||
          phase === "done"
        }
        compact={compact}
      >
        <PhonemeRibbon utterance={utterance} progress={scanProgress} phase={phase} />
        <Legend
          items={[
            { color: TEAL, label: "match" },
            { color: RED, label: "produced (distorted)" },
            { color: GREEN, label: "intended" },
          ]}
        />
      </Section>

      <Section
        title="03 · Probability resolution"
        active={
          phase === "probabilities" ||
          phase === "translation" ||
          phase === "done"
        }
        compact={compact}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {utterance.probDistributions.map((dist, i) => (
            <ProbabilityCard
              key={i}
              dist={dist}
              revealed={i < revealedProbs}
              resolved={resolvedProbs.has(i)}
            />
          ))}
        </div>
      </Section>

      <Section
        title="04 · Translation"
        active={phase === "translation" || phase === "done"}
        compact={compact}
      >
        <TranslationReveal
          utterance={utterance}
          revealedCount={revealedWords}
          done={phase === "done"}
        />
        <Legend
          items={[
            { color: "#e5e7eb", label: "recognized" },
            { color: TEAL, label: "corrected by distortion model" },
            { color: BLUE, label: "context-assisted correction" },
          ]}
        />
      </Section>

      {phase === "done" && (
        <div className="text-center">
          <button
            onClick={run}
            className="px-3 py-1.5 rounded text-xs border border-white/10 hover:bg-white/5"
          >
            ↻ Replay animation
          </button>
        </div>
      )}

      <style jsx>{`
        @keyframes pulse-red {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.4); }
        }
      `}</style>
    </div>
  );
}

// ============================================================ chrome

function ContextBar({ utterance }: { utterance: VizUtterance }) {
  return (
    <div className="space-y-2">
      {utterance.speakerContext && (
        <div
          className="rounded-md px-3 py-1.5 flex items-baseline gap-3 text-xs"
          style={{
            background:
              "linear-gradient(90deg, rgba(108, 140, 255, 0.12) 0%, rgba(108, 140, 255, 0.04) 100%)",
            border: "1px solid rgba(108, 140, 255, 0.25)",
          }}
        >
          <span className="text-[9px] tracking-[0.2em] font-mono text-blue-300/80">
            CONTEXT
          </span>
          <span className="text-blue-200/90">
            <strong>{utterance.speakerContext.speaker}:</strong>{" "}
            "{utterance.speakerContext.text}"
          </span>
        </div>
      )}
      <div
        className="rounded-md px-3 py-1.5 flex items-baseline gap-3 text-xs"
        style={{
          background: "rgba(255, 255, 255, 0.03)",
          border: "1px solid rgba(255, 255, 255, 0.08)",
        }}
      >
        <span className="text-[9px] tracking-[0.2em] font-mono opacity-50">
          RAW
        </span>
        <span className="font-mono opacity-80">"{utterance.raw}"</span>
      </div>
    </div>
  );
}

function Section({
  title,
  active,
  children,
  compact,
}: {
  title: string;
  active: boolean;
  children: React.ReactNode;
  compact?: boolean;
}) {
  return (
    <section
      className="rounded-lg p-3 transition-all duration-500"
      style={{
        background: "rgba(255, 255, 255, 0.025)",
        border: `1px solid ${active ? "rgba(78, 205, 196, 0.3)" : "rgba(255, 255, 255, 0.05)"}`,
        boxShadow: active
          ? "0 0 24px rgba(78, 205, 196, 0.06) inset, 0 0 12px rgba(78, 205, 196, 0.05)"
          : "none",
        opacity: active ? 1 : 0.55,
      }}
    >
      {!compact && (
        <div className="text-[10px] tracking-[0.25em] font-mono opacity-60 mb-2">
          {title}
        </div>
      )}
      {children}
    </section>
  );
}

function Legend({
  items,
}: {
  items: { color: string; label: string }[];
}) {
  return (
    <div className="flex gap-4 mt-2 text-[10px] font-mono opacity-50">
      {items.map((it, i) => (
        <span key={i} className="flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-1.5 rounded-sm"
            style={{ background: it.color }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

// ============================================================ waveform

const WAVEFORM_BARS = 96;

function Waveform({
  utterance,
  progress,
  phase,
}: {
  utterance: VizUtterance;
  progress: number;
  phase: Phase;
}) {
  const bars = useMemo(() => buildWaveform(utterance), [utterance]);
  const distortionZones = useMemo(() => {
    const total = utterance.durationMs || 1;
    return utterance.phonemes
      .filter((p) => !p.match)
      .map((p) => ({
        start: p.time / total,
        end: Math.min(1, (p.time + 200) / total),
        type: p.distortionType ?? "distortion",
      }));
  }, [utterance]);

  return (
    <div className="relative h-24 mb-1">
      {distortionZones.map((z, i) => (
        <div
          key={i}
          className="absolute inset-y-0 rounded-sm"
          style={{
            left: `${z.start * 100}%`,
            width: `${(z.end - z.start) * 100}%`,
            background:
              progress >= z.start
                ? "linear-gradient(180deg, rgba(239, 68, 68, 0.18) 0%, rgba(239, 68, 68, 0.04) 100%)"
                : "transparent",
            transition: "background 0.4s",
          }}
        />
      ))}
      <div className="absolute inset-0 flex items-center gap-[1px]">
        {bars.map((amp, i) => {
          const t = i / WAVEFORM_BARS;
          const scanned = t <= progress;
          const inDistortion = distortionZones.some(
            (z) => t >= z.start && t <= z.end,
          );
          const color = !scanned
            ? "rgba(75, 85, 99, 0.6)"
            : inDistortion
              ? RED
              : TEAL;
          return (
            <div
              key={i}
              className="flex-1 rounded-[2px]"
              style={{
                height: `${Math.max(4, amp * 100)}%`,
                background: color,
                boxShadow:
                  scanned && inDistortion
                    ? `0 0 8px ${RED}`
                    : scanned
                      ? `0 0 4px ${TEAL}66`
                      : "none",
                transition: "background 0.18s, box-shadow 0.18s",
              }}
            />
          );
        })}
      </div>
      {phase !== "idle" && progress < 1 && (
        <div
          className="absolute inset-y-0 w-[2px]"
          style={{
            left: `${progress * 100}%`,
            background: BLUE,
            boxShadow: `0 0 12px ${BLUE}, 0 0 24px ${BLUE}77`,
          }}
        />
      )}
    </div>
  );
}

function buildWaveform(utt: VizUtterance): number[] {
  const seed = hashString(utt.id);
  const out: number[] = [];
  for (let i = 0; i < WAVEFORM_BARS; i++) {
    const t = i / WAVEFORM_BARS;
    const envelope = Math.sin(Math.PI * t) ** 0.6;
    const noise = pseudoRand(seed + i * 9301) * 0.6 + 0.3;
    let amp = envelope * noise;
    for (const p of utt.phonemes) {
      const pt = p.time / Math.max(1, utt.durationMs);
      const d = Math.abs(t - pt);
      if (d < 0.04) {
        amp = Math.min(1, amp + (1 - d / 0.04) * (p.match ? 0.1 : 0.25));
      }
    }
    out.push(amp);
  }
  return out;
}

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

function pseudoRand(n: number): number {
  let x = n + 0x9e3779b9;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 0xffffffff;
}

// ============================================================ spectrogram

const SPEC_COLS = 96;
const SPEC_ROWS = 22;

function Spectrogram({
  utterance,
  progress,
}: {
  utterance: VizUtterance;
  progress: number;
}) {
  const grid = useMemo(() => buildSpectrogram(utterance), [utterance]);
  const distortionRanges = useMemo(() => {
    const total = utterance.durationMs || 1;
    return utterance.phonemes
      .filter((p) => !p.match)
      .map((p) => ({
        start: p.time / total,
        end: Math.min(1, (p.time + 200) / total),
      }));
  }, [utterance]);

  return (
    <div
      className="relative grid w-full h-16 rounded-sm overflow-hidden"
      style={{
        gridTemplateColumns: `repeat(${SPEC_COLS}, 1fr)`,
        gridTemplateRows: `repeat(${SPEC_ROWS}, 1fr)`,
        gap: 0,
        marginTop: 8,
      }}
    >
      {grid.map((cell, idx) => {
        const col = idx % SPEC_COLS;
        const t = col / SPEC_COLS;
        const scanned = t <= progress;
        const inDistortion = distortionRanges.some(
          (r) => t >= r.start && t <= r.end,
        );
        let color: string;
        if (!scanned) {
          const v = Math.round(20 + cell * 30);
          color = `rgb(${v}, ${v}, ${v + 4})`;
        } else if (inDistortion) {
          const v = cell;
          color = `rgba(239, 68, 68, ${0.15 + v * 0.85})`;
        } else {
          const v = cell;
          color = `rgba(78, 205, 196, ${0.08 + v * 0.55})`;
        }
        return (
          <div key={idx} style={{ background: color, transition: "background 0.2s" }} />
        );
      })}
    </div>
  );
}

function buildSpectrogram(utt: VizUtterance): number[] {
  const seed = hashString(utt.id + "spec");
  const cells: number[] = [];
  for (let r = 0; r < SPEC_ROWS; r++) {
    for (let c = 0; c < SPEC_COLS; c++) {
      const t = c / SPEC_COLS;
      const f = r / SPEC_ROWS;
      const freqEnvelope = Math.exp(-2.5 * f) * 0.7 + 0.3;
      const timeEnvelope = Math.sin(Math.PI * t) ** 0.4;
      const noise = pseudoRand(seed + r * 1117 + c * 31);
      let v = freqEnvelope * timeEnvelope * (0.4 + noise * 0.6);
      for (const p of utt.phonemes) {
        const pt = p.time / Math.max(1, utt.durationMs);
        if (Math.abs(t - pt) < 0.03) {
          v += (p.match ? 0.05 : 0.15) * (1 - Math.abs(t - pt) / 0.03);
        }
      }
      cells.push(Math.min(1, v));
    }
  }
  return cells;
}

// ============================================================ phonemes

function PhonemeRibbon({
  utterance,
  progress,
  phase,
}: {
  utterance: VizUtterance;
  progress: number;
  phase: Phase;
}) {
  const total = Math.max(1, utterance.durationMs);
  const everythingRevealed =
    phase === "phonemes" ||
    phase === "probabilities" ||
    phase === "translation" ||
    phase === "done";

  return (
    <div className="space-y-2 py-1">
      <RibbonRow
        label="PRODUCED"
        phonemes={utterance.phonemes}
        progress={progress}
        total={total}
        revealAll={everythingRevealed}
        which="produced"
      />
      <RibbonRow
        label="INTENDED"
        phonemes={utterance.phonemes}
        progress={progress}
        total={total}
        revealAll={everythingRevealed}
        which="intended"
      />
    </div>
  );
}

function RibbonRow({
  label,
  phonemes,
  progress,
  total,
  revealAll,
  which,
}: {
  label: string;
  phonemes: VizPhoneme[];
  progress: number;
  total: number;
  revealAll: boolean;
  which: "produced" | "intended";
}) {
  return (
    <div>
      <div className="text-[9px] tracking-[0.25em] font-mono opacity-40 mb-1">
        {label}
      </div>
      <div className="flex items-center gap-1 flex-wrap">
        {phonemes.map((p, i) => {
          const t = p.time / total;
          const visible = revealAll || t <= progress;
          const symbol = which === "produced" ? p.produced : p.intended;
          const isMismatch = !p.match;
          const baseColor = !isMismatch
            ? TEAL
            : which === "produced"
              ? RED
              : GREEN;
          return (
            <PhonemeChip
              key={i}
              symbol={symbol || "—"}
              color={baseColor}
              visible={visible}
              distortionType={
                isMismatch && which === "produced"
                  ? p.distortionType
                  : undefined
              }
              isMismatch={isMismatch}
            />
          );
        })}
      </div>
    </div>
  );
}

function PhonemeChip({
  symbol,
  color,
  visible,
  distortionType,
  isMismatch,
}: {
  symbol: string;
  color: string;
  visible: boolean;
  distortionType?: string;
  isMismatch: boolean;
}) {
  return (
    <div
      className="flex flex-col items-center"
      style={{
        opacity: visible ? 1 : 0.15,
        transform: visible ? "translateY(0)" : "translateY(4px)",
        transition: "opacity 0.25s, transform 0.25s",
      }}
    >
      <div
        className="px-2 py-0.5 rounded text-[10px] font-mono relative"
        style={{
          background: `${color}22`,
          border: `1px solid ${color}66`,
          color,
          boxShadow: isMismatch && visible ? `0 0 6px ${color}55` : "none",
          minWidth: "1.5rem",
          textAlign: "center",
        }}
      >
        {symbol}
        {isMismatch && visible && (
          <span
            className="absolute -top-1 -right-1 w-1.5 h-1.5 rounded-full"
            style={{
              background: RED,
              animation: "pulse-red 1.4s ease-in-out infinite",
            }}
          />
        )}
      </div>
      {distortionType && visible && (
        <div className="text-[9px] mt-0.5 opacity-60 max-w-[6rem] text-center leading-tight">
          {distortionType}
        </div>
      )}
    </div>
  );
}

// ============================================================ probabilities

function ProbabilityCard({
  dist,
  revealed,
  resolved,
}: {
  dist: VizProbDistribution;
  revealed: boolean;
  resolved: boolean;
}) {
  return (
    <div
      className="rounded-md p-3 transition-all duration-500"
      style={{
        background: "rgba(255, 255, 255, 0.03)",
        border: `1px solid ${
          resolved
            ? "rgba(16, 185, 129, 0.4)"
            : revealed
              ? "rgba(78, 205, 196, 0.25)"
              : "rgba(255, 255, 255, 0.06)"
        }`,
        opacity: revealed ? 1 : 0.25,
        transform: revealed ? "translateY(0)" : "translateY(8px)",
        boxShadow: resolved ? "0 0 16px rgba(16, 185, 129, 0.15)" : "none",
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-mono opacity-80 truncate">{dist.label}</div>
        {resolved && (
          <span className="text-xs flex items-center gap-1" style={{ color: GREEN }}>
            ✓
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        {dist.candidates.map((c, i) => {
          const isCorrect = c.phoneme === dist.correctedTo;
          const widthPct = resolved
            ? isCorrect
              ? 100
              : c.prob * 35
            : c.prob * 100;
          const color = resolved
            ? isCorrect
              ? GREEN
              : "#4b5563"
            : ["#4ecdc4", "#6c8cff", "#a78bfa", "#f59e0b"][i % 4];
          const opacity = resolved && !isCorrect ? 0.4 : 1;
          return (
            <div key={c.phoneme} className="flex items-center gap-2">
              <div
                className="font-mono text-xs w-10 text-right opacity-80"
                style={{ color: resolved ? (isCorrect ? GREEN : "#9ca3af") : "#d1d5db" }}
              >
                {c.phoneme}
              </div>
              <div className="flex-1 h-2.5 rounded-sm overflow-hidden bg-white/5">
                <div
                  className="h-full rounded-sm"
                  style={{
                    width: `${widthPct}%`,
                    background: color,
                    opacity,
                    boxShadow: resolved && isCorrect ? `0 0 10px ${GREEN}66` : "none",
                    transition:
                      "width 0.7s cubic-bezier(0.2, 0.8, 0.2, 1), background 0.5s, opacity 0.5s",
                  }}
                />
              </div>
              <div
                className="font-mono text-[10px] w-10 text-right opacity-70 tabular-nums"
                style={{ color: resolved && isCorrect ? GREEN : "#9ca3af" }}
              >
                {(c.prob * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>
      <div
        className="mt-2 text-[10px] leading-snug font-mono opacity-70"
        style={{
          opacity: resolved ? 0.85 : 0,
          transition: "opacity 0.4s 0.3s",
        }}
      >
        {dist.reason}
      </div>
    </div>
  );
}

// ============================================================ translation

function TranslationReveal({
  utterance,
  revealedCount,
  done,
}: {
  utterance: VizUtterance;
  revealedCount: number;
  done: boolean;
}) {
  const overallConf = useMemo(() => {
    const xs = utterance.words.map((w) => w.conf);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
  }, [utterance]);

  return (
    <div className="mt-1">
      <div className="flex flex-wrap gap-x-3 gap-y-2 items-end mb-2">
        {utterance.words.map((w, i) => (
          <RevealedWord key={i} word={w} visible={i < revealedCount} />
        ))}
      </div>
      <div
        className="text-[11px] font-mono opacity-70 transition-opacity duration-500"
        style={{ opacity: done ? 0.7 : 0 }}
      >
        overall confidence{" "}
        <span style={{ color: overallConf > 0.8 ? GREEN : overallConf > 0.6 ? "#f59e0b" : RED }}>
          {(overallConf * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function RevealedWord({
  word,
  visible,
}: {
  word: VizWord;
  visible: boolean;
}) {
  const color = word.contextUsed
    ? BLUE
    : word.corrected
      ? TEAL
      : "#e5e7eb";
  const showOriginal = word.corrected && word.raw !== word.translated;
  return (
    <div
      className="flex flex-col items-center"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(8px)",
        transition: "opacity 0.4s, transform 0.4s",
      }}
    >
      <div
        className="text-2xl"
        style={{
          color,
          textDecoration: word.corrected ? "underline" : "none",
          textDecorationColor: `${color}88`,
          textUnderlineOffset: 5,
          fontWeight: 500,
          textShadow: word.contextUsed
            ? `0 0 18px ${BLUE}44`
            : word.corrected
              ? `0 0 14px ${TEAL}33`
              : "none",
        }}
      >
        {word.translated}
      </div>
      {showOriginal && (
        <div className="text-[10px] font-mono opacity-50 mt-0.5">
          ← {word.raw}
        </div>
      )}
    </div>
  );
}
