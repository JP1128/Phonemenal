"use client";

export type LiveState = "idle" | "listening" | "transcribing";

export function LiveStatus({
  state,
  level,
}: {
  state: LiveState;
  level: number;
}) {
  if (state === "idle") return null;

  const isListening = state === "listening";
  return (
    <div
      role="status"
      aria-live="polite"
      className="inline-flex items-center justify-center gap-2 h-6 px-2.5 rounded-full bg-black/30 border border-white/10 whitespace-nowrap shrink-0"
    >
      {isListening ? <LevelMeter level={level} /> : <ThinkingDots />}
      <span
        className={`text-[10px] tracking-[0.2em] font-mono uppercase ${
          isListening ? "opacity-70" : "text-amber-200/90"
        }`}
      >
        {isListening ? "listening" : "transcribing"}
      </span>
    </div>
  );
}

function LevelMeter({ level }: { level: number }) {
  // Same compression curve as MicIndicator so the meter feels consistent
  // with the existing mic-button visualization.
  const pct = Math.min(1, Math.pow(Math.max(0, level) * 6, 0.7));
  return (
    <span className="block w-12 h-1 bg-white/10 rounded-full overflow-hidden">
      <span
        className="block h-full bg-emerald-400/80 rounded-full"
        style={{ width: `${pct * 100}%`, transition: "width 80ms linear" }}
      />
    </span>
  );
}

function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-0.5">
      <span
        className="block w-1 h-1 rounded-full bg-amber-300"
        style={{ animation: "live-dot 1.1s ease-in-out 0s infinite" }}
      />
      <span
        className="block w-1 h-1 rounded-full bg-amber-300"
        style={{ animation: "live-dot 1.1s ease-in-out 0.18s infinite" }}
      />
      <span
        className="block w-1 h-1 rounded-full bg-amber-300"
        style={{ animation: "live-dot 1.1s ease-in-out 0.36s infinite" }}
      />
      <style jsx>{`
        @keyframes live-dot {
          0%,
          100% {
            opacity: 0.25;
            transform: scale(0.85);
          }
          50% {
            opacity: 1;
            transform: scale(1.15);
          }
        }
      `}</style>
    </span>
  );
}

export function DemoBadge() {
  return (
    <span
      title="Streaming a TORGO test clip — not live mic input"
      className="inline-flex items-center justify-center gap-1.5 h-6 px-2 rounded-full text-[9px] tracking-[0.3em] font-mono uppercase border border-amber-400/40 bg-amber-400/10 text-amber-200 whitespace-nowrap shrink-0"
    >
      <span className="block w-1.5 h-1.5 rounded-full bg-amber-300 animate-pulse" />
      demo
    </span>
  );
}
