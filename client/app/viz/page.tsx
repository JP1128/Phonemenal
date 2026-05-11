"use client";

// /viz — same animation as the home page's pipeline view, but driven by
// hardcoded sample utterances so judges can see the explanation without
// needing a calibrated patient or a TORGO clip in flight.

import { useEffect, useMemo, useState } from "react";
import { PipelineViz } from "../../components/pipeline-viz";
import { VIZ_SAMPLES } from "../../lib/viz-samples";

export default function VizPage() {
  const [sampleIndex, setSampleIndex] = useState(0);
  const [runKey, setRunKey] = useState(0);
  const sample = useMemo(() => VIZ_SAMPLES[sampleIndex], [sampleIndex]);

  // Replay button increments the run key.
  useEffect(() => {
    setRunKey((k) => k + 1);
  }, [sampleIndex]);

  return (
    <main
      className="min-h-screen text-neutral-100 px-6 py-6"
      style={{
        background:
          "radial-gradient(circle at 30% 0%, #11141d 0%, #06080e 60%, #04050a 100%)",
        fontFamily: "ui-sans-serif, system-ui",
      }}
    >
      <header className="flex flex-wrap items-center gap-3 mb-6">
        <h1 className="text-xl font-semibold tracking-wide">
          Phonemer · Pipeline (canned demo)
        </h1>
        <span className="text-xs opacity-50 font-mono">
          live audio → phonemes → distortion model → LLM → text
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs opacity-60">sample:</span>
          {VIZ_SAMPLES.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setSampleIndex(i)}
              className={`px-2.5 py-1 rounded text-xs ${
                i === sampleIndex
                  ? "bg-teal-500/20 text-teal-300 border border-teal-500/40"
                  : "bg-white/5 hover:bg-white/10 border border-transparent"
              }`}
            >
              {s.id}
            </button>
          ))}
          <button
            onClick={() => setRunKey((k) => k + 1)}
            className="ml-3 px-4 py-1.5 rounded text-xs font-semibold bg-teal-500/20 hover:bg-teal-500/30 text-teal-200 border border-teal-500/40"
          >
            ▶ Replay
          </button>
          <a
            href="/"
            className="ml-2 text-xs px-2 py-0.5 rounded bg-white/5 hover:bg-white/10"
          >
            ← Home
          </a>
        </div>
      </header>

      <div className="max-w-6xl mx-auto">
        <PipelineViz utterance={sample} runKey={runKey} autoPlay />
      </div>
    </main>
  );
}
