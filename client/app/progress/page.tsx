"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AccessibilityPanel } from "../../components/accessibility-panel";
import { ThemeToggle } from "../../components/theme-toggle";
import { useAccessibility } from "../../lib/accessibility";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";

type Snapshot = {
  calibration_id: number | null;
  week_start: number;
  n_utterances: number;
  diagonal_mass: number;
  confirmed_rate: number;
  avg_overall_confidence: number;
};

type ShiftAlert = {
  flagged: boolean;
  z_score: number;
  current_diagonal_rate: number;
  baseline_mean: number;
  baseline_sigma: number;
  n_observations: number;
} | null;

type ProgressResponse = {
  calibration_id: number | null;
  snapshots: Snapshot[];
  shift: ShiftAlert;
};

type VocabItem = {
  id: number;
  kind: string;
  term: string;
  notes?: string | null;
  boost: number;
};

const KINDS = ["name", "medication", "place", "preferred", "other"];

export default function ProgressPage() {
  const a11y = useAccessibility();

  const [data, setData] = useState<ProgressResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vocab, setVocab] = useState<VocabItem[]>([]);
  const [newKind, setNewKind] = useState<string>("name");
  const [newTerm, setNewTerm] = useState("");

  const refresh = useCallback(async (rebuild = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${SERVER_HTTP}/progress?rebuild=${rebuild ? "true" : "false"}`,
      );
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as ProgressResponse;
      setData(json);
    } catch (err) {
      setError(`could not load progress: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshVocab = useCallback(async () => {
    try {
      const res = await fetch(`${SERVER_HTTP}/vocabulary`);
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setVocab(json.items as VocabItem[]);
    } catch {
      /* ignore — vocabulary is optional */
    }
  }, []);

  useEffect(() => {
    refresh(true);
    refreshVocab();
  }, [refresh, refreshVocab]);

  const addTerm = useCallback(async () => {
    if (!newTerm.trim()) return;
    try {
      await fetch(`${SERVER_HTTP}/vocabulary`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ kind: newKind, term: newTerm.trim() }),
      });
      setNewTerm("");
      refreshVocab();
    } catch (err) {
      setError(`could not save term: ${err}`);
    }
  }, [newKind, newTerm, refreshVocab]);

  const removeTerm = useCallback(
    async (id: number) => {
      try {
        await fetch(`${SERVER_HTTP}/vocabulary/${id}`, { method: "DELETE" });
        refreshVocab();
      } catch (err) {
        setError(`could not delete term: ${err}`);
      }
    },
    [refreshVocab],
  );

  return (
    <main
      className="min-h-screen px-6 py-6"
      style={{
        ...a11y.styleVars,
        background: "var(--phonemer-bg-color)",
        color: "var(--phonemer-text-color)",
      }}
    >
      <header className="flex items-center gap-3 max-w-4xl mx-auto mb-6">
        <h1 className="text-lg font-semibold tracking-wide">
          Phonemenal · Progress
        </h1>
        <button
          onClick={() => refresh(true)}
          className="text-xs px-2 py-0.5 rounded bg-white/5 hover:bg-white/10"
          disabled={loading}
        >
          {loading ? "rebuilding…" : "↻ refresh"}
        </button>
        <div className="ml-auto flex items-center gap-2 text-xs opacity-70">
          <a href="/" className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10">
            ← Home
          </a>
          <a href="/calibrate" className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10">
            Calibrate
          </a>
        </div>
      </header>

      <div className="max-w-4xl mx-auto space-y-6">
        {error && (
          <div className="rounded bg-rose-900/40 px-3 py-2 text-rose-200 text-sm">
            {error}
          </div>
        )}

        {data?.shift?.flagged && (
          <ShiftAlertBanner shift={data.shift} />
        )}

        <section className="rounded-lg border border-white/10 p-5 space-y-4">
          <h2 className="text-sm font-semibold opacity-80">Therapy gain</h2>
          {data && data.snapshots.length === 0 ? (
            <div className="opacity-60 text-sm">
              No data yet — confirm or correct a few translations on{" "}
              <code>/translate</code> and refresh.
            </div>
          ) : (
            <ProgressChart snapshots={data?.snapshots ?? []} />
          )}
          {data?.shift && !data.shift.flagged && (
            <div className="text-[11px] opacity-60 font-mono">
              recent diagonal rate {data.shift.current_diagonal_rate.toFixed(2)}{" "}
              · baseline {data.shift.baseline_mean.toFixed(2)} ±{" "}
              {data.shift.baseline_sigma.toFixed(2)} (z={data.shift.z_score})
            </div>
          )}
        </section>

        <section className="rounded-lg border border-white/10 p-5 space-y-3">
          <h2 className="text-sm font-semibold opacity-80">
            Personal vocabulary
          </h2>
          <p className="text-xs opacity-70 leading-relaxed">
            Names, medications, and preferred terms get a probability boost in
            the LLM's candidate generation. Add anything Phonemenal should
            recognize even when the phoneme evidence is thin.
          </p>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={newKind}
              onChange={(e) => setNewKind(e.target.value)}
              className="bg-neutral-900 border border-white/10 rounded px-2 py-1 text-sm"
            >
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <input
              type="text"
              value={newTerm}
              onChange={(e) => setNewTerm(e.target.value)}
              placeholder="e.g. Maria, lisinopril, the porch"
              className="flex-1 min-w-[14rem] bg-neutral-900 border border-white/10 rounded px-2 py-1 text-sm"
              onKeyDown={(e) => {
                if (e.key === "Enter") addTerm();
              }}
            />
            <button
              onClick={addTerm}
              disabled={!newTerm.trim()}
              className="px-3 py-1.5 rounded text-xs font-semibold disabled:opacity-50"
              style={{ background: "var(--phonemer-accent)", color: "#0a0a0a" }}
            >
              Add
            </button>
          </div>

          {vocab.length === 0 ? (
            <div className="opacity-60 text-sm">no terms added yet</div>
          ) : (
            <ul className="space-y-1">
              {vocab.map((v) => (
                <li
                  key={v.id}
                  className="flex items-center gap-2 bg-neutral-900 rounded px-2 py-1 text-sm"
                >
                  <span className="opacity-50 w-20 text-xs uppercase tracking-wide">
                    {v.kind}
                  </span>
                  <span className="flex-1">{v.term}</span>
                  <span className="opacity-50 text-xs">
                    boost {v.boost.toFixed(1)}
                  </span>
                  <button
                    onClick={() => removeTerm(v.id)}
                    className="opacity-50 hover:opacity-100 hover:text-rose-300"
                    title="Delete"
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <ThemeToggle />
      <AccessibilityPanel
        settings={a11y.settings}
        onChange={a11y.setSettings}
      />
    </main>
  );
}

// ============================================================ chart

function ProgressChart({ snapshots }: { snapshots: Snapshot[] }) {
  const series = useMemo(() => {
    return snapshots.map((s) => ({
      label: weekLabel(s.week_start),
      diagonal: s.diagonal_mass,
      confirmed: s.confirmed_rate,
      confidence: s.avg_overall_confidence,
      n: s.n_utterances,
    }));
  }, [snapshots]);

  if (series.length === 0) return null;

  const w = 720;
  const h = 220;
  const pad = 24;

  const xFor = (i: number) =>
    pad + (i / Math.max(1, series.length - 1)) * (w - pad * 2);
  const yFor = (v: number) => h - pad - v * (h - pad * 2);

  const linePath = (key: keyof (typeof series)[number]) =>
    series
      .map((d, i) => {
        const v = (d[key] as number) || 0;
        return `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)} ${yFor(v).toFixed(1)}`;
      })
      .join(" ");

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-56">
        {/* Background grid */}
        {[0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line
              x1={pad}
              y1={yFor(g)}
              x2={w - pad}
              y2={yFor(g)}
              stroke="rgba(255,255,255,0.06)"
            />
            <text
              x={pad - 4}
              y={yFor(g) + 4}
              textAnchor="end"
              fontSize="10"
              fill="rgba(255,255,255,0.4)"
              fontFamily="ui-monospace, monospace"
            >
              {(g * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        <path d={linePath("diagonal")} stroke="#34d399" strokeWidth="2" fill="none" />
        <path d={linePath("confirmed")} stroke="#60a5fa" strokeWidth="2" fill="none" strokeDasharray="3 3" />
        <path d={linePath("confidence")} stroke="#fbbf24" strokeWidth="2" fill="none" strokeDasharray="1 3" />
        {series.map((d, i) => (
          <g key={i}>
            <circle cx={xFor(i)} cy={yFor(d.diagonal)} r="3" fill="#34d399" />
            <text
              x={xFor(i)}
              y={h - 6}
              textAnchor="middle"
              fontSize="9"
              fill="rgba(255,255,255,0.45)"
              fontFamily="ui-monospace, monospace"
            >
              {d.label}
            </text>
          </g>
        ))}
      </svg>
      <div className="flex gap-4 text-[11px] opacity-70">
        <Legend color="#34d399" label="diagonal mass (P=intended)" />
        <Legend color="#60a5fa" label="confirmed rate" dashed />
        <Legend color="#fbbf24" label="avg confidence" dotted />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        {series.slice(-4).map((d) => (
          <div key={d.label} className="bg-neutral-900 rounded p-2">
            <div className="opacity-60">{d.label}</div>
            <div className="opacity-90">
              n={d.n} · diag {(d.diagonal * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Legend({
  color,
  label,
  dashed,
  dotted,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  dotted?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="22" height="6">
        <line
          x1="0"
          y1="3"
          x2="22"
          y2="3"
          stroke={color}
          strokeWidth="2"
          strokeDasharray={dashed ? "3 3" : dotted ? "1 3" : undefined}
        />
      </svg>
      {label}
    </span>
  );
}

function weekLabel(ts: number): string {
  const d = new Date(ts * 1000);
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${m}/${day}`;
}

// ============================================================ shift

function ShiftAlertBanner({
  shift,
}: {
  shift: NonNullable<ShiftAlert>;
}) {
  return (
    <div
      className="rounded-lg p-4 border"
      style={{
        background: "rgba(239, 68, 68, 0.08)",
        borderColor: "rgba(239, 68, 68, 0.4)",
      }}
    >
      <div className="text-sm font-semibold text-rose-200 mb-1">
        ⚠ Possible neurological shift detected
      </div>
      <div className="text-xs opacity-80">
        Recent diagonal rate {(shift.current_diagonal_rate * 100).toFixed(0)}% is{" "}
        {Math.abs(shift.z_score).toFixed(1)}σ below the running baseline of{" "}
        {(shift.baseline_mean * 100).toFixed(0)}%. This often coincides with a
        change in patient state — consider re-calibrating, and check for new
        symptoms.
      </div>
    </div>
  );
}
