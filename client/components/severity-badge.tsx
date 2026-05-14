"use client";

import type { Severity } from "../lib/torgo-speakers";

const PALETTE: Record<Severity, string> = {
  mild: "bg-emerald-500/15 text-emerald-200 border-emerald-400/30",
  moderate: "bg-amber-500/15 text-amber-200 border-amber-400/30",
  severe: "bg-rose-500/15 text-rose-200 border-rose-400/30",
};

export function SeverityBadge({
  severity,
  className = "",
}: {
  severity: Severity;
  className?: string;
}) {
  return (
    <span
      className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border whitespace-nowrap ${PALETTE[severity]} ${className}`}
      title={`${severity} dysarthria`}
    >
      {severity}
    </span>
  );
}

export function ControlBadge({ className = "" }: { className?: string }) {
  return (
    <span
      className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border whitespace-nowrap bg-sky-500/15 text-sky-200 border-sky-400/30 ${className}`}
      title="Non-dysarthric reference speaker"
    >
      control
    </span>
  );
}
