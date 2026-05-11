"use client";

import { useEffect, useState } from "react";
import {
  MicPermissionState,
  queryMicPermission,
} from "../lib/audio-devices";

export type MicState =
  | "idle"
  | "connecting"
  | "recording"
  | "blocked"
  | "error";

type MicButtonProps = {
  state: MicState;
  level: number;            // 0..1 RMS
  errorMessage?: string;
  onStart: () => void | Promise<void>;
  onStop: () => void | Promise<void>;
};

export function MicButton({
  state,
  level,
  errorMessage,
  onStart,
  onStop,
}: MicButtonProps) {
  const [perm, setPerm] = useState<MicPermissionState>("unknown");

  useEffect(() => {
    queryMicPermission().then(setPerm);
  }, []);

  const isLive = state === "recording";
  const isBusy = state === "connecting";

  let label = "🎙 Start mic";
  let tone = "bg-emerald-700 hover:bg-emerald-600 border-emerald-500/40";
  if (isBusy) {
    label = "⏳ Starting…";
    tone = "bg-amber-700 border-amber-500/40 cursor-wait";
  } else if (isLive) {
    label = "⏹ Stop mic";
    tone = "bg-rose-700 hover:bg-rose-600 border-rose-500/40";
  } else if (state === "blocked") {
    label = "🔒 Mic blocked";
    tone = "bg-neutral-800 border-rose-500/40 text-rose-300 cursor-not-allowed";
  } else if (state === "error") {
    label = "⚠ Retry";
    tone = "bg-rose-900/60 hover:bg-rose-800 border-rose-500/40";
  }

  return (
    <div className="inline-flex flex-col items-stretch gap-1">
      <div className="flex items-center gap-2">
        <button
          disabled={isBusy || state === "blocked"}
          onClick={() => (isLive ? onStop() : onStart())}
          className={`px-3 py-1.5 rounded text-xs border transition-colors ${tone} disabled:opacity-60`}
        >
          {label}
        </button>
        <LevelMeter level={level} active={isLive} />
        <PermissionDot state={perm} />
      </div>
      {errorMessage && (
        <span className="text-[10px] text-rose-300 max-w-[14rem] leading-tight">
          {errorMessage}
        </span>
      )}
    </div>
  );
}

function LevelMeter({ level, active }: { level: number; active: boolean }) {
  // Power-curve scaling so quiet speech is still visible.
  const v = Math.min(1, Math.pow(level * 6, 0.7));
  const segments = 12;
  const lit = Math.max(0, Math.round(v * segments));
  return (
    <div className="flex gap-[2px] items-end h-4">
      {Array.from({ length: segments }).map((_, i) => {
        const isLit = active && i < lit;
        const color =
          i < segments * 0.6
            ? "#10b981"
            : i < segments * 0.85
              ? "#f59e0b"
              : "#ef4444";
        return (
          <div
            key={i}
            className="w-1 rounded-sm"
            style={{
              height: `${30 + (i / segments) * 70}%`,
              background: isLit ? color : "rgba(64,64,64,0.6)",
              transition: "background 80ms",
            }}
          />
        );
      })}
    </div>
  );
}

function PermissionDot({ state }: { state: MicPermissionState }) {
  const color =
    state === "granted"
      ? "#10b981"
      : state === "denied"
        ? "#ef4444"
        : state === "prompt"
          ? "#f59e0b"
          : "#6b7280";
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full"
      title={`mic permission: ${state}`}
      style={{ background: color, boxShadow: `0 0 6px ${color}77` }}
    />
  );
}
