"use client";

import { useEffect, useRef, useState } from "react";
import type { AccessibilitySettings } from "../lib/accessibility";

type Props = {
  settings: AccessibilitySettings;
  onChange: (next: AccessibilitySettings) => void;
};

export function AccessibilityPanel({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Display settings"
        aria-expanded={open}
        className={`inline-flex items-center justify-center h-7 px-2.5 rounded-full text-xs font-mono tracking-wider transition-colors border ${
          open
            ? "bg-white/10 border-white/20 text-white"
            : "bg-transparent border-transparent text-white/70 hover:text-white hover:bg-white/[0.06]"
        }`}
      >
        Aa
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Display settings"
          className="absolute top-9 right-0 z-40 w-72 p-4 rounded-lg shadow-2xl bg-neutral-950/95 backdrop-blur-md border border-white/10 space-y-3 text-sm"
        >
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Display</h3>
            <button
              onClick={() => setOpen(false)}
              className="opacity-70 hover:opacity-100"
              aria-label="Close display settings"
            >
              ✕
            </button>
          </div>

          <label className="block">
            <span className="text-xs opacity-70">
              Text size: {settings.fontSize}px
            </span>
            <input
              type="range"
              min={14}
              max={48}
              step={2}
              value={settings.fontSize}
              onChange={(e) =>
                onChange({ ...settings, fontSize: Number(e.target.value) })
              }
              className="w-full"
            />
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={settings.highContrast}
              onChange={(e) =>
                onChange({ ...settings, highContrast: e.target.checked })
              }
            />
            <span>High contrast</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={settings.dyslexiaFont}
              onChange={(e) =>
                onChange({ ...settings, dyslexiaFont: e.target.checked })
              }
            />
            <span>Dyslexia-friendly font</span>
          </label>
        </div>
      )}
    </div>
  );
}
