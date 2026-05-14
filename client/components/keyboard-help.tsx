"use client";

import { useEffect, useRef, useState } from "react";

const SHORTCUTS: { keys: string[]; label: string }[] = [
  { keys: ["Space"], label: "Toggle microphone" },
  { keys: ["Enter"], label: "Confirm latest utterance" },
  { keys: ["E"], label: "Correct latest utterance" },
  { keys: ["Esc"], label: "Cancel correction" },
  { keys: ["F"], label: "Toggle focus mode" },
];

export function KeyboardHelp() {
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
        aria-label="Keyboard shortcuts"
        aria-expanded={open}
        title="Keyboard shortcuts"
        className={`inline-flex items-center justify-center h-7 w-7 rounded-full text-xs font-mono transition-colors border ${
          open
            ? "bg-white/10 border-white/20 text-white"
            : "bg-transparent border-transparent text-white/70 hover:text-white hover:bg-white/[0.06]"
        }`}
      >
        ?
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Keyboard shortcuts"
          className="absolute top-9 right-0 z-40 w-64 p-3 rounded-lg shadow-2xl bg-neutral-950/95 backdrop-blur-md border border-white/10 text-sm"
        >
          <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase mb-2">
            shortcuts
          </h3>
          <ul className="space-y-1.5">
            {SHORTCUTS.map((s) => (
              <li key={s.label} className="flex items-center justify-between">
                <span className="opacity-80 text-xs">{s.label}</span>
                <span className="flex gap-1">
                  {s.keys.map((k) => (
                    <kbd
                      key={k}
                      className="px-1.5 py-0.5 rounded border border-white/15 bg-white/[0.04] text-[10px] font-mono"
                    >
                      {k}
                    </kbd>
                  ))}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[10px] opacity-40 leading-snug">
            Shortcuts pause while a text field is focused.
          </p>
        </div>
      )}
    </div>
  );
}
