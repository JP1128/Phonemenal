"use client";

import { useState } from "react";
import type { AccessibilitySettings } from "../lib/accessibility";

type Props = {
  settings: AccessibilitySettings;
  onChange: (next: AccessibilitySettings) => void;
};

export function AccessibilityPanel({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-4 right-4 z-30">
      {open ? (
        <div className="rounded-lg shadow-2xl bg-neutral-900 border border-neutral-700 p-4 w-72 space-y-3 text-sm">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Display</h3>
            <button
              onClick={() => setOpen(false)}
              className="opacity-70 hover:opacity-100"
              aria-label="Close accessibility panel"
            >
              ✕
            </button>
          </div>

          <label className="block">
            <span className="text-xs opacity-70">Text size: {settings.fontSize}px</span>
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
      ) : (
        <button
          onClick={() => setOpen(true)}
          aria-label="Display settings"
          className="rounded-full w-10 h-10 bg-neutral-900 border border-neutral-700 hover:bg-neutral-800 shadow-lg"
        >
          Aa
        </button>
      )}
    </div>
  );
}
