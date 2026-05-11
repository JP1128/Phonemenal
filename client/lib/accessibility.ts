"use client";

import { useEffect, useState } from "react";

export type AccessibilitySettings = {
  fontSize: number;        // base px
  highContrast: boolean;
  dyslexiaFont: boolean;
};

const DEFAULTS: AccessibilitySettings = {
  fontSize: 18,
  highContrast: false,
  dyslexiaFont: false,
};

const STORAGE_KEY = "phonemer-a11y";

export function useAccessibility() {
  const [settings, setSettings] = useState<AccessibilitySettings>(DEFAULTS);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setSettings({ ...DEFAULTS, ...JSON.parse(raw) });
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings, hydrated]);

  const styleVars: React.CSSProperties = {
    // Used as a multiplier on the page's local sizing.
    ["--phonemer-base-size" as string]: `${settings.fontSize}px`,
    ["--phonemer-text-color" as string]: settings.highContrast ? "#ffffff" : "#f5f5f5",
    ["--phonemer-bg-color" as string]: settings.highContrast ? "#000000" : "#0a0a0a",
    ["--phonemer-accent" as string]: settings.highContrast ? "#ffd34d" : "#4ecdc4",
    fontFamily: settings.dyslexiaFont
      ? '"OpenDyslexic", "Lexend", "Comic Sans MS", system-ui, sans-serif'
      : 'ui-sans-serif, system-ui',
  };

  return {
    settings,
    setSettings,
    styleVars,
  };
}
