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
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setSettings({ ...DEFAULTS, ...JSON.parse(raw) });
    } catch {
      /* ignore */
    }
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "light") setTheme("light");
    const obs = new MutationObserver(() => {
      const t = document.documentElement.getAttribute("data-theme");
      setTheme(t === "light" ? "light" : "dark");
    });
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    setHydrated(true);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings, hydrated]);

  // High-contrast pushes toward pure black/white per theme; standard mode
  // uses near-black/near-white so we don't blow out the eye. Light mode
  // gets a faint radial wash so the empty/listening states don't read as
  // a sterile sheet of white — high-contrast skips the gradient since
  // accessibility users typically want flat surfaces.
  const isLight = theme === "light";
  const textColor = settings.highContrast
    ? (isLight ? "#000000" : "#ffffff")
    : (isLight ? "#0a0a0a" : "#f5f5f5");
  const bgColor = settings.highContrast
    ? (isLight ? "#ffffff" : "#000000")
    : isLight
      ? // Cool slate-tinted off-white. Warm cream was fighting the cool
        // cyan / violet palette on /how-it-works; a slightly-blue base
        // sits comfortably under those accents instead of clashing.
        "radial-gradient(ellipse 100% 80% at 50% -10%, #e2e8f0 0%, rgba(226,232,240,0) 70%), #f8fafc"
      : "#0a0a0a";

  const styleVars: React.CSSProperties = {
    // Used as a multiplier on the page's local sizing.
    ["--phonemer-base-size" as string]: `${settings.fontSize}px`,
    ["--phonemer-text-color" as string]: textColor,
    ["--phonemer-bg-color" as string]: bgColor,
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
