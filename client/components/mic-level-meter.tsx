"use client";

const BAR_COUNT = 6;
const BARS = Array.from({ length: BAR_COUNT }, (_, i) => i);

/**
 * Threshold-based VU meter. Each bar lights up when the compressed mic level
 * crosses its threshold; lower bars are emerald (good), upper bars amber/rose
 * (peaking). Faded out when not active so the layout stays stable.
 */
export function MicLevelMeter({
  level,
  active,
}: {
  level: number;
  active: boolean;
}) {
  // Same compression curve as the existing in-button mic indicator so the
  // meter matches what the user already sees on the start button.
  const v = active ? Math.min(1, Math.pow(Math.max(0, level) * 6, 0.7)) : 0;
  return (
    <div
      className="flex items-end gap-0.5 h-4 w-12 shrink-0 transition-opacity"
      style={{ opacity: active ? 1 : 0.25 }}
      aria-hidden
      title={
        active ? `mic level ${(v * 100).toFixed(0)}%` : "mic off"
      }
    >
      {BARS.map((i) => {
        const threshold = i / BAR_COUNT;
        const isOn = v > threshold;
        const heightPct = 25 + (i / (BAR_COUNT - 1)) * 75;
        const color =
          i < BAR_COUNT - 2
            ? "#10b981"
            : i < BAR_COUNT - 1
              ? "#fbbf24"
              : "#ef4444";
        return (
          <span
            key={i}
            className="w-0.5 rounded-full"
            style={{
              height: `${heightPct}%`,
              background: isOn ? color : "rgba(255,255,255,0.1)",
              transition: "background 80ms",
            }}
          />
        );
      })}
    </div>
  );
}
