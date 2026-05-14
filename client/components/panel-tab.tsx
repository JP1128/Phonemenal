"use client";

export type PanelSide = "left" | "right" | "bottom";

export type PanelTabProps = {
  label: string;
  side: PanelSide;
  onClick: () => void;
  /**
   * Optional unread count. When > 0, renders a small rose pill on the tab
   * so the user knows something new has landed behind the closed panel.
   */
  badge?: number;
};

export function PanelTab({ label, side, onClick, badge }: PanelTabProps) {
  const showBadge = typeof badge === "number" && badge > 0;
  const badgeText = showBadge && badge! > 9 ? "9+" : String(badge);
  const ariaLabel = showBadge
    ? `Show ${label} panel (${badge} new)`
    : `Show ${label} panel`;

  if (side === "bottom") {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-label={ariaLabel}
        className="shrink-0 flex items-center justify-center gap-2 px-4 py-1.5 border-t border-white/5 bg-black/20 hover:bg-white/[0.04] text-[10px] tracking-[0.25em] font-mono opacity-60 hover:opacity-100 uppercase transition-colors"
      >
        <Chevron direction="up" />
        <span>{label}</span>
        {showBadge && <Badge text={badgeText} />}
      </button>
    );
  }

  const chevronDir = side === "left" ? "right" : "left";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      title={showBadge ? `${badge} new` : undefined}
      className={`group shrink-0 relative flex flex-col items-center gap-3 w-7 py-3 bg-black/20 hover:bg-white/[0.04] transition-colors ${
        side === "left" ? "border-r border-white/5" : "border-l border-white/5"
      }`}
    >
      <span className="opacity-60 group-hover:opacity-100 transition-opacity">
        <Chevron direction={chevronDir} />
      </span>
      <span
        className="text-[10px] tracking-[0.25em] font-mono opacity-60 group-hover:opacity-100 uppercase transition-opacity"
        style={{
          writingMode: "vertical-rl",
          transform: side === "left" ? "rotate(180deg)" : "none",
        }}
      >
        {label}
      </span>
      {showBadge && (
        <span
          className={`absolute top-1 ${side === "left" ? "right-1" : "left-1"} block w-2 h-2 rounded-full bg-rose-500 ring-2 ring-rose-300/30 animate-pulse`}
          aria-hidden
        />
      )}
    </button>
  );
}

function Badge({ text }: { text: string }) {
  return (
    <span
      className="inline-flex items-center justify-center min-w-[16px] h-4 px-1 rounded-full text-[9px] font-mono bg-rose-500 text-white border border-rose-300/40 shadow-md shadow-rose-900/40"
      aria-hidden
    >
      {text}
    </span>
  );
}

function Chevron({
  direction,
}: {
  direction: "up" | "down" | "left" | "right";
}) {
  const rotation = {
    up: "rotate(-90deg)",
    down: "rotate(90deg)",
    left: "rotate(180deg)",
    right: "rotate(0deg)",
  }[direction];
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      style={{ transform: rotation }}
    >
      <path
        d="M3 1.5 L7 5 L3 8.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
