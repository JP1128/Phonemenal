"use client";

import { useEffect, useRef, useState } from "react";

export type ActionMenuProps = {
  showPipeline: boolean;
  showTorgo: boolean;
  showContext: boolean;
  onTogglePipeline: () => void;
  onToggleTorgo: () => void;
  onToggleContext: () => void;
};

export function ActionMenu({
  showPipeline,
  showTorgo,
  showContext,
  onTogglePipeline,
  onToggleTorgo,
  onToggleContext,
}: ActionMenuProps) {
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
    <div ref={ref} className="fixed top-5 right-5 z-40">
      <button
        type="button"
        aria-expanded={open}
        aria-label="Menu"
        onClick={() => setOpen((v) => !v)}
        className={`group flex items-center justify-center w-10 h-10 rounded-full border transition-colors ${
          open
            ? "bg-white/10 border-white/20"
            : "bg-white/[0.03] border-white/10 hover:bg-white/[0.07]"
        }`}
      >
        <Caret open={open} />
      </button>

      <div
        className={`absolute top-12 right-0 min-w-[200px] origin-top-right rounded-xl border border-white/10 bg-neutral-950/90 backdrop-blur-md shadow-2xl shadow-black/40 overflow-hidden transition-all ${
          open
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "opacity-0 -translate-y-1 pointer-events-none"
        }`}
        style={{ transitionDuration: "160ms" }}
        role="menu"
      >
        <ToggleItem
          label="pipeline"
          active={showPipeline}
          onClick={() => {
            onTogglePipeline();
            setOpen(false);
          }}
        />
        <ToggleItem
          label="TORGO"
          active={showTorgo}
          onClick={() => {
            onToggleTorgo();
            setOpen(false);
          }}
        />
        <ToggleItem
          label="context"
          active={showContext}
          onClick={() => {
            onToggleContext();
            setOpen(false);
          }}
        />
        <div className="h-px bg-white/10" />
        <LinkItem href="/debug" label="go to debug" />
        <LinkItem href="/audio" label="diagnose audio" />
      </div>
    </div>
  );
}

function ToggleItem({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitemcheckbox"
      aria-checked={active}
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-white/[0.06] transition-colors text-left"
    >
      <span className="opacity-50 text-[11px] font-mono w-9 shrink-0">
        {active ? "hide" : "show"}
      </span>
      <span className="flex-1">{label}</span>
      <span
        className={`w-1.5 h-1.5 rounded-full transition-colors ${
          active ? "bg-teal-300" : "bg-white/15"
        }`}
      />
    </button>
  );
}

function LinkItem({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      role="menuitem"
      className="block px-4 py-2.5 text-sm hover:bg-white/[0.06] transition-colors opacity-80"
    >
      {label}
      <span className="ml-2 opacity-40">→</span>
    </a>
  );
}

function Caret({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      style={{
        transform: open ? "rotate(180deg)" : "rotate(0)",
        transition: "transform 200ms",
      }}
    >
      <circle cx="3" cy="7" r="1.1" fill="currentColor" />
      <circle cx="7" cy="7" r="1.1" fill="currentColor" />
      <circle cx="11" cy="7" r="1.1" fill="currentColor" />
    </svg>
  );
}
