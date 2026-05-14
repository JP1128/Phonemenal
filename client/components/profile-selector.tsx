"use client";

import { useEffect, useRef, useState } from "react";
import { ControlBadge, SeverityBadge } from "./severity-badge";
import type { CalibratedSpeaker, Profile } from "../lib/profile";
import { profileLabel } from "../lib/profile";
import { isControlSpeaker, severityFor } from "../lib/torgo-speakers";

export type ProfileSelectorProps = {
  profile: Profile;
  onChange: (p: Profile) => void;
  /** Enriched entries (id, label, kind) from ``GET /adapters``. */
  calibratedSpeakers: CalibratedSpeaker[];
};

/**
 * Dropdown that picks the active patient profile. Renders the current
 * profile label as a header chip; clicking opens a menu with all
 * calibrated speakers, a single Demo TORGO entry, and a routing link to
 * /calibrate for new enrollment.
 */
export function ProfileSelector({
  profile,
  onChange,
  calibratedSpeakers,
}: ProfileSelectorProps) {
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

  const kindAccent =
    profile.kind === "calibrated"
      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-100"
      : "border-amber-400/30 bg-amber-400/10 text-amber-100";

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={`Active profile: ${profileLabel(profile, calibratedSpeakers)}`}
        className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full text-[10px] font-mono uppercase tracking-wider border whitespace-nowrap shrink-0 transition-colors ${kindAccent} hover:brightness-125`}
      >
        <span className="opacity-70">profile:</span>
        <span className="font-medium">
          {profileLabel(profile, calibratedSpeakers)}
        </span>
        {profile.kind === "calibrated" && (
          <SpeakerKindBadge id={profile.speakerId} />
        )}
        <Caret open={open} />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute top-9 left-0 z-40 w-60 rounded-lg shadow-2xl bg-neutral-950/95 backdrop-blur-md border border-white/10 overflow-hidden text-sm"
        >
          <Section label="Calibrated">
            {calibratedSpeakers.length === 0 ? (
              <div className="px-3 py-1.5 text-xs opacity-50 italic">
                no calibrated speakers yet
              </div>
            ) : (
              calibratedSpeakers.map((s) => {
                const active =
                  profile.kind === "calibrated" && profile.speakerId === s.id;
                return (
                  <ProfileMenuItem
                    key={s.id}
                    active={active}
                    accent="emerald"
                    onClick={() => {
                      onChange({ kind: "calibrated", speakerId: s.id });
                      setOpen(false);
                    }}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className={
                          s.kind === "torgo" ? "font-mono" : "font-medium"
                        }
                      >
                        {s.label}
                      </span>
                      {s.kind === "patient" && s.label !== s.id && (
                        <span className="text-[9px] opacity-40 font-mono">
                          {s.id}
                        </span>
                      )}
                      <SpeakerKindBadge id={s.id} />
                    </span>
                  </ProfileMenuItem>
                );
              })
            )}
          </Section>

          <Divider />

          <Section label="Demo">
            <ProfileMenuItem
              active={profile.kind === "demo"}
              accent="amber"
              onClick={() => {
                onChange({ kind: "demo" });
                setOpen(false);
              }}
            >
              TORGO uncalibrated
            </ProfileMenuItem>
          </Section>

          <Divider />

          <a
            href="/calibrate"
            role="menuitem"
            className="block px-3 py-2 text-sm text-sky-200 hover:bg-white/[0.06]"
          >
            + New profile
            <span className="ml-2 text-[10px] opacity-50">
              calibrate a new patient
            </span>
          </a>
        </div>
      )}
    </div>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="py-1">
      <div className="px-3 py-1 text-[9px] tracking-[0.25em] font-mono opacity-50 uppercase">
        {label}
      </div>
      {children}
    </div>
  );
}

function Divider() {
  return <div className="h-px bg-white/10" />;
}

function ProfileMenuItem({
  active,
  accent,
  onClick,
  children,
}: {
  active: boolean;
  accent: "emerald" | "amber";
  onClick: () => void;
  children: React.ReactNode;
}) {
  const activeClass =
    accent === "emerald"
      ? "bg-emerald-400/10 text-emerald-200"
      : "bg-amber-400/10 text-amber-200";
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      className={`w-full text-left px-3 py-1.5 text-sm hover:bg-white/[0.06] flex items-center justify-between ${
        active ? activeClass : ""
      }`}
    >
      <span>{children}</span>
      {active && (
        <span className="text-[9px] opacity-70 uppercase tracking-wider">
          active
        </span>
      )}
    </button>
  );
}

// Renders the appropriate badge for a TORGO speaker id:
//   - a `SeverityBadge` when the speaker is dysarthric (mild/moderate/severe)
//   - a `ControlBadge` when the id matches the FC* or MC* control pattern
//   - nothing otherwise
function SpeakerKindBadge({ id }: { id: string }) {
  const severity = severityFor(id);
  if (severity) return <SeverityBadge severity={severity} />;
  if (isControlSpeaker(id)) return <ControlBadge />;
  return null;
}

function Caret({ open }: { open: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      fill="none"
      style={{
        transform: open ? "rotate(180deg)" : "rotate(0)",
        transition: "transform 180ms",
      }}
      aria-hidden
    >
      <path
        d="M2 4 L5 7 L8 4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
