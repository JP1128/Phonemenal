"use client";

import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  ReactNode,
} from "react";

/**
 * Single button system used across the app. The size scale is fixed so
 * buttons sitting next to each other in any row line up to one of four
 * canonical heights:
 *
 *   xs  →  h-6 / 24px   pills, status chips
 *   sm  →  h-7 / 28px   secondary actions in compact contexts (thread rows, dropdowns, header)
 *   md  →  h-9 / 36px   primary actions in cards (active utterance feedback)
 *   lg  →  h-11 / 44px  hero CTAs (mic bar)
 *
 * Intent picks color / chrome:
 *   primary    emerald  (confirm / commit)
 *   warning    amber    (correct / draft attention)
 *   danger     rose     (stop / destructive)
 *   secondary  neutral  (default outlined button)
 *   ghost      bare     (link-like; subtle hover)
 *   chip       neutral  (no border, mono uppercase — for status pills)
 */

export type ButtonSize = "xs" | "sm" | "md" | "lg";
export type ButtonIntent =
  | "primary"
  | "warning"
  | "danger"
  | "secondary"
  | "ghost"
  | "chip";
export type ButtonShape = "pill" | "round" | "square";

const SIZE: Record<ButtonSize, string> = {
  xs: "h-6 px-2 text-[10px]",
  sm: "h-7 px-3 text-xs",
  md: "h-9 px-4 text-sm",
  lg: "h-11 px-6 text-sm",
};

const ICON_SIZE: Record<ButtonSize, string> = {
  xs: "h-6 w-6 text-[10px]",
  sm: "h-7 w-7 text-xs",
  md: "h-9 w-9 text-sm",
  lg: "h-11 w-11 text-sm",
};

const INTENT: Record<ButtonIntent, string> = {
  primary:
    "bg-emerald-600/85 hover:bg-emerald-500 text-white border border-emerald-400/40",
  warning:
    "bg-amber-600/85 hover:bg-amber-500 text-white border border-amber-400/40",
  danger:
    "bg-rose-600/85 hover:bg-rose-500 text-white border border-rose-400/40",
  secondary:
    "bg-white/[0.04] hover:bg-white/[0.09] border border-white/15 text-white/85",
  ghost:
    "border border-transparent hover:bg-white/[0.06] text-white/75 hover:text-white",
  chip:
    "bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-white/70 font-mono uppercase tracking-wider",
};

const SHAPE: Record<ButtonShape, string> = {
  pill: "rounded-full",
  round: "rounded-md",
  square: "rounded",
};

const BASE =
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shrink-0";

function classes({
  size,
  intent,
  shape,
  icon,
  className,
}: {
  size: ButtonSize;
  intent: ButtonIntent;
  shape: ButtonShape;
  icon: boolean;
  className: string;
}): string {
  const sizeClass = icon ? ICON_SIZE[size] : SIZE[size];
  return `${BASE} ${sizeClass} ${INTENT[intent]} ${SHAPE[shape]} ${className}`;
}

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  size?: ButtonSize;
  intent?: ButtonIntent;
  shape?: ButtonShape;
  icon?: boolean;
  children?: ReactNode;
};

export function Button({
  size = "md",
  intent = "secondary",
  shape = "pill",
  icon = false,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      className={classes({ size, intent, shape, icon, className })}
    >
      {children}
    </button>
  );
}

export type ButtonLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  size?: ButtonSize;
  intent?: ButtonIntent;
  shape?: ButtonShape;
  icon?: boolean;
  children?: ReactNode;
};

export function ButtonLink({
  size = "md",
  intent = "secondary",
  shape = "pill",
  icon = false,
  className = "",
  children,
  ...rest
}: ButtonLinkProps) {
  return (
    <a
      {...rest}
      className={classes({ size, intent, shape, icon, className })}
    >
      {children}
    </a>
  );
}
