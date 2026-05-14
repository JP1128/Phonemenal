"use client";

import { useEffect, useRef } from "react";

export type PanelResizerProps = {
  axis: "x" | "y";
  onDrag: (delta: number) => void;
  ariaLabel?: string;
};

export function PanelResizer({ axis, onDrag, ariaLabel }: PanelResizerProps) {
  const dragging = useRef(false);
  const lastPos = useRef(0);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const pos = axis === "x" ? e.clientX : e.clientY;
      const delta = pos - lastPos.current;
      lastPos.current = pos;
      if (delta !== 0) onDrag(delta);
    };
    const stop = () => {
      if (!dragging.current) return;
      dragging.current = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", stop);
    window.addEventListener("blur", stop);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", stop);
      window.removeEventListener("blur", stop);
    };
  }, [axis, onDrag]);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    lastPos.current = axis === "x" ? e.clientX : e.clientY;
    document.body.style.userSelect = "none";
    document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";
  };

  const isHorizontal = axis === "x";
  return (
    <div
      role="separator"
      aria-orientation={isHorizontal ? "vertical" : "horizontal"}
      aria-label={ariaLabel}
      onMouseDown={onMouseDown}
      className={
        isHorizontal
          ? "shrink-0 w-1 cursor-col-resize bg-white/5 hover:bg-teal-400/50 active:bg-teal-400/70 transition-colors"
          : "shrink-0 h-1 cursor-row-resize bg-white/5 hover:bg-teal-400/50 active:bg-teal-400/70 transition-colors"
      }
    />
  );
}
