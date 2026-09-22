"use client";

import { useMemo, useState } from "react";

import type { TimedEvent } from "@/lib/types";

interface ExecutionTraceProps {
  events: TimedEvent[];
  isStreaming: boolean;
}

type PointKind = "start" | "step" | "tool" | "ok" | "fail" | "error" | "end";

interface Point {
  label: string;
  detail?: string;
  kind: PointKind;
  at: number;
}

export function ExecutionTrace({ events, isStreaming }: ExecutionTraceProps) {
  const [collapsed, setCollapsed] = useState(false);
  const points = useMemo(() => toPoints(events), [events]);
  const total = points.length > 1 ? points[points.length - 1].at - points[0].at : 0;

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="hidden shrink-0 border-l border-line bg-panel px-3 font-mono text-[11px] tracking-wide text-ink-faint hover:text-ink-dim lg:block"
        title="Show execution trace"
      >
        trace
      </button>
    );
  }

  return (
    <aside className="hidden w-[300px] shrink-0 flex-col border-l border-line bg-panel lg:flex">
      <div className="flex items-center justify-between border-b border-line px-4 py-3.5">
        <div className="flex items-center gap-2">
          <h2 className="font-mono text-[13px] text-ink-dim">execution trace</h2>
          {isStreaming ? <span className="live h-1.5 w-1.5 rounded-full bg-signal" /> : null}
        </div>
        <button
          className="font-mono text-[11px] text-ink-faint hover:text-ink-dim"
          onClick={() => setCollapsed(true)}
          type="button"
        >
          hide
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {points.length === 0 ? (
          <p className="font-mono text-[12px] leading-relaxed text-ink-faint">
            Steps, tool calls, and timings appear here as the agent runs.
          </p>
        ) : (
          <ol className="relative ml-1.5 border-l border-line">
            {points.map((point, i) => {
              const next = points[i + 1];
              const running = !next && isStreaming;
              const duration = next ? next.at - point.at : running ? null : 0;
              return (
                <li key={i} className="settle relative pb-4 pl-5 last:pb-0">
                  <span
                    className={`absolute -left-[5px] top-1 h-2.5 w-2.5 rounded-full ${dotClass(
                      point.kind
                    )} ${running ? "live" : ""}`}
                  />
                  <div className="flex items-baseline justify-between gap-2">
                    <span className={`font-mono text-[12.5px] ${textClass(point.kind)}`}>
                      {point.label}
                    </span>
                    <span className="shrink-0 font-mono text-[11px] text-ink-faint">
                      {duration === null ? "running" : fmt(duration)}
                    </span>
                  </div>
                  {point.detail ? (
                    <p className="mt-0.5 truncate font-mono text-[11px] text-ink-faint">
                      {point.detail}
                    </p>
                  ) : null}
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <div className="border-t border-line px-4 py-3 font-mono text-[11px] text-ink-faint">
        elapsed{" "}
        <span className="text-ink-dim">{points.length > 1 ? fmt(total) : "—"}</span>
        <span className="ml-1 text-ink-faint">(measured client-side)</span>
      </div>
    </aside>
  );
}

function toPoints(events: TimedEvent[]): Point[] {
  // Only the latest turn, to keep the rail focused.
  let start = 0;
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].type === "user") {
      start = i;
      break;
    }
  }
  const slice = events.slice(start);
  const points: Point[] = [];
  for (const event of slice) {
    switch (event.type) {
      case "user":
        points.push({ label: "received", kind: "start", at: event.at });
        break;
      case "step_start":
        points.push({
          label: `step ${event.step}`,
          detail: event.description,
          kind: "step",
          at: event.at
        });
        break;
      case "tool_call":
        points.push({ label: event.tool, detail: "tool call", kind: "tool", at: event.at });
        break;
      case "tool_result":
        points.push({
          label: event.tool ?? "tool",
          detail: event.success ? "returned" : "failed",
          kind: event.success ? "ok" : "fail",
          at: event.at
        });
        break;
      case "error":
        points.push({ label: "error", detail: event.message, kind: "error", at: event.at });
        break;
      case "done":
        points.push({ label: "done", kind: "end", at: event.at });
        break;
      default:
        break;
    }
  }
  return points;
}

function dotClass(kind: PointKind): string {
  switch (kind) {
    case "ok":
      return "bg-ok";
    case "fail":
    case "error":
      return "bg-bad";
    case "tool":
      return "bg-tool";
    case "end":
      return "bg-signal";
    case "step":
      return "bg-ink-dim";
    default:
      return "bg-ink-faint";
  }
}

function textClass(kind: PointKind): string {
  switch (kind) {
    case "ok":
      return "text-ok";
    case "fail":
    case "error":
      return "text-bad";
    case "tool":
      return "text-tool";
    case "end":
      return "text-signal";
    default:
      return "text-ink-dim";
  }
}

function fmt(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
