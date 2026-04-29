"use client";

import { useMemo, useState } from "react";

import type { AgentEvent } from "@/lib/types";

interface ExecutionTraceProps {
  events: AgentEvent[];
}

export function ExecutionTrace({ events }: ExecutionTraceProps) {
  const [collapsed, setCollapsed] = useState(false);
  const entries = useMemo(() => toTraceEntries(events), [events]);
  const total = entries.length ? entries.length * 150 : 0;

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="border-l border-border bg-surface px-3 font-mono text-xs text-muted"
      >
        TRACE
      </button>
    );
  }

  return (
    <aside className="hidden w-[300px] shrink-0 border-l border-border bg-surface lg:flex lg:flex-col">
      <div className="flex items-center justify-between border-b border-border p-4">
        <h2 className="text-sm font-semibold">Execution Trace</h2>
        <button className="text-xs text-muted" onClick={() => setCollapsed(true)} type="button">
          Hide
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {entries.map((entry, index) => (
          <div key={`${entry.label}-${index}`} className={`rounded-lg border p-3 text-xs ${entry.className}`}>
            <div className="flex items-center justify-between">
              <span className="font-medium">{entry.label}</span>
              <span className="font-mono text-muted">{entry.durationMs}ms</span>
            </div>
            <p className="mt-2 text-muted">{entry.detail}</p>
          </div>
        ))}
      </div>
      <div className="border-t border-border p-4 font-mono text-xs text-muted">Total elapsed: {total}ms</div>
    </aside>
  );
}

function toTraceEntries(events: AgentEvent[]) {
  return events.flatMap((event) => {
    if (event.type === "step_start") {
      return [
        {
          label: `Step ${event.step}`,
          detail: event.description,
          durationMs: 150,
          className: "border-blue-500/40 text-blue-200"
        }
      ];
    }
    if (event.type === "tool_call") {
      return [
        {
          label: event.tool,
          detail: "tool call",
          durationMs: 150,
          className: "border-violet-500/50 text-violet-200"
        }
      ];
    }
    if (event.type === "error") {
      return [
        {
          label: "error",
          detail: event.message,
          durationMs: 0,
          className: "border-red-500/60 text-red-200"
        }
      ];
    }
    return [];
  });
}
