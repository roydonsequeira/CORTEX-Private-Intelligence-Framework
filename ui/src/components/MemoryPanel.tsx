"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { MemorySearchResult, ToolSchema } from "@/lib/types";

interface MemoryPanelProps {
  sessionId: string | null;
}

export function MemoryPanel({ sessionId }: MemoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MemorySearchResult[]>([]);
  const [sessions, setSessions] = useState<string[]>([]);
  const [tools, setTools] = useState<ToolSchema[]>([]);

  useEffect(() => {
    void api.listSessions().then(setSessions).catch(() => setSessions([]));
    void api.listTools().then(setTools).catch(() => setTools([]));
  }, []);

  useEffect(() => {
    if (!open || query.trim().length < 2) {
      setResults([]);
      return;
    }
    const timer = window.setTimeout(() => {
      void api.searchMemory(query).then(setResults).catch(() => setResults([]));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [open, query]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const memoryCount = useMemo(() => results.length || sessions.length, [results.length, sessions.length]);

  return (
    <aside className="hidden w-[240px] shrink-0 border-r border-border bg-surface md:flex md:flex-col">
      <div className="border-b border-border p-4">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-muted">CORTEX</p>
        <h2 className="mt-3 text-lg font-semibold">Local Agent</h2>
      </div>
      <div className="space-y-4 p-4 text-sm">
        <Metric label="Session" value={sessionId ?? "new"} />
        <Metric label="Memory" value={`${memoryCount} items`} />
        <Metric label="Tools" value={`${tools.length} online`} />
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-left text-muted transition duration-150 ease-cortex hover:border-accent hover:text-primary"
        >
          Search memory... <span className="float-right font-mono text-xs">⌘K</span>
        </button>
      </div>

      {open ? (
        <div className="fixed inset-0 z-50 bg-black/60 p-6" onClick={() => setOpen(false)}>
          <div
            className="mx-auto mt-24 max-w-2xl rounded-xl border border-border bg-surface p-4 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search working, episodic, and semantic memory..."
              className="w-full rounded-lg border border-border bg-background px-4 py-3 font-mono text-sm text-primary focus:border-accent focus:ring-accent"
            />
            <div className="mt-4 max-h-96 space-y-2 overflow-y-auto">
              {results.map((item) => (
                <div key={item.id} className="rounded-lg border border-border bg-background p-3 text-sm">
                  <p className="text-primary">{item.content}</p>
                  <p className="mt-1 font-mono text-xs text-muted">{item.memory_type}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-background p-3">
      <p className="font-mono text-xs uppercase text-muted">{label}</p>
      <p className="mt-1 truncate text-sm text-primary">{value}</p>
    </div>
  );
}
