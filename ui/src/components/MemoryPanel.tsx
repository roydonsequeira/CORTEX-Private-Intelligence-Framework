"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

import { api } from "@/lib/api";
import type { MemorySearchResult, ToolSchema } from "@/lib/types";

interface MemoryPanelProps {
  sessionId: string | null;
}

interface SearchOutcome {
  term: string;
  results: MemorySearchResult[];
  failed: boolean;
}

export function MemoryPanel({ sessionId }: MemoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [outcome, setOutcome] = useState<SearchOutcome | null>(null);
  const [sessions, setSessions] = useState<string[]>([]);
  const [tools, setTools] = useState<ToolSchema[]>([]);

  useEffect(() => {
    void api.listTools().then(setTools).catch(() => setTools([]));
  }, []);

  // Refreshed when a conversation starts, so a new session is counted.
  useEffect(() => {
    void api.listSessions().then(setSessions).catch(() => setSessions([]));
  }, [sessionId]);

  const term = query.trim();
  const searching = open && term.length >= 2;

  useEffect(() => {
    if (!searching) {
      return;
    }
    // Only the latest search may update the list: a slow earlier request must
    // not overwrite the results of the query the user typed since.
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .searchMemory(term)
        .then((results) => {
          if (!cancelled) setOutcome({ term, results, failed: false });
        })
        .catch(() => {
          if (!cancelled) setOutcome({ term, results: [], failed: true });
        });
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [searching, term]);

  // Results belong to the term they were fetched for; while a newer search is
  // pending nothing is shown, so "No matches." never flashes before results.
  const current = searching && outcome?.term === term ? outcome : null;
  const results = current?.results ?? [];

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // The shortcut works with Ctrl or Cmd; show the key this keyboard has.
  const isMac = useSyncExternalStore(
    subscribeToNothing,
    () => /Mac|iPhone|iPad/.test(navigator.platform),
    () => true
  );

  return (
    <aside className="hidden w-[248px] shrink-0 flex-col border-r border-line bg-panel md:flex">
      <div className="flex items-center gap-2.5 border-b border-line px-5 py-4">
        <span className="h-2 w-2 rounded-full bg-signal" />
        <div>
          <p className="font-mono text-[13px] font-semibold tracking-wide text-ink">CORTEX</p>
          <p className="font-mono text-[11px] text-ink-faint">local agent · private</p>
        </div>
      </div>

      <div className="space-y-2.5 p-4">
        <Metric label="session" value={sessionId ? `${sessionId.slice(0, 8)}…` : "new"} />
        <Metric label="sessions in memory" value={String(sessions.length)} />
        <Metric label="tools online" value={String(tools.length)} />
      </div>

      <div className="px-4">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex w-full items-center justify-between rounded-card border border-line bg-ground px-3 py-2.5 text-[13px] text-ink-faint transition-colors hover:border-ink-faint hover:text-ink-dim"
        >
          Search memory
          <span className="font-mono text-[11px]">{isMac ? "⌘K" : "Ctrl K"}</span>
        </button>
      </div>

      {tools.length > 0 ? (
        <div className="mt-6 px-5">
          <p className="mb-2 font-mono text-[11px] text-ink-faint">available tools</p>
          <ul className="space-y-1.5">
            {tools.map((tool) => (
              <li key={tool.name} className="font-mono text-[12px] text-ink-dim">
                {tool.name}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-auto px-5 py-4">
        <p className="text-[11px] leading-relaxed text-ink-faint">
          Runs on your hardware via Ollama. Prompts, files, and memory never leave this
          machine.
        </p>
      </div>

      {open ? (
        <div
          className="fixed inset-0 z-50 bg-ground/70 backdrop-blur-xs"
          onClick={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Search memory"
            className="mx-auto mt-[12vh] max-w-xl rounded-card border border-line bg-panel p-4 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search working, episodic, and semantic memory…"
              aria-label="Search memory"
              className="w-full rounded-lg border border-line bg-ground px-4 py-3 text-[14px] text-ink placeholder:text-ink-faint focus:border-signal/50 focus:ring-0"
            />
            <div className="mt-3 max-h-[50vh] space-y-2 overflow-y-auto">
              {current?.failed ? (
                <p className="px-1 py-2 font-mono text-[12px] text-bad">
                  Memory search failed. Is the CORTEX API running?
                </p>
              ) : current && results.length === 0 ? (
                <p className="px-1 py-2 font-mono text-[12px] text-ink-faint">No matches.</p>
              ) : null}
              {results.map((item) => (
                <div key={item.id} className="rounded-lg border border-line bg-ground p-3">
                  <p className="text-[13.5px] leading-relaxed text-ink">{item.content}</p>
                  <p className="mt-1.5 font-mono text-[11px] text-ink-faint">
                    {item.memory_type}
                  </p>
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
    <div className="flex items-center justify-between rounded-card border border-line bg-ground px-3 py-2.5">
      <span className="font-mono text-[11px] text-ink-faint">{label}</span>
      <span className="max-w-[55%] truncate font-mono text-[12.5px] text-ink">{value}</span>
    </div>
  );
}

function subscribeToNothing(): () => void {
  return () => {};
}
