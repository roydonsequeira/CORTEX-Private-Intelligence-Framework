"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import type { TimedEvent } from "@/lib/types";

interface AgentOutputProps {
  events: TimedEvent[];
  isStreaming: boolean;
  onRun: (prompt: string) => void;
}

interface Turn {
  user: string;
  events: TimedEvent[];
}

const EXAMPLES = [
  "Use Python to compute the 20th Fibonacci number.",
  "Create a file called demo.txt with the text 'CORTEX works', then read it back.",
  "My name is Roy and my favorite language is Python.",
  "Run this Python: os = json.codecs.sys.modules.get('os'); print(os.getcwd())"
];

export function AgentOutput({ events, isStreaming, onRun }: AgentOutputProps) {
  const turns = groupTurns(events);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [events]);

  return (
    <main className="flex min-h-0 flex-1 flex-col bg-ground">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-6 py-8">
          {turns.length === 0 ? (
            <EmptyState onRun={onRun} />
          ) : (
            <div className="space-y-8">
              {turns.map((turn, index) => (
                <TurnBlock
                  key={index}
                  turn={turn}
                  streaming={isStreaming && index === turns.length - 1}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

// Small models often wrap a Markdown table or list in a ```markdown fence, which
// would render as raw pipes in a code block. Unwrap fences labelled markdown/md;
// real code fences (python, bash, …) are left untouched.
const MARKDOWN_FENCE = /```(?:markdown|md)[ \t]*\n([\s\S]*?)```/g;

function unwrapMarkdownFences(text: string): string {
  return text.replace(MARKDOWN_FENCE, (_match, inner: string) => inner);
}

function TurnBlock({ turn, streaming }: { turn: Turn; streaming: boolean }) {
  const plan = turn.events.find((e) => e.type === "plan") as
    | Extract<TimedEvent, { type: "plan" }>
    | undefined;
  const completedSteps = turn.events.filter((e) => e.type === "step_start").length;
  const activity = turn.events.filter(
    (e) => e.type === "tool_call" || e.type === "tool_result" || e.type === "error"
  );
  // A token_reset means text streamed so far was preamble before a tool call,
  // not the answer — keep only the tokens that follow the last reset.
  const lastReset = turn.events.map((e) => e.type).lastIndexOf("token_reset");
  const answer = turn.events
    .slice(lastReset + 1)
    .filter((e): e is Extract<TimedEvent, { type: "token" }> => e.type === "token")
    .map((e) => e.value)
    .join("");
  const done = turn.events.find((e) => e.type === "done") as
    | Extract<TimedEvent, { type: "done" }>
    | undefined;

  return (
    <div className="space-y-3">
      {turn.user ? (
        <div className="flex justify-end">
          <p className="max-w-[85%] rounded-card rounded-br-sm bg-raised px-4 py-2.5 text-[15px] leading-relaxed text-ink">
            {turn.user}
          </p>
        </div>
      ) : null}

      {plan && plan.steps.length > 0 ? (
        <div className="settle rounded-card border border-line bg-panel p-4">
          <p className="mb-3 font-mono text-[11px] tracking-wide text-ink-faint">plan</p>
          <ol className="space-y-2">
            {plan.steps.map((step, i) => {
              const stepDone = i < completedSteps;
              return (
                <li key={i} className="flex gap-3 text-sm">
                  <span
                    className={`mt-px grid h-5 w-5 shrink-0 place-items-center rounded-full border font-mono text-[11px] ${
                      stepDone
                        ? "border-signal/40 bg-signal/10 text-signal"
                        : "border-line text-ink-faint"
                    }`}
                  >
                    {stepDone ? "✓" : i + 1}
                  </span>
                  <span className={stepDone ? "text-ink-dim" : "text-ink"}>{step}</span>
                </li>
              );
            })}
          </ol>
        </div>
      ) : null}

      {activity.map((event, i) => (
        <ActivityCard key={i} event={event} />
      ))}

      {answer ? (
        <div className="settle answer rounded-card border border-line bg-panel px-5 py-4 text-[15px] leading-relaxed text-ink">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
            {unwrapMarkdownFences(answer)}
          </ReactMarkdown>
          {streaming ? <span className="caret" aria-hidden /> : null}
        </div>
      ) : streaming && !answer ? (
        <WorkingIndicator turn={turn} />
      ) : null}

      {done && !streaming ? (
        <p className="font-mono text-[11px] text-ink-faint">
          done · {done.steps_taken} step{done.steps_taken === 1 ? "" : "s"}
        </p>
      ) : null}
    </div>
  );
}

function ActivityCard({ event }: { event: TimedEvent }) {
  if (event.type === "tool_call") {
    return (
      <div className="settle rounded-card border border-tool/30 bg-tool/[0.06] p-3.5">
        <div className="mb-2 flex items-center gap-2">
          <ToolGlyph />
          <span className="font-mono text-[13px] text-tool">{event.tool}</span>
          <span className="text-[12px] text-ink-faint">called</span>
        </div>
        <pre className="max-h-40 overflow-auto rounded-md bg-ground p-2.5 font-mono text-[12px] leading-relaxed text-ink-dim">
          {formatArgs(event.args)}
        </pre>
      </div>
    );
  }
  if (event.type === "tool_result") {
    const ok = event.success;
    return (
      <div
        className={`settle rounded-card border p-3.5 ${
          ok ? "border-ok/30 bg-ok/[0.06]" : "border-bad/40 bg-bad/[0.07]"
        }`}
      >
        <div className="mb-2 flex items-center gap-2">
          <span className={`text-[13px] ${ok ? "text-ok" : "text-bad"}`}>{ok ? "●" : "▲"}</span>
          <span className={`font-mono text-[13px] ${ok ? "text-ok" : "text-bad"}`}>
            {event.tool ?? "tool"}
          </span>
          <span className="text-[12px] text-ink-faint">{ok ? "returned" : "failed"}</span>
        </div>
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-ground p-2.5 font-mono text-[12px] leading-relaxed text-ink-dim">
          {event.error ?? event.output}
        </pre>
      </div>
    );
  }
  return (
    <div className="settle rounded-card border border-bad/40 bg-bad/[0.07] p-3.5 text-sm text-bad">
      {event.type === "error" ? event.message : null}
    </div>
  );
}

function WorkingIndicator({ turn }: { turn: Turn }) {
  const label = phaseLabel(turn.events);
  return (
    <div className="flex items-center gap-2.5 px-1 py-1 text-sm text-ink-dim">
      <span className="live h-2 w-2 rounded-full bg-signal" />
      <span className="font-mono text-[13px]">{label}</span>
    </div>
  );
}

function EmptyState({ onRun }: { onRun: (prompt: string) => void }) {
  return (
    <div className="flex min-h-[60vh] flex-col justify-center">
      <CortexMark />
      <h1 className="mt-6 text-[26px] font-semibold tracking-tight text-ink">
        A private agent that thinks on your machine.
      </h1>
      <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-ink-dim">
        CORTEX plans, calls local tools, and remembers across the conversation. Nothing
        leaves this computer. Ask it something, or start with one of these:
      </p>
      <div className="mt-6 grid gap-2.5 sm:grid-cols-2">
        {EXAMPLES.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onRun(prompt)}
            className="group rounded-card border border-line bg-panel px-4 py-3 text-left text-[13.5px] leading-snug text-ink-dim transition-colors duration-150 hover:border-signal/50 hover:text-ink"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

function ToolGlyph() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M14.7 6.3a4 4 0 0 0-5 5L3 18l3 3 6.7-6.7a4 4 0 0 0 5-5l-2.6 2.6-2.1-.4-.4-2.1 2.6-2.6Z"
        stroke="var(--tool)"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CortexMark() {
  return (
    <svg width="34" height="34" viewBox="0 0 40 40" fill="none" aria-hidden>
      <circle cx="20" cy="9" r="3.4" fill="var(--signal)" />
      <circle cx="9" cy="28" r="3.4" fill="var(--signal)" />
      <circle cx="31" cy="28" r="3.4" fill="var(--signal)" />
      <circle cx="20" cy="20" r="2.2" fill="var(--ink-dim)" />
      <path
        d="M20 12.4 20 17.8M17.9 21.4 11 25.6M22.1 21.4 29 25.6"
        stroke="var(--line)"
        strokeWidth="1.4"
      />
    </svg>
  );
}

function groupTurns(events: TimedEvent[]): Turn[] {
  const turns: Turn[] = [];
  for (const event of events) {
    if (event.type === "user") {
      turns.push({ user: event.value, events: [] });
    } else if (turns.length > 0) {
      turns[turns.length - 1].events.push(event);
    } else {
      turns.push({ user: "", events: [event] });
    }
  }
  return turns;
}

function phaseLabel(events: TimedEvent[]): string {
  const last = [...events].reverse().find((e) => e.type !== "session_id");
  if (!last) return "thinking";
  if (last.type === "plan") return "planning";
  if (last.type === "step_start" || last.type === "tool_call") return "executing";
  if (last.type === "tool_result") return "reading result";
  if (last.type === "token") return "responding";
  return "thinking";
}

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 1 && typeof entries[0][1] === "string") {
    return entries[0][1] as string;
  }
  return JSON.stringify(args, null, 2);
}
