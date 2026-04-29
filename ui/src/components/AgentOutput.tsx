"use client";

import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import type { AgentEvent } from "@/lib/types";

interface AgentOutputProps {
  events: AgentEvent[];
  isStreaming: boolean;
}

export function AgentOutput({ events, isStreaming }: AgentOutputProps) {
  const plan = events.find((event): event is Extract<AgentEvent, { type: "plan" }> => event.type === "plan");
  const completedSteps = events.filter((event) => event.type === "step_start").length;
  const finalAnswer = events
    .filter((event): event is Extract<AgentEvent, { type: "token" }> => event.type === "token")
    .map((event) => event.value)
    .join("");

  return (
    <main className="flex min-h-0 flex-1 flex-col bg-background">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-4xl space-y-4">
          <header className="rounded-xl border border-border bg-surface p-5">
            <p className="font-mono text-xs uppercase tracking-[0.35em] text-muted">CORTEX Runtime</p>
            <h1 className="mt-3 text-2xl font-semibold text-primary">Private local agent</h1>
            <p className="mt-2 text-sm text-muted">Plans, tool calls, and memory traces stay on this machine.</p>
          </header>

          {plan ? (
            <section className="rounded-xl border border-border bg-surface p-4">
              <h2 className="text-sm font-semibold text-primary">Plan</h2>
              <ol className="mt-3 space-y-2">
                {plan.steps.map((step, index) => (
                  <li key={step} className="flex gap-3 text-sm">
                    <span className={index < completedSteps ? "text-accent" : "text-muted"}>
                      {index < completedSteps ? "✓" : index + 1}
                    </span>
                    <span className="text-primary">{step}</span>
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          {events.map((event, index) => {
            if (event.type === "tool_call") {
              return (
                <details key={index} className="rounded-xl border border-border bg-surface p-4 tool-flash">
                  <summary className="cursor-pointer text-sm font-medium text-accent">Tool call: {event.tool}</summary>
                  <pre className="mt-3 overflow-x-auto rounded-lg bg-background p-3 font-mono text-xs text-primary">
                    {JSON.stringify(event.args, null, 2)}
                  </pre>
                </details>
              );
            }
            if (event.type === "tool_result") {
              return (
                <details key={index} className="rounded-xl border border-border bg-surface p-4 tool-flash">
                  <summary className="cursor-pointer text-sm font-medium text-primary">
                    Tool result {event.success ? "succeeded" : "failed"}
                  </summary>
                  <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-background p-3 font-mono text-xs text-primary">
                    {event.error ?? event.output}
                  </pre>
                </details>
              );
            }
            if (event.type === "error") {
              return (
                <div key={index} className="rounded-xl border border-border border-l-red-500 bg-surface p-4 text-sm text-red-300">
                  {event.message}
                </div>
              );
            }
            return null;
          })}

          {finalAnswer ? (
            <article className="prose prose-invert max-w-none rounded-xl border border-border bg-surface p-5 prose-pre:bg-background">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                {finalAnswer}
              </ReactMarkdown>
              {isStreaming ? <span className="ml-1 inline-block h-4 w-2 bg-accent align-middle thinking-dot" /> : null}
            </article>
          ) : null}
        </div>
      </div>
    </main>
  );
}
