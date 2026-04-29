"use client";

import { useState } from "react";

import type { ModelCapability } from "@/lib/types";

interface ChatInputProps {
  disabled: boolean;
  onSubmit: (message: string, capability: ModelCapability) => void;
}

export function ChatInput({ disabled, onSubmit }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [capability, setCapability] = useState<ModelCapability>("FAST");

  function submit() {
    const trimmed = message.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSubmit(trimmed, capability);
    setMessage("");
  }

  return (
    <div className="border-t border-border bg-surface/80 p-4">
      <div className="rounded-xl border border-border bg-background/70 p-3 shadow-2xl">
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          placeholder="Ask CORTEX to reason, search memory, or use local tools..."
          className="min-h-24 w-full resize-none border-0 bg-transparent font-mono text-sm text-primary placeholder:text-muted focus:ring-0"
          disabled={disabled}
        />
        <div className="mt-3 flex items-center justify-between">
          <select
            value={capability}
            onChange={(event) => setCapability(event.target.value as ModelCapability)}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-xs text-primary focus:border-accent focus:ring-accent"
          >
            <option value="FAST">Fast</option>
            <option value="REASONING">Reasoning</option>
          </select>
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !message.trim()}
            className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition duration-150 ease-cortex hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {disabled ? <span className="h-2 w-2 rounded-full bg-white thinking-dot" /> : null}
            Send <span className="font-mono text-xs opacity-70">⌘↵</span>
          </button>
        </div>
      </div>
    </div>
  );
}
