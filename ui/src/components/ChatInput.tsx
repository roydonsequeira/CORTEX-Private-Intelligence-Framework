"use client";

import { useState } from "react";

import type { ModelCapability } from "@/lib/types";

interface ChatInputProps {
  disabled: boolean;
  onSubmit: (message: string, capability: ModelCapability) => void;
}

// The API rejects longer messages (422); the box stops at the same limit.
const MAX_MESSAGE_CHARS = 8192;

const CAPABILITIES: { value: ModelCapability; label: string }[] = [
  { value: "FAST", label: "Fast" },
  { value: "REASONING", label: "Reasoning" }
];

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
    <div className="border-t border-line bg-ground px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-card border border-line bg-panel focus-within:border-ink-faint">
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Ask CORTEX to reason, use a local tool, or recall memory…"
            rows={2}
            maxLength={MAX_MESSAGE_CHARS}
            className="w-full resize-none border-0 bg-transparent px-4 pt-3.5 text-[15px] leading-relaxed text-ink placeholder:text-ink-faint focus:ring-0"
            disabled={disabled}
          />
          <div className="flex items-center justify-between px-3 pb-3">
            <div className="flex items-center gap-3">
              <div className="inline-flex rounded-lg border border-line p-0.5">
                {CAPABILITIES.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setCapability(option.value)}
                    className={`rounded-md px-2.5 py-1 font-mono text-[12px] transition-colors ${
                      capability === option.value
                        ? "bg-raised text-ink"
                        : "text-ink-faint hover:text-ink-dim"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              {message.length > MAX_MESSAGE_CHARS * 0.9 ? (
                <span className="font-mono text-[11px] text-ink-faint">
                  {message.length.toLocaleString()} / {MAX_MESSAGE_CHARS.toLocaleString()}
                </span>
              ) : null}
            </div>
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !message.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-signal px-4 py-2 text-[13px] font-semibold text-ground transition-opacity duration-150 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {disabled ? (
                <span className="live h-1.5 w-1.5 rounded-full bg-ground/70" />
              ) : null}
              {disabled ? "Running" : "Send"}
              <span className="font-mono text-[11px] opacity-60">↵</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
