"use client";

import { useCallback, useRef, useState } from "react";

import { API_URL } from "./api";
import type { AgentEvent, ModelCapability, TimedEvent } from "./types";

interface StreamState {
  events: TimedEvent[];
  isStreaming: boolean;
  error: string | null;
  sessionId: string | null;
  sendMessage: (message: string, capability: ModelCapability) => Promise<void>;
  reset: () => void;
}

export function useAgentStream(initialSessionId: string | null = null): StreamState {
  const [events, setEvents] = useState<TimedEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId);
  const abortRef = useRef<AbortController | null>(null);

  const append = useCallback((event: AgentEvent) => {
    setEvents((prev) => [...prev, { ...event, at: performance.now() }]);
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setEvents([]);
    setError(null);
    setIsStreaming(false);
  }, []);

  const sendMessage = useCallback(
    async (message: string, capability: ModelCapability) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);
      setError(null);
      append({ type: "user", value: message });

      try {
        const response = await fetch(`${API_URL}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            session_id: sessionId,
            model_capability: capability
          }),
          signal: controller.signal
        });
        if (!response.ok || !response.body) {
          throw new Error(httpErrorMessage(response.status));
        }
        await parseSse(response.body, (event) => {
          append(event);
          if (event.type === "session_id") {
            setSessionId(event.value);
          }
          if (event.type === "error") {
            setError(event.message);
          }
        });
      } catch (exc) {
        if ((exc as Error).name !== "AbortError") {
          const message = (exc as Error).message;
          setError(message);
          append({ type: "error", message });
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [append, sessionId]
  );

  return { events, isStreaming, error, sessionId, sendMessage, reset };
}

function httpErrorMessage(status: number): string {
  switch (status) {
    case 422:
      return "CORTEX could not accept that message: it must be 1 to 8,192 characters.";
    case 429:
      return "Too many requests. Wait a few seconds and try again.";
    case 401:
    case 403:
      return "The CORTEX API refused the request (API key or origin not allowed).";
    case 503:
      return "CORTEX is busy or restarting. Try again in a moment.";
    default:
      return `The CORTEX API returned an error (HTTP ${status}).`;
  }
}

async function parseSse(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: AgentEvent) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (!line) {
        continue;
      }
      onEvent(JSON.parse(line.slice(6)) as AgentEvent);
    }
  }
}
