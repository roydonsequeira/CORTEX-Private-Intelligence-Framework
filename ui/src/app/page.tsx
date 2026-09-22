"use client";

import { AgentOutput } from "@/components/AgentOutput";
import { ChatInput } from "@/components/ChatInput";
import { ExecutionTrace } from "@/components/ExecutionTrace";
import { MemoryPanel } from "@/components/MemoryPanel";
import { useAgentStream } from "@/lib/streaming";
import type { ModelCapability } from "@/lib/types";

export default function Home() {
  const { events, isStreaming, sessionId, sendMessage } = useAgentStream(null);

  function submit(message: string, capability: ModelCapability) {
    void sendMessage(message, capability);
  }

  function runExample(prompt: string) {
    if (!isStreaming) {
      void sendMessage(prompt, "FAST");
    }
  }

  return (
    <div className="flex h-screen bg-ground text-ink">
      <MemoryPanel sessionId={sessionId} />
      <div className="flex min-w-0 flex-1 flex-col">
        <AgentOutput events={events} isStreaming={isStreaming} onRun={runExample} />
        <ChatInput disabled={isStreaming} onSubmit={submit} />
      </div>
      <ExecutionTrace events={events} isStreaming={isStreaming} />
    </div>
  );
}
