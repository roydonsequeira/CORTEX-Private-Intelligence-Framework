"use client";

import { AgentOutput } from "@/components/AgentOutput";
import { ChatInput } from "@/components/ChatInput";
import { ExecutionTrace } from "@/components/ExecutionTrace";
import { MemoryPanel } from "@/components/MemoryPanel";
import { useAgentStream } from "@/lib/streaming";
import type { ModelCapability } from "@/lib/types";

export default function Home() {
  const { events, isStreaming, error, sessionId, sendMessage } = useAgentStream(null);

  function submit(message: string, capability: ModelCapability) {
    void sendMessage(message, capability);
  }

  return (
    <div className="flex h-screen bg-background text-primary">
      <MemoryPanel sessionId={sessionId} />
      <div className="flex min-w-0 flex-1 flex-col">
        <AgentOutput events={events} isStreaming={isStreaming} />
        {error ? <div className="border-l-4 border-red-500 bg-red-950/40 px-4 py-2 text-sm text-red-200">{error}</div> : null}
        <ChatInput disabled={isStreaming} onSubmit={submit} />
      </div>
      <ExecutionTrace events={events} />
    </div>
  );
}
