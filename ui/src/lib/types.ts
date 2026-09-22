export type ModelCapability = "FAST" | "REASONING";

export type AgentEvent =
  // "user" is a client-only event: the message the operator sent. The server
  // never emits it; the streaming hook inserts it so the transcript can show
  // both sides of the conversation.
  | { type: "user"; value: string }
  | { type: "session_id"; value: string }
  | { type: "plan"; steps: string[] }
  | { type: "step_start"; step: number; description: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool?: string; success: boolean; output: string; error?: string | null }
  | { type: "token"; value: string }
  | { type: "done"; steps_taken: number }
  | { type: "error"; message: string };

/** An agent event tagged with the client-side time (ms) it arrived. */
export type TimedEvent = AgentEvent & { at: number };

export interface AgentState {
  session_id: string;
  user_input: string;
  plan: string[];
  steps_taken: number;
  final_answer: string | null;
  status: "planning" | "executing" | "reflecting" | "complete" | "failed";
}

export interface ToolSchema {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface MemorySearchResult {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  timestamp: string;
  memory_type: string;
}
