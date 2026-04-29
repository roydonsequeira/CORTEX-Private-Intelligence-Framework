export type ModelCapability = "FAST" | "REASONING";

export type AgentEvent =
  | { type: "session_id"; value: string }
  | { type: "plan"; steps: string[] }
  | { type: "step_start"; step: number; description: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool?: string; success: boolean; output: string; error?: string | null }
  | { type: "token"; value: string }
  | { type: "done"; steps_taken: number }
  | { type: "error"; message: string };

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
