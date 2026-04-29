import type { AgentState, MemorySearchResult, ToolSchema } from "./types";

const API_URL = process.env.NEXT_PUBLIC_CORTEX_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  chatMessage(message: string, sessionId: string | null): Promise<AgentState> {
    return request<AgentState>("/chat/message", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId })
    });
  },

  listTools(): Promise<ToolSchema[]> {
    return request<ToolSchema[]>("/tools");
  },

  async searchMemory(query: string): Promise<MemorySearchResult[]> {
    const data = await request<{ results: MemorySearchResult[] }>(
      `/memory/search?q=${encodeURIComponent(query)}&types=semantic,episodic&top_k=8`
    );
    return data.results;
  },

  async listSessions(): Promise<string[]> {
    const data = await request<{ sessions: string[] }>("/memory/sessions");
    return data.sessions;
  }
};

export { API_URL };
