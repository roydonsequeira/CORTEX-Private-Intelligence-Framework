# CORTEX Demo

These examples assume CORTEX is running, either natively (`cortex serve`) or with
Docker (`make up && make pull-models`).

API base URL: `http://localhost:8000`

## 1. Simple Q&A

```bash
curl -s -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"In one sentence, what is CORTEX?","session_id":null}'
```

Expected shape:

```json
{
  "session_id": "...",
  "status": "complete",
  "final_answer": "CORTEX is a local, privacy-first AI agent framework..."
}
```

## 2. File Write + Read Back (filesystem tool)

```bash
curl -s -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"Write a file called demo.txt containing hello from CORTEX, then read it back.","session_id":null}'
```

The response's `tool_results` show two `filesystem` calls (a write, then a read),
and `final_answer` confirms the content. Writes never overwrite an existing file
unless you explicitly ask for it.

## 3. Code Execution (sandboxed python_exec)

```bash
curl -s -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"Use Python to compute the 20th Fibonacci number.","session_id":null}'
```

Expected: a `python_exec` tool result of `6765` and a matching `final_answer`.

To call a tool directly, bypassing the agent (for debugging), set
`debug_tool_endpoint: true` in `cortex.yaml`, restart, then:

```bash
curl -s -X POST http://localhost:8000/tools/python_exec/execute \
  -H "Content-Type: application/json" \
  -d '{"code":"print(21 * 2)"}'
```

## 4. Multi-Step Research Task

```bash
curl -s -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task":"Research the CORTEX architecture and summarize the memory and tool systems.","priority":"normal","orchestration":"supervisor"}'
```

Expected shape:

```json
{"task_id": "...", "status": "queued"}
```

Check the task:

```bash
curl -s http://localhost:8000/tasks/<task_id>
```

Expected status transitions from `queued` to `running` to `done`.

## 5. Memory Recall Across Sessions

Store a fact:

```bash
curl -s -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"Remember that my favourite programming language is Rust.","session_id":null}'
```

Search memory:

```bash
curl -s "http://localhost:8000/memory/search?q=favourite%20programming%20language&types=episodic,semantic&top_k=5"
```

Expected output includes the remembered preference in `results`.

## 6. SSE Streaming

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain CORTEX in three bullets.","session_id":null}'
```

Expected event order:

```text
data: {"type":"session_id",...}
data: {"type":"plan",...}
data: {"type":"step_start",...}
data: {"type":"token",...}
data: {"type":"done",...}
```
