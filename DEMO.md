# CORTEX Demo

These examples assume the stack is running:

```bash
make up
make pull-models
```

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

## 2. File Write + Read Back

```bash
curl -s -X POST http://localhost:8000/tools/filesystem/execute \
  -H "Content-Type: application/json" \
  -d '{"action":"write_file","path":"demo.txt","content":"hello from CORTEX"}'

curl -s -X POST http://localhost:8000/tools/filesystem/execute \
  -H "Content-Type: application/json" \
  -d '{"action":"read_file","path":"demo.txt"}'
```

Expected output includes:

```json
{"success": true, "output": "hello from CORTEX"}
```

## 3. Code Execution

```bash
curl -s -X POST http://localhost:8000/tools/python_exec/execute \
  -H "Content-Type: application/json" \
  -d '{"code":"print(21 * 2)"}'
```

Expected output:

```json
{"tool_name": "python_exec", "success": true, "output": "42"}
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
  -d '{"message":"Remember that my preferred local model is llama3.1:8b.","session_id":null}'
```

Search memory:

```bash
curl -s "http://localhost:8000/memory/search?q=preferred%20local%20model&types=episodic,semantic&top_k=5"
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
