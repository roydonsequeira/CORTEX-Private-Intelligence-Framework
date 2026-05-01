# ADR-003: Why SSE Over WebSockets

## Context

Agent output is primarily a one-way event stream: session IDs, plans, step starts, tool calls, tool results, tokens, completion, and errors. The frontend does not need full-duplex low-latency messaging for the current runtime.

## Decision

CORTEX uses Server-Sent Events for `/chat/stream` instead of WebSockets.

## Consequences

SSE is HTTP-native, simple to test with ASGI clients, easy to consume from browsers, and naturally reconnectable. It avoids bidirectional framing complexity while matching the agent-to-client event shape. If future features need full-duplex collaboration or voice transport, WebSockets can be added as a separate endpoint without replacing SSE.
