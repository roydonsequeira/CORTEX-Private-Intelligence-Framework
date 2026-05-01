# ADR-001: Why Not LangChain

## Context

CORTEX is a reference implementation for local, privacy-first agents. The core loop needs to be observable, debuggable, testable, and understandable by engineers reading the source. Agent control flow, tool execution, memory retrieval, and model routing are first-class architecture concepts in this project.

## Decision

CORTEX does not use LangChain as a runtime dependency. The agent kernel, planner, executor, reflector, memory stack, and tool registry are implemented directly.

## Consequences

This keeps behavior explicit and avoids framework indirection around prompts, tool calls, retries, and tracing. It also makes it easier to audit privacy and sandbox boundaries. The tradeoff is that CORTEX owns more infrastructure code, so each subsystem needs focused tests and careful API boundaries.
