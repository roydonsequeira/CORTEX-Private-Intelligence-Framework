# ADR-004: Why LATS Over Pure ReAct

## Context

Flat ReAct loops are effective for simple tool use but can stall on tasks that require exploring alternative plans. CORTEX needs a serious local-agent orchestration story for complex tasks without depending on external agent frameworks.

## Decision

CORTEX implements Language Agent Tree Search (LATS), based on Yao et al. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" (arXiv:2310.04406). LATS is used when explicitly enabled or when the reflector detects repeated lack of progress.

## Consequences

LATS gives CORTEX a principled fallback when linear reasoning fails: select, expand, simulate, evaluate, and backpropagate. It costs more model calls than pure ReAct, so it is budgeted and opt-in by default. The implementation remains transparent and testable inside the local agent kernel.
