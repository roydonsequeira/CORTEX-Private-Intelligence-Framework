# ADR-006: A Model Reading Email Gets No Tools

Status: accepted, recorded 26 Nov 2026. Not built yet; see [the design note](../design/inbox.md).

## Context

In 2.0.0a1 the chat agent can read a file or a web page and then call tools in
the same run. Its guards ([THREAT_MODEL.md](../../THREAT_MODEL.md)) limit what a
call can do, but nothing stops the model from trying what the text says. Anyone
can write an email, so the inbox features need a rule that doesn't depend on the
model ignoring it. Keeping untrusted text away from a model that can act is one
of the patterns in "Design Patterns for Securing LLM Agents"
([arXiv 2506.08837](https://arxiv.org/abs/2506.08837)).

## Decision

- A model call that contains text from an email, a page or a file gets no tool
  schemas. This is checked where model calls are made, and a call that mixes
  that text with tools raises an error before anything is sent.
- The Reader reads one email per call and must return a fixed schema. Code
  checks each number, date and name against the email, and code decides what
  happens with the result.
- For Ask, code finds the emails, and the model answers from them with no tools,
  citing them.
- The tool-using chat agent never receives email text and has no inbox tools.

## Consequences

- An email can't make Quietpost call a tool. It can still lie: a false claim in
  an email can reach a summary. The UI shows email text as quoted sender content
  with its source, and the threat model lists deception as out of scope.
- Ask can't take actions. An action goes through the approval queue.
- One email per Reader call costs more model calls than batching, and a hostile
  email can affect only its own result.
- Every path that calls a model needs a test that tries to break the rule.
