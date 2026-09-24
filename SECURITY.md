# Security Policy

## Supported versions

Security fixes ship on the latest release. Please reproduce on the most recent
tag (or `main`) before reporting an issue.

| Version | Supported |
|---|---|
| latest release | ✅ |
| older tags | ❌ |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Report privately through GitHub's **"Report a vulnerability"** button under the
repository's **Security** tab (Private Vulnerability Reporting). If that is
unavailable, contact the maintainer through their GitHub profile and request a
private channel before sharing details.

Please include:

- a description of the issue and its impact,
- the version or commit you tested,
- reproduction steps or a proof of concept,
- any suggested remediation.

We aim to acknowledge a report within 72 hours and to agree on a disclosure
timeline once the issue is confirmed. Please give us a reasonable window to ship
a fix before any public disclosure.

## Threat model and scope

CORTEX is designed to run on hardware you control. Its guardrails chiefly
contain code the LLM generates from a malicious or injected prompt. By default
the API is reachable only from the local machine: it binds to `127.0.0.1`,
accepts browser requests only from the local UI's origin, rejects unknown
`Host` names (DNS rebinding), and keeps the direct tool-execution endpoint
disabled. Exposing it beyond loopback without `api_key` is unsupported. See the
**Security Model** section of the [README](README.md) for what each guardrail
does and does not guarantee.

In scope:

- sandbox escapes from `python_exec` (see the RestrictedPython notes in the README),
- filesystem-guard bypasses (workspace-root escape, extension bypass),
- authentication/authorization bypasses when `api_key` is configured,
- request handling that lets one client affect another,
- ways for a web page or another origin to reach a default (loopback) instance.

Out of scope:

- attacks that assume untrusted code is safe to run in the best-effort
  in-process RestrictedPython sandbox — run the executor inside the provided
  Docker container for OS-level isolation of untrusted workloads,
- denial of service from unbounded local resource use on your own machine,
- issues in third-party models or Ollama itself (report those upstream).
