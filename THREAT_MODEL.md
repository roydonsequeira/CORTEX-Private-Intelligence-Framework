# Threat model

This file says what Quietpost protects, against what, and which code and tests
enforce each guard. It covers 2.0.0a1, the release on PyPI from 15 Nov 2026: a
local agent with a chat UI, tools and memory. The inbox features aren't built
yet. Their design, and the rules they must keep, are in
[docs/design/inbox.md](docs/design/inbox.md). A rule moves into this file when
the code and the test that enforce it ship.

To report a problem, see [SECURITY.md](SECURITY.md).

## What is protected

- The files in your workspace folder, the only folder the agent's tools can read
  and write.
- Your conversations and memory, in the data folder.
- The rest of your computer: files outside the workspace, Quietpost's own code
  and plugins, and the other services on your network.
- Your config file, which can hold an API key.

## In scope and out of scope

| Quietpost defends against | Quietpost doesn't claim to stop |
|---|---|
| Text the agent reads (a file, a document, a web page) making it write files, fetch addresses or send data off the machine | Text that misleads you. A page can put false statements into an answer. |
| Code the model writes reaching outside the workspace | Malware, or another administrator, on the same computer |
| A web page in your browser reaching the local API | A malicious model file. You choose and pull models yourself. |
| Your prompts going to a model server on the internet | Loss of the device. [ADR-007](docs/architecture-decisions/ADR-007-encryption-at-rest.md) covers stored data. |
| A release changed between its build and PyPI | Denial of service through your own use of local resources |

## Guards in 2.0.0a1

Paths are under `src/quietpost/` unless they start with `ui/`, `tests/` or `.github/`.

| Guard | Code | Tests |
|---|---|---|
| The API listens on `127.0.0.1` and accepts only `localhost` and `127.0.0.1` as the Host name, which stops DNS rebinding from a web page. Browsers may call it only from local origins. Rate limits are on (200 requests a minute per client address, 60 for chat), an API key is optional, and the endpoint that runs a tool directly is off. | `config/settings.py`, `api/server.py`, `api/middleware/` | `tests/integration/test_api.py` |
| Tools reach only the workspace. A path must resolve inside the workspace folder, through any links; hidden files and folders are refused; a link swapped in after the check can't redirect the read or the write. Quietpost's own package and the plugins folder can't be written, and writes are limited to `allowed_write_extensions` (`.txt`, `.md`, `.json`, `.csv`, `.py` by default). | `tools/workspace.py`, `tools/builtin/filesystem.py` | `tests/unit/test_tools.py`, `tests/security/test_sr06_fs_toctou.py` |
| A file is written only when your own words ask to save one, and an existing file is replaced only when you ask for that. Text the agent reads never counts as you asking, and neither does text you quote, or paste after a request such as "summarize this". A paste with no quote marks and no such request can't be told apart from your own words. | `agent/executor.py`, `provenance.py` | `tests/unit/test_kernel.py`, `tests/security/test_sr03_instruction_text.py` |
| `web_fetch` opens only web addresses you wrote in the conversation; addresses from a file, a page or the model's guess are refused. The path and query must match what you wrote, and the host must match as written. It refuses loopback, private, link-local and reserved addresses, checks again at each redirect, and reads at most 2 MB. | `agent/executor.py`, `tools/builtin/web_fetch.py`, `network.py` | `tests/unit/test_kernel.py`, `tests/unit/test_tools.py` |
| Code the model writes runs under RestrictedPython in a separate process, with a time limit (10 seconds by default). Its `open()` reads the workspace and can't write. With `code_sandbox: container`, each run uses a Docker container with no network instead. The default sandbox is best effort; [SECURITY.md](SECURITY.md) says what that means. | `tools/sandbox.py`, `tools/builtin/code_exec.py` | `tests/unit/test_sandbox.py` |
| When the plan needs no tool (a direct answer, a refusal, writing code without running it), the model gets no tool schemas, so it has nothing to call. | `agent/kernel.py`, `agent/executor.py` | `tests/unit/test_kernel.py` |
| The model server must be on this machine or the local network. A public address is refused, and so are Ollama's cloud models: by name before any request, and by the `remote_host` Ollama reports for a model. | `models/endpoints.py`, `models/provider.py` | `tests/unit/test_model_endpoints.py`, `tests/unit/test_provider.py` |
| Answers can't load anything from another host. The UI shows images in an answer as links, and its Content-Security-Policy lets the page connect only to Quietpost's own server. | `ui/src/components/AgentOutput.tsx`, `api/web.py` | `tests/unit/test_web.py` (the policy). The image rule has no automated test. |
| Memory learns only from your own words, judged the same way as for file writes, so quoted material never becomes a fact about you. A run in which text from a file or page may have chosen a tool isn't learned as a habit. | `provenance.py`, `memory/semantic.py`, `agent/kernel.py` | `tests/security/test_sr03_instruction_text.py`, `test_sr04_semantic_memory.py`, `test_sr05_procedural_memory.py` |
| No telemetry. Trace and metric export is off by default (`telemetry_enabled: false`). | `config/settings.py`, `observability/` | `tests/unit/test_config.py` |
| Releases are built by a GitHub workflow from the tag and published through PyPI's trusted publishing, with an attestation for each file. | `.github/workflows/release.yml`, `scripts/check_dist.py` | `tests/unit/test_release_workflow.py` |

## Known gaps

- The chat agent can read a file or a page and then call tools in the same run.
  The guards above limit what those calls can do, but nothing stops the model
  from trying what the text says. The inbox features use a different design for
  this reason ([ADR-006](docs/architecture-decisions/ADR-006-no-tools-while-reading-mail.md)).
- `web_fetch` resolves a host name to check it, and the HTTP client resolves it
  again to connect. A DNS server that answers differently the second time could
  point the fetch at a private address ([#41](https://github.com/roydonsequeira/CORTEX-Private-Intelligence-Framework/issues/41)).
- The default sandbox runs RestrictedPython, which is best effort. Use
  `code_sandbox: container` for code you don't trust.
- Quietpost doesn't switch off Chroma's own telemetry setting. In Chroma 1.5.9,
  the version this was checked against, its telemetry client does nothing, so
  nothing is sent, but that depends on Chroma's version.
- `api_key` and `model_server_api_key` are kept as plain text in the config file
  or the environment.
- The image rule in the UI has no automated test.
