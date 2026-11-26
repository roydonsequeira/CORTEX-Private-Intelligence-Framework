# Design note: the inbox features

Status: design, 26 Nov 2026. None of this is built yet;
[THREAT_MODEL.md](../../THREAT_MODEL.md) covers what is. This note says what
the inbox features will store, how their modules fit together, and the rules
each part must keep, so that each part can be built and tested on its own.

## What Quietpost stores today

`quietpost init` puts the config file and the data folder in the user's own
folders: `AppData\Local\quietpost` on Windows, `~/Library/Application
Support/quietpost` on macOS, and `~/.config/quietpost` for the config and
`~/.local/share/quietpost` for the data on Linux. The workspace is
`~/Quietpost`.

| File | Contents |
|---|---|
| `quietpost.yaml`, in the config folder | Settings. `api_key` and `model_server_api_key`, when set here, are plain text. |
| `quietpost.db`, in the data folder | SQLite, table `episodes`: each chat turn, with its session, role, text and time. |
| `chroma/`, in the data folder | Two Chroma collections: `cortex_semantic`, with facts about you and the chunks of documents that `doc_search` indexed, and `cortex_procedural`, with the tool sequences of successful runs. |
| `tasks.db`, in the data folder | Background task results, only with `task_store: sqlite`. By default they stay in memory. |
| The workspace | The files the agent reads and writes. |

Logs go to standard output, not to a file. Traces and metrics are exported only
when `telemetry_enabled` is on, and aren't stored.

## What the inbox features will store

Everything below goes into one new SQLite database, `inbox.db`, in the data
folder, except the email chunks for Ask, which go into Chroma.

| Data | Why it's kept |
|---|---|
| One row per message: where it came from (the `.mbox` file and byte offset, or the IMAP folder, its UIDVALIDITY and the UID), the Message-ID, thread, date, sender, recipients and subject, the list headers (`List-Unsubscribe`, `List-Unsubscribe-Post`, `List-Id`, `Precedence`), the receiving server's `Authentication-Results`, the size, and the names and sizes of attachments | Every analysis, and the link from each figure back to its email |
| The message's text: the plain-text part, or the text of the HTML part with the markup removed | Search, the Reader, Ask, and showing the email |
| A full-text index (SQLite FTS5) over sender, subject and text | Search and Ask |
| Sync state: for each IMAP folder its UIDVALIDITY and the highest UID read, and for each `.mbox` file its path, size and the offset reached | Resuming an import or a sync without reading everything again |
| Signals per message (bulk sender, replied to, lookalike domain, account email, text aimed at an AI) and the audit's results | The audit |
| The Reader's results (subscriptions, bills, deadlines), each with its message and the text that supports it | The source beside every figure |
| A copy of the public breach catalogue, with the date it was downloaded | Matching breaches on this machine |
| The network ledger: one row per outbound connection (time, host, port, purpose). Each row holds the hash of the row before it, so an edit to an earlier row shows. | The Trust page |
| Proposed actions that wait for your click, and what you decided | The approval queue |
| The results of the attack drill | The Trust page |
| Email text in chunks, with their embeddings, in a Chroma collection of its own, apart from the facts about you | Ask's retrieval |

The IMAP app password is kept in the operating system's keyring (Windows
Credential Manager, the macOS keychain, or the Secret Service on Linux). It
never goes into a config file, a log or a model's context.

Not kept: attachment contents, images and other remote content, the HTML of a
message, and mail you haven't imported.

[ADR-007](../architecture-decisions/ADR-007-encryption-at-rest.md) covers
encryption at rest. One command will delete the mail store, its search index and
its Chroma collection. Deleting the data folder removes everything Quietpost
stored.

## How the parts fit

1. Import. `inbox/mbox.py` reads a Google Takeout `.mbox` file where it is,
   without copying or changing it. `inbox/imap.py` syncs read-only (I2). Both
   parse with `inbox/mime.py`, which applies the limits below, and write
   through `inbox/store.py`.
2. Signals and the audit. `inbox/signals.py` and the `audit/` modules are
   plain code: people waiting on you, newsletters you never open, who has your
   data, and breach matches. They need no model.
3. The Reader. `reader/` reads one email per model call, with no tools (I1),
   and the model's output must match a fixed schema. `reader/grounding.py`
   drops every number, date and name that doesn't appear in the email (I5).
4. Ask. `ask/` finds the emails with code (full-text search and
   embeddings). The model then answers from them with no tools and cites them.
   The tool-using chat agent never receives email text and has no inbox tools.
5. Trust. `trust/` holds the network guard and its ledger (I4), the
   approval queue (I8), the tripwires, and the attack drill.

The proposed layout:

```
src/quietpost/
  inbox/       mbox.py  imap.py  mime.py  store.py  signals.py
  audit/       waiting.py  newsletters.py  footprint.py  breaches.py
  reader/      reader.py  grounding.py
  ask/         retrieve.py  answer.py
  trust/       guard.py  ledger.py  approvals.py  tripwires.py
  gauntlet/    attack emails, and the code the drill runs them with
  api/routes/  inbox.py  audit.py  ask.py  trust.py
ui/src/app/    today/  audit/  ask/  trust/  settings/
```

Every model call ends in one of two classes: `OllamaProvider` in
`models/provider.py` or `OpenAICompatibleProvider` in `models/openai_compat.py`.
I1 is checked in a helper that both call before sending, so a new code path
can't skip it.

## Limits on hostile mail

Each limit is a setting. The defaults below are proposals, tested with
oversized fixtures when the parser is built.

- A message over 50 MB is skipped and counted, not parsed.
- The parser stops after 20 levels of MIME nesting or 500 parts.
- At most 1 MB of text is kept per message. The Reader gets what fits in the
  model's context, and its result records when the text was cut.

## The rules

| Rule | Where it's enforced | How it's tested |
|---|---|---|
| I1. A model call that contains text from outside (an email, a page, a file) gets no tool schemas. | Each `Message` records where its text came from, and the providers' shared helper raises an error for a call with outside text and tools, before anything is sent. | A test sends outside text with tools on every path that calls a model and expects the error each time. |
| I2. The mailbox is opened read-only, with `EXAMINE` and `BODY.PEEK`, and no command that changes it is sent. There is no code that sends mail. | `inbox/imap.py` ([ADR-005](../architecture-decisions/ADR-005-read-only-mailbox.md)) | A test mail server records every command, and the test fails on any command outside a fixed list. |
| I3. Models run on this machine or the local network. | Built: see THREAT_MODEL.md. | |
| I4. Outbound connections go only to your mail server, the local model server, the breach catalogue's host, unsubscribe links you approved, and addresses you wrote. Each one is recorded in the ledger. | `trust/guard.py`, in the HTTP clients and the IMAP connection | A test that fails on any other host, and an `.mbox` audit with networking blocked. |
| I5. Every number, date and name the Reader returns appears in its email, or it is dropped. | `reader/grounding.py` | A mocked model that returns invented values, and property tests on how amounts and dates are normalised. |
| I6. Text from an email is shown as quoted sender content, never in Quietpost's own voice. It loads nothing remote, and links show their real domain. | The UI's components and its Content-Security-Policy | Browser checks, and the existing test of the policy. |
| I7. Memory learns only from your own words, so email text never becomes a fact about you. | `memory/semantic.py`, built for chat | The existing tests, plus fixtures made of email text. |
| I8. Anything that goes out or can't be undone needs your click, and the approval shows exactly what will happen: the URL to be opened, or the request to be saved. | `trust/approvals.py` | Tests that no action runs without an approval record. |
| I9. Tools reach only the workspace, and generated code runs in the sandbox. | Built: see THREAT_MODEL.md. | |
| I10. No telemetry, with Chroma's own telemetry setting switched off in code. | `memory/chroma.py` | A test that each Chroma client is created with telemetry off. |
| I11. Microphone audio is recorded only while you hold the button, transcribed on this machine, and never stored. Nothing is sent until you send it. | The push-to-talk route and its UI | A fake transcriber in unit tests, and a socket test that shows no connection during a transcription. |

[ADR-006](../architecture-decisions/ADR-006-no-tools-while-reading-mail.md)
explains I1 and the split between the Reader, Ask and the chat agent.

## Out of scope

- Sending, deleting, moving or flagging mail.
- Opening attachments.
- Outlook.com and other accounts that need OAuth. Gmail works through a Takeout
  file, or through IMAP with an app password.
- A model server on the internet (I3).
