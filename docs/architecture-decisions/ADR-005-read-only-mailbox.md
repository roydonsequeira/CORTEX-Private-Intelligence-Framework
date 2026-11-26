# ADR-005: The Mailbox Is Read-Only, and Nothing Sends Mail

Status: accepted, recorded 26 Nov 2026. Not built yet; see [the design note](../design/inbox.md).

## Context

Public research shows that email agents can be made to act by text inside the
mail they read ([arXiv 2507.02699](https://arxiv.org/abs/2507.02699)). An agent
that can change a mailbox or send mail can be turned against its user by anyone
who writes to them. The audit, the Reader and Ask only need to read.

## Decision

Quietpost opens mailboxes with `EXAMINE` and fetches with `BODY.PEEK`, so
reading doesn't even mark a message as read. It sends no IMAP command that
changes a mailbox (such as `STORE`, `COPY`, `MOVE`, `EXPUNGE`, `APPEND` or
`DELETE`), and it has no SMTP code. A Takeout `.mbox` file is read where it is
and never written.

The two actions that do reach outside are separate, and each needs your click.
An unsubscribe sends the one-click request that RFC 8058 defines to the HTTPS
link in the sender's `List-Unsubscribe` header. A data-deletion request is text you copy, or a
`mailto:` link that your own mail program opens. Neither sends mail.

## Consequences

- No text in an email can make Quietpost send, delete or move mail, because the
  code to do it doesn't exist.
- A test mail server records every command Quietpost sends, and the test fails
  on any command outside a fixed list.
- Quietpost can't archive, label or mark mail as read for you.
