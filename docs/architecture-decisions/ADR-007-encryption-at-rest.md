# ADR-007: Stored Mail Relies on Full-Disk Encryption

Status: accepted, 26 Nov 2026. Not built yet.

## Context

The inbox features will store message text, a full-text index and embeddings on
your machine ([the design note](../design/inbox.md)). On a lost or stolen laptop,
anyone who can read the disk can read them. There are three options:

1. Rely on the operating system's full-disk encryption: BitLocker or Device
   Encryption on Windows, FileVault on macOS, LUKS on Linux.
2. Encrypt the mail database with SQLCipher, with the key in the operating
   system's keyring.
3. Keep only derived data, and drop message text after the analysis.

Option 2 protects a laptop whose disk isn't encrypted: on Windows and macOS the
keyring is locked by your login, so a copy of the disk doesn't give the key
without your password. But it leaves Chroma's files readable, and Ask's
collection holds email text and embeddings there. Covering them would mean
keeping the text out of Chroma, and embeddings still reveal some of what the
text says. On Linux the Secret Service isn't always running, so the key would
need a fallback. Option 2 also adds a compiled dependency, SQLCipher's Python
bindings, whose wheels and FTS5 support would have to be checked on three
operating systems and Python 3.12 to 3.14.

Option 3 breaks the source shown beside each figure and Ask's citations, unless
the text is read again from where it came from: the `.mbox` file would have to
stay in place, and IMAP would need the network and the password for every look.
The full-text index would still hold the words.

## Decision

Option 1, with these additions:

- `quietpost doctor` and the Trust page say whether the drive that holds the
  data folder is encrypted, where the operating system lets a normal program
  find out, and how to turn encryption on when it's off.
- The store keeps only what the features need: text, not HTML, attachments or
  images.
- One command deletes the mail store, its search index and its Chroma
  collection.

Look at option 2 again if users ask for a store locked with its own passphrase,
or if Chroma can be kept free of email text.

## Consequences

- On a laptop without full-disk encryption, stored mail is readable by anyone
  who has the disk, and Quietpost says so.
- No new compiled dependency, and no key whose loss would lock the data.
