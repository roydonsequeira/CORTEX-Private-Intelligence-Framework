# CORTEX workspace

This folder is the only place CORTEX's tools can read and write files. Put
documents here that you want the agent to read, search or summarise. Files it
creates (for example, "save that to notes.md") land here too.

- Hidden files such as `.env` are never read, searched or written.
- Git ignores everything here except this README, so your files stay local.
- To use a different folder, set `allowed_root` in `cortex.yaml`.
