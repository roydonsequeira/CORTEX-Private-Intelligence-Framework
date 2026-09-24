## Summary

<!-- What does this change and why? Link the issue it addresses, if any. -->

## Test plan

- [ ] `python -m pytest tests/`
- [ ] `python -m ruff check src tests` and `python -m mypy src tests`
- [ ] `cd ui && npm run lint && npm run typecheck && npm run build` (if the UI changed)
- [ ] Tried it against a running model (`cortex serve`), if agent behaviour changed

## Checklist

- [ ] Tests added or updated for new behaviour or fixed bugs
- [ ] Docs updated (README, CHANGELOG `[Unreleased]`) if behaviour or setup changed
- [ ] No secrets, personal data, or runtime data (`.cortex/`) in the diff
