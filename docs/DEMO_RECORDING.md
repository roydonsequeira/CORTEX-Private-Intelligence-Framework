# Recording the CORTEX demo GIF

A short screen recording is the most reliable way to show CORTEX off — it always
works, needs no hosting, and drops straight into the README. Aim for **under a minute**
for the README GIF; a longer walkthrough works better as a linked MP4.

## 1. Bring up a fast local stack

Use the low-resource stack so responses are snappy on any machine:

```bash
make demo
make pull-models-demo
```

Then **warm the model** with one throwaway query (the first call loads it and is
slow) before you hit record:

```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","session_id":null}'
```

## 2. What to show (in order)

1. **Streaming** — in the UI (http://localhost:3000), ask a simple question and show
   tokens streaming in live, with the plan / step / tool events in the trace panel.
2. **A sandboxed tool call** — ask something that needs computation, e.g.
   *"Use Python to compute the 20th Fibonacci number."* Show `python_exec` running in
   the sandbox and the result feeding back into the answer.
3. **Memory** — tell it a fact, then ask a follow-up that relies on it; show retrieval
   from memory. Open the memory-search panel.
4. **Observability** — open Jaeger (http://localhost:16686) and show the trace tree:
   planning → model calls → tool execution → memory, each a timed span.

Keep prompts short. Have this recording as a fallback even if you also demo live.

## 3. Capture and export

- **Windows:** [ScreenToGif](https://www.screentogif.com/) records straight to GIF and
  lets you trim frames.
- **macOS:** record with QuickTime (or `⇧⌘5`), then convert:
  `ffmpeg -i demo.mov -vf "fps=12,scale=1280:-1:flags=lanczos" demo.gif`
- **Linux:** [Peek](https://github.com/phw/peek), or `ffmpeg` as above.

Keep the GIF under ~10 MB and trim dead air. For a longer or higher-quality version,
keep an MP4 alongside and link it. Lessons from the current recording:

- **Record in a private window.** Browser extensions draw over the page — Grammarly
  pops an icon over the chat every time you paste a prompt.
- **Crop to the chat column (~660 px wide).** GitHub shows README images at most
  ~880 px wide, so a full-screen capture is shrunk until its text is unreadable.
- **Use frame delays of 20 ms or more.** Browsers slow any GIF frame of 10 ms or less
  down to 100 ms, so a GIF "sped up" with 10 ms frames plays several times slower
  on GitHub than in the editor. Speed up waiting by dropping frames instead.
- **Hold each finished answer for 2–3 seconds** so it can be read.

## 4. Wire it into the README

The README embeds `docs/assets/demo.gif` right under the badges. Replace that file
to update the demo; keep it under ~10 MB so it renders inline.
