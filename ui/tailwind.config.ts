import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ground: "var(--ground)",
        panel: "var(--panel)",
        raised: "var(--raised)",
        line: "var(--line)",
        ink: "var(--ink)",
        "ink-dim": "var(--ink-dim)",
        "ink-faint": "var(--ink-faint)",
        signal: "var(--signal)",
        ok: "var(--ok)",
        bad: "var(--bad)",
        think: "var(--think)",
        tool: "var(--tool)"
      },
      fontFamily: {
        mono: ["var(--font-mono)"],
        sans: ["var(--font-sans)"]
      },
      borderRadius: {
        card: "10px"
      },
      transitionTimingFunction: {
        out: "cubic-bezier(0.16, 1, 0.3, 1)"
      }
    }
  },
  plugins: [require("@tailwindcss/forms")]
};

export default config;
