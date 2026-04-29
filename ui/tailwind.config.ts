import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        surface: "var(--surface)",
        border: "var(--border)",
        accent: "var(--accent)",
        primary: "var(--text-primary)",
        muted: "var(--text-muted)"
      },
      fontFamily: {
        mono: ["var(--font-jetbrains)", "monospace"],
        sans: ["var(--font-inter)", "sans-serif"]
      },
      transitionTimingFunction: {
        cortex: "ease-out"
      }
    }
  },
  plugins: [require("@tailwindcss/forms")]
};

export default config;
