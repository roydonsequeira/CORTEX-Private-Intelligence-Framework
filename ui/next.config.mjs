/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // Under an AI coding agent, `next dev` would write AGENTS.md and CLAUDE.md
  // into ui/, which then show up as untracked files after every dev run.
  agentRules: false
};

export default nextConfig;
