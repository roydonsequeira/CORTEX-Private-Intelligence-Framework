// Builds the static UI that ships inside the quietpost Python package and copies
// it to src/quietpost/web/, where the Quietpost API serves it on its own address.
// The API address is left empty, so the page calls the server it came from.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { cp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const ui = fileURLToPath(new URL("..", import.meta.url));
const out = fileURLToPath(new URL("../out", import.meta.url));
const target = fileURLToPath(new URL("../../src/quietpost/web", import.meta.url));

// Next.js doesn't empty out/ first, so files from an older build could reach the package.
await rm(out, { recursive: true, force: true });
const build = spawnSync(process.execPath, [require.resolve("next/dist/bin/next"), "build"], {
  cwd: ui,
  stdio: "inherit",
  env: { ...process.env, QUIETPOST_UI_BUNDLE: "1", NEXT_PUBLIC_QUIETPOST_API_URL: "" }
});
if (build.status !== 0) {
  process.exit(build.status ?? 1);
}
if (!existsSync(`${out}/index.html`)) {
  console.error(`The static export did not produce ${out}/index.html.`);
  process.exit(1);
}
// Not cpSync: on Windows, Node 24 exits silently with code 127 when the path has
// non-ASCII characters (such as an em dash in a folder name). The async copy works.
await rm(target, { recursive: true, force: true });
// public/.gitkeep only keeps the empty folder in git.
await cp(out, target, { recursive: true, filter: (source) => basename(source) !== ".gitkeep" });
console.log(`Copied the UI to ${target}`);
