// Two builds:
// - `npm run build`: a standalone Node.js server, for Docker and development.
// - `npm run build:bundle`: static files that ship inside the quietpost Python
//   package and are served by the Quietpost API itself, so nobody needs Node.js.
//   The API then sends the security headers below (src/quietpost/api/web.py).
const bundle = process.env.QUIETPOST_UI_BUNDLE === "1";

// The page may talk only to itself and the Quietpost API. Even if text the model
// read got markup into an answer, the browser would refuse to send anything
// elsewhere: no requests to other hosts, no remote images, no form posts, no
// plugins, and no other site can frame the UI.
function securityHeaders() {
  const apiUrl =
    process.env.NEXT_PUBLIC_QUIETPOST_API_URL ??
    process.env.NEXT_PUBLIC_CORTEX_API_URL ?? // the name from before the rename
    "http://localhost:8000";
  const apiOrigin = new URL(apiUrl).origin;
  const isDev = process.env.NODE_ENV !== "production";
  const contentSecurityPolicy = [
    "default-src 'self'",
    // Next.js inlines its bootstrap scripts; the dev server also needs eval for hot reload.
    `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    `connect-src 'self' ${apiOrigin}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'"
  ].join("; ");
  return [
    { key: "Content-Security-Policy", value: contentSecurityPolicy },
    { key: "X-Content-Type-Options", value: "nosniff" },
    { key: "Referrer-Policy", value: "no-referrer" },
    { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" }
  ];
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: bundle ? "export" : "standalone",
  agentRules: false,
  poweredByHeader: false,
  // A static export has no server to send headers; the API sends them instead.
  ...(bundle
    ? {}
    : {
        async headers() {
          return [{ source: "/:path*", headers: securityHeaders() }];
        }
      })
};

export default nextConfig;
