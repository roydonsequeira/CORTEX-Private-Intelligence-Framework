// The page may talk only to itself and the CORTEX API. Even if text the model
// read got markup into an answer, the browser would refuse to send anything
// elsewhere: no requests to other hosts, no remote images, no form posts, no
// plugins, and no other site can frame the UI.
const apiOrigin = new URL(process.env.NEXT_PUBLIC_CORTEX_API_URL ?? "http://localhost:8000").origin;
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

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  agentRules: false,
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: contentSecurityPolicy },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" }
        ]
      }
    ];
  }
};

export default nextConfig;
