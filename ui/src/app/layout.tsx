import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "CORTEX — Local Agent",
  description: "Private Intelligence Framework — intelligence that stays yours."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-ground font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
