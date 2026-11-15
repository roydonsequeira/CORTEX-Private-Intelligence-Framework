import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Quietpost",
  description: "A private AI chief of staff that runs on your own machine."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-ground font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
