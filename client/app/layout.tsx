import type { Metadata } from "next";
import "./globals.css";
import { THEME_INIT_SCRIPT } from "../lib/theme";

export const metadata: Metadata = {
  title: "Phonemenal",
  description: "Live speech translator for brain injury patients.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          // Runs before paint to set data-theme from localStorage, preventing
          // a dark→light flash on reload when the user has chosen light mode.
          dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }}
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
