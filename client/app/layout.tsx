import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Phonemer",
  description: "Live speech translator for brain injury patients.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
