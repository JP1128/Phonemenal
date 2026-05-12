import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Hide Next.js's in-page dev overlay (the floating "N" / build-activity
  // indicator). It collides with Phonemer's own status badges in the
  // corners and the listener-facing translate view should look clean.
  devIndicators: false,
};

export default nextConfig;
