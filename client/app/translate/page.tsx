"use client";

// Live translation lives on the home page now; this stub keeps any old
// bookmarks working.

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function TranslateRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/");
  }, [router]);
  return (
    <main className="min-h-screen flex items-center justify-center text-sm opacity-60 bg-neutral-950 text-neutral-100">
      Live translation moved to <code className="ml-1">/</code>. Redirecting…
    </main>
  );
}
