"use client";

// The TORGO demo is now a drawer on the home page; this stub redirects
// any old bookmarks and signals the home page to open the drawer.

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DemoRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/?demo=1");
  }, [router]);
  return (
    <main className="min-h-screen flex items-center justify-center text-sm opacity-60 bg-neutral-950 text-neutral-100">
      TORGO demo moved into the home page. Redirecting…
    </main>
  );
}
