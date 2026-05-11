"use client";

export type MicPermissionState =
  | "unknown"
  | "prompt"
  | "granted"
  | "denied"
  | "unsupported";

export async function queryMicPermission(): Promise<MicPermissionState> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices) {
    return "unsupported";
  }
  try {
    // navigator.permissions is best-effort; not every browser ships it.
    const perms = (navigator as unknown as { permissions?: Permissions })
      .permissions;
    if (perms && "query" in perms) {
      const status = await perms.query({ name: "microphone" as PermissionName });
      return status.state as MicPermissionState;
    }
  } catch {
    /* ignore */
  }
  return "unknown";
}

export async function listInputDevices(): Promise<MediaDeviceInfo[]> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices) return [];
  const devs = await navigator.mediaDevices.enumerateDevices();
  return devs.filter((d) => d.kind === "audioinput");
}
