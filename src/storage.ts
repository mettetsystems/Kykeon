export function readStorage(key: string): string | null {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string): string | null {
  try {
    if (!globalThis.localStorage) return "Browser storage is unavailable.";
    globalThis.localStorage.setItem(key, value);
    return null;
  } catch (error) {
    return error instanceof Error
      ? error.message
      : "Browser storage is unavailable or full.";
  }
}
