let counter = 0;
/** Event identity only, never an authentication token or a catalog identifier. */
export function newPurchaseRecordId(prefix: string): string {
  try {
    if (typeof globalThis.crypto?.randomUUID === 'function')
      return `${prefix}-${globalThis.crypto.randomUUID()}`;
  } catch {
    /* Older native runtimes can use the bounded fallback. */
  }
  return `${prefix}-${Date.now().toString(36)}-${++counter}-${Math.random().toString(36).slice(2, 12)}`;
}
