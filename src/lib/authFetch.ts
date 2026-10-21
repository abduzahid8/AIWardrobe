import { withRequestDeadline } from './requestDeadline';

export const AUTH_REQUEST_TIMEOUT_MS = 15_000;
export const STORAGE_UPLOAD_TIMEOUT_MS = 60_000;
function cancellationError() {
  const error = new Error('The request was cancelled.');
  error.name = 'AbortError';
  return error;
}

/** Bound this project's auth transport, including its JSON body, without retrying mutations. */
export function createAuthDeadlineFetch(projectUrl: string, transport: typeof fetch, timeoutMs = AUTH_REQUEST_TIMEOUT_MS): typeof fetch {
  return createScopedDeadlineFetch(projectUrl, transport, timeoutMs, url => url.pathname.startsWith('/auth/v1/'));
}

/** Bound object uploads, including acknowledgement bodies, without replaying writes. */
export function createStorageUploadDeadlineFetch(projectUrl: string, transport: typeof fetch, timeoutMs = STORAGE_UPLOAD_TIMEOUT_MS): typeof fetch {
  return createScopedDeadlineFetch(projectUrl, transport, timeoutMs, (url, method) =>
    (method === 'POST' || method === 'PUT') && /^\/storage\/v1\/object\/wardrobe-images\//.test(url.pathname));
}

function createScopedDeadlineFetch(projectUrl: string, transport: typeof fetch, timeoutMs: number, matches: (url: URL, method: string) => boolean): typeof fetch {
  const origin = new URL(projectUrl).origin;
  return async (input, init) => {
    const rawUrl = typeof input === 'string' ? input : 'url' in input ? input.url : input.href;
    const method = (init?.method ?? (typeof input === 'object' && 'method' in input ? input.method : 'GET')).toUpperCase();
    let boundedRequest = false;
    try { const url = new URL(rawUrl); boundedRequest = url.origin === origin && matches(url, method); } catch { /* Preserve transport handling of invalid URLs. */ }
    if (!boundedRequest) return transport(input, init);
    const incoming = init?.signal ?? (typeof input === 'object' && 'signal' in input ? input.signal : undefined);
    if (incoming?.aborted) throw cancellationError();
    const controller = new AbortController();
    let rejectCancellation!: (error: Error) => void;
    const cancelled = new Promise<never>((_, reject) => { rejectCancellation = reject; });
    const cancel = () => { controller.abort(); rejectCancellation(cancellationError()); };
    incoming?.addEventListener('abort', cancel, { once: true });
    let removeDeadlineListener: () => void = () => undefined;
    try {
      return await withRequestDeadline(deadlineSignal => {
        const abort = () => controller.abort();
        deadlineSignal.addEventListener('abort', abort, { once: true });
        removeDeadlineListener = () => deadlineSignal.removeEventListener('abort', abort);
        const request = (async () => {
          const response = await transport(input, { ...init, signal: controller.signal });
          // These responses are small JSON documents. Drain a clone so stalled
          // response bodies cannot escape the deadline before the SDK reads them.
          await response.clone().arrayBuffer();
          return response;
        })();
        return Promise.race([request, cancelled]);
      }, timeoutMs);
    } finally {
      incoming?.removeEventListener('abort', cancel);
      removeDeadlineListener();
    }
  };
}
