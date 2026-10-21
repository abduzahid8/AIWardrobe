export const DATA_REQUEST_TIMEOUT_MS = 15_000;

export class RequestTimeoutError extends Error {
  constructor() {
    super('The request timed out. Please try again.');
    this.name = 'RequestTimeoutError';
  }
}

/** Settle the caller on time and cancel cooperative transports. Late results lose the race. */
export async function withRequestDeadline<T>(start: (signal: AbortSignal) => PromiseLike<T>, timeoutMs = DATA_REQUEST_TIMEOUT_MS, cancelSignal?: AbortSignal): Promise<T> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid request deadline');
  const controller = new AbortController();
  const cancelledError = () => Object.assign(new Error('The request was cancelled.'), {name: 'AbortError'});
  if (cancelSignal?.aborted) throw cancelledError();
  let onCancel: () => void;
  const cancelled = new Promise<never>((_, reject) => {
    onCancel = () => { reject(cancelledError()); controller.abort(); };
    cancelSignal?.addEventListener('abort', onCancel, {once: true});
  });
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      reject(new RequestTimeoutError());
      controller.abort();
    }, timeoutMs);
  });
  try { return await Promise.race([start(controller.signal), deadline, cancelled]); }
  finally { clearTimeout(timer!); cancelSignal?.removeEventListener('abort', onCancel!); }
}
