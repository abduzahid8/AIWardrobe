import { scheduleIdleTask } from '../src/lib/scheduleIdleTask';

/** Evaluate common screen modules one at a time while idle. No screens mount
 * and no screen effects run until navigation opens them. */
export function scheduleScreenPreloads(loaders: Array<() => unknown>, initialDelayMs = 1000): () => void {
  let stopped = false;
  let next = 0;
  let cancelPending: (() => void) | undefined;
  const schedule = (delay: number) => {
    cancelPending = scheduleIdleTask(() => { void run(); }, delay);
  };
  const run = async () => {
    if (stopped) return;
    try { await loaders[next++](); } catch { /* Navigation can retry imports. */ }
    if (!stopped && next < loaders.length) schedule(500);
  };
  if (loaders.length) schedule(initialDelayMs);
  return () => {
    stopped = true;
    cancelPending?.();
  };
}
