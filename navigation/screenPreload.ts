/** Evaluate one screen at a time while idle, so a first tap can reuse its module.
 * This only loads code; screens stay unmounted and perform no data requests.
 */
export function scheduleScreenPreloads(loaders: Array<() => unknown>): () => void {
  let stopped = false;
  let next = 0;
  let timer: ReturnType<typeof setTimeout>;
  let idle: number | undefined;
  const schedule = (delay: number) => {
    timer = setTimeout(() => {
      if (stopped) return;
      if (typeof requestIdleCallback === 'function') {
        idle = requestIdleCallback(() => { void run(); });
      } else {
        void run();
      }
    }, delay);
  };
  const run = async () => {
    if (stopped) return;
    try { await loaders[next++](); } catch { /* Navigation can retry a failed import. */ }
    if (!stopped && next < loaders.length) schedule(500);
  };
  if (loaders.length) schedule(1000);
  return () => {
    stopped = true;
    clearTimeout(timer);
    if (idle !== undefined && typeof cancelIdleCallback === 'function') cancelIdleCallback(idle);
  };
}
