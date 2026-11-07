/** A real minimum delay, followed by idle time. Cancellation also removes the
 * idle callback, so unmounting never leaves a startup job behind. */
export function scheduleIdleTask(task: () => void, delayMs: number): () => void {
  let cancelled = false;
  let idle: number | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const request = () => {
    if (cancelled) return;
    if (typeof requestIdleCallback === 'function') {
      idle = requestIdleCallback(deadline => {
        idle = undefined;
        if (cancelled) return;
        if (deadline.timeRemaining() < 8) {
          timer = setTimeout(request, 50);
          return;
        }
        task();
      });
    } else {
      task();
    }
  };
  timer = setTimeout(request, delayMs);
  return () => {
    cancelled = true;
    if (timer !== undefined) clearTimeout(timer);
    if (idle !== undefined && typeof cancelIdleCallback === 'function') cancelIdleCallback(idle);
  };
}
