/**
 * One-shot "app finished initializing" signal.
 *
 * RootNavigator calls markAppReady() once its gate passes; SplashTransition
 * subscribes via onAppReady() and owns the handoff from the native splash
 * into the app (hide + fade). Keeps the overlay wiring out of the
 * navigator's render tree.
 */

type Listener = () => void;

const listeners = new Set<Listener>();
let isReady = false;

/** Fire the app-ready signal (idempotent). */
export function markAppReady(): void {
  if (isReady) return;
  isReady = true;
  listeners.forEach((listener) => listener());
  listeners.clear();
}

/** Subscribe to app-ready. Fires immediately if the signal already fired. */
export function onAppReady(listener: Listener): () => void {
  if (isReady) {
    listener();
    return () => undefined;
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Test-only: reset the signal between test cases. */
export function resetAppReadyForTests(): void {
  isReady = false;
  listeners.clear();
}
