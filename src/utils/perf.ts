/**
 * perf.ts — Lightweight performance timing utility
 *
 * Provides console.log-based timing for:
 *   - Page/screen transitions (how long until a screen is interactive)
 *   - Button action latency (how long from press to result)
 *
 * Usage:
 *   import { perfMark, perfMeasure, perfAction } from '../src/utils/perf';
 *
 *   // Screen mount timing
 *   perfMark('HomeScreen:mount');
 *   // ... after data loads ...
 *   perfMeasure('HomeScreen:mount', 'HomeScreen:ready');
 *
 *   // Button action timing
 *   const done = perfAction('addToWardrobe');
 *   await addItem(...);
 *   done(); // logs elapsed time
 */

const marks: Record<string, number> = {};

/** Record a named timestamp. */
export function perfMark(name: string): void {
  marks[name] = Date.now();
  console.log(`[PERF] ⏱  mark  "${name}"`);
}

/**
 * Measure elapsed time between two marks (or from a mark to now).
 * Logs the result and returns the elapsed ms.
 */
export function perfMeasure(startMark: string, endMark?: string): number {
  const start = marks[startMark];
  if (start === undefined) {
    console.warn(`[PERF] ⚠️  measure: start mark "${startMark}" not found`);
    return -1;
  }
  const end = endMark ? (marks[endMark] ?? Date.now()) : Date.now();
  const elapsed = end - start;
  const label = endMark ? `"${startMark}" → "${endMark}"` : `"${startMark}" → now`;
  const emoji = elapsed < 300 ? '🟢' : elapsed < 800 ? '🟡' : '🔴';
  console.log(`[PERF] ${emoji} measure ${label}: ${elapsed}ms`);
  return elapsed;
}

/**
 * Start timing a button action. Returns a `done()` function that logs
 * the elapsed time when called.
 *
 * @example
 *   const done = perfAction('toggleFavorite');
 *   await toggleFavorite(id);
 *   done();
 */
export function perfAction(actionName: string): () => number {
  const start = Date.now();
  console.log(`[PERF] ▶️  action "${actionName}" started`);
  return () => {
    const elapsed = Date.now() - start;
    const emoji = elapsed < 100 ? '🟢' : elapsed < 500 ? '🟡' : '🔴';
    console.log(`[PERF] ${emoji} action "${actionName}" completed in ${elapsed}ms`);
    return elapsed;
  };
}

/**
 * Wrap an async function with automatic timing.
 * Logs start, success (with elapsed), and failure.
 */
export async function perfAsync<T>(
  label: string,
  fn: () => Promise<T>,
): Promise<T> {
  const start = Date.now();
  console.log(`[PERF] ▶️  async "${label}" started`);
  try {
    const result = await fn();
    const elapsed = Date.now() - start;
    const emoji = elapsed < 300 ? '🟢' : elapsed < 1000 ? '🟡' : '🔴';
    console.log(`[PERF] ${emoji} async "${label}" resolved in ${elapsed}ms`);
    return result;
  } catch (err) {
    const elapsed = Date.now() - start;
    console.log(`[PERF] 🔴 async "${label}" FAILED after ${elapsed}ms`, err);
    throw err;
  }
}

/**
 * Log tab switch timing. Call from the tab bar's onPress handler.
 * Automatically pairs with the screen's own perfMark to give end-to-end timing.
 */
export function perfTabSwitch(fromTab: string, toTab: string): void {
  const key = `tabSwitch:${fromTab}→${toTab}`;
  marks[key] = Date.now();
  console.log(`[PERF] 🔀 tab switch "${fromTab}" → "${toTab}" initiated`);
}

/**
 * Call when the tab animation has fully settled and the new screen is visible.
 * Logs the total elapsed time from the tab press to the screen being shown.
 * Use in the Tab.Navigator's `screenListeners` → `focus` event.
 */
export function perfTabSwitchComplete(toTab: string): void {
  // Find the most recent tab switch that targeted this screen
  const matchKey = Object.keys(marks)
    .filter((k) => k.startsWith('tabSwitch:') && k.endsWith(`→${toTab}`))
    .sort((a, b) => marks[b] - marks[a])[0];

  if (matchKey) {
    const elapsed = Date.now() - marks[matchKey];
    const emoji = elapsed < 150 ? '🟢' : elapsed < 400 ? '🟡' : '🔴';
    console.log(`[PERF] ${emoji} TAB SWITCH COMPLETE "${toTab}": ${elapsed}ms (press → screen visible)`);
    // Clean up the mark so it doesn't match future switches to the same tab
    delete marks[matchKey];
  }
}

/**
 * Call at the end of a screen's first meaningful render (after data is ready).
 * Pairs with the most recent perfTabSwitch that targeted this screen.
 */
export function perfScreenReady(screenName: string): void {
  // Find the most recent tab switch that targeted this screen
  const matchKey = Object.keys(marks)
    .filter((k) => k.startsWith('tabSwitch:') && k.endsWith(`→${screenName}`))
    .sort((a, b) => marks[b] - marks[a])[0];

  if (matchKey) {
    const elapsed = Date.now() - marks[matchKey];
    const emoji = elapsed < 200 ? '🟢' : elapsed < 600 ? '🟡' : '🔴';
    console.log(`[PERF] ${emoji} screen "${screenName}" ready after tab switch: ${elapsed}ms`);
  } else {
    // No tab switch recorded — just log the screen ready event
    console.log(`[PERF] ✅ screen "${screenName}" ready`);
  }
}
