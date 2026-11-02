type StartupPerformance = Performance & {
  rnStartupTiming?: { startTime?: number | null; initializeRuntimeStart?: number | null };
};

/** Native RN startup through the completely revealed first screen, in ms.
 * Enable EXPO_PUBLIC_MEASURE_STARTUP=true in a Release build to collect it.
 * No account data is included, and no network/storage work runs on this path.
 */
export function recordStartupVisible(): void {
  const perf = globalThis.performance as StartupPerformance | undefined;
  if (!perf) return;
  const timing = perf.rnStartupTiming;
  const start = timing?.startTime ?? timing?.initializeRuntimeStart;
  if (start == null) return;
  const durationMs = Math.round(perf.now() - start);
  if (process.env.EXPO_PUBLIC_MEASURE_STARTUP === 'true') {
    const result = { durationMs, budgetMs: 1000, passed: durationMs <= 1000,
      origin: timing?.startTime != null ? 'native-app-start' : 'native-runtime-start' };
    console.warn('[STARTUP_VISIBLE]', JSON.stringify(result));
    // Benchmark tooling reads this from the simulator container after reveal.
    void import('expo-file-system').then(({ File, Paths }) => {
      new File(Paths.document, 'startup-timing.json').write(JSON.stringify(result));
    }).catch(() => undefined);
  }
}
