/** Host-only bounded stress evidence; cannot measure device image decoding or UI memory. */
import { performance } from 'perf_hooks';
import { writeFileSync, mkdirSync } from 'fs';
import { buyCheck } from '../../src/services/purchase/buyCheck';
import {
  cacheKey,
  setCached,
  getCached,
  cacheStats,
  clearInferenceCache,
} from '../../src/services/vision/inferenceCache';
import { createBoard, boardSnapshot } from '../../src/services/vision/outfitBoard';
import { owned, shoe, context } from '../purchase/fixtures';
test('220-item wardrobe and repeated BuyChecks remain bounded with valid evidence', () => {
  const wardrobe = Array.from({ length: 220 }, (_, i) => ({
    ...owned[i % owned.length],
    id: `stress-${i}`,
  }));
  const samples: number[] = [],
    before = process.memoryUsage().heapUsed;
  for (let i = 0; i < 25; i++) {
    const start = performance.now(),
      result = buyCheck(shoe, wardrobe, context);
    samples.push(performance.now() - start);
    expect(result.wardrobe.usefulOutfitUnlocks.looks.length).toBeLessThanOrEqual(120);
    expect(result.wardrobe.duplicateRisk.substitutes.length).toBeLessThanOrEqual(220);
    expect(result.fit.confidence).toBe('low');
  }
  const sorted = [...samples].sort((a, b) => a - b),
    report = {
      schemaVersion: 1,
      device: 'Node host',
      buildType: 'Jest transformed TypeScript; non-release',
      method: 'performance.now around real buyCheck; 25 sequential calls; 220 synthetic items',
      samples: samples.length,
      p50Ms: sorted[Math.floor(sorted.length / 2)],
      rangeMs: [sorted[0], sorted.at(-1)],
      heapDeltaBytes: process.memoryUsage().heapUsed - before,
      heapMethod: 'process.memoryUsage, no forced GC; diagnostic only',
      productionClaim: false,
      notMeasured: ['startup', 'native images', 'scroll', 'background', 'VTON result decoding'],
    };
  if (process.env.AIWARDROBE_WRITE_STRESS_REPORT === '1') {
    mkdirSync('docs/launch/evidence', { recursive: true });
    writeFileSync('docs/launch/evidence/host-stress.json', JSON.stringify(report, null, 2) + '\n');
  }
});

test('reference metadata cache evicts at 300 entries and outfit history snapshots cap at ten', () => {
  clearInferenceCache();
  for (let i = 0; i < 350; i++) {
    const key = cacheKey({
      assetId: `synthetic-reference-${i}`,
      task: 'parse-garment',
      modelVersion: 'local-v1',
    });
    setCached(key, 'parse-garment', 'local-v1', { kind: 'shirt', referenceIndex: i });
  }
  expect(cacheStats().size).toBe(300);
  expect(
    getCached(
      cacheKey({
        assetId: 'synthetic-reference-0',
        task: 'parse-garment',
        modelVersion: 'local-v1',
      })
    )
  ).toBeNull();
  let board = createBoard('synthetic-owner', ['shirt', 'pants', 'shoes', 'layer']);
  for (let i = 0; i < 40; i++) board = boardSnapshot(board, `v${i}`);
  expect(board.versions).toHaveLength(10);
  clearInferenceCache();
});
