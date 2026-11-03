/** @jest-environment node */
jest.mock('../../lib/supabase', () => ({ supabase: { from: jest.fn() } }));
import { writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { deriveFitProfile } from '../../src/services/purchase/fitEvidence';
import { buyCheck } from '../../src/services/purchase/buyCheck';
import { maximumWardrobeGain } from '../../src/services/purchase/wardrobeTools';
import { fit, shoe, owned, context, row, now } from './fixtures';
test('measurement harness records only synthetic counts and timings while enforcing bounded work', () => {
  const evidence = Array.from({ length: 160 }, (_, i) => fit('synthetic-' + i));
  let start = performance.now();
  const profile = deriveFitProfile(evidence, now),
    fitMs = performance.now() - start;
  const wardrobe = [
    ...owned,
    row('blue', 'Light blue cotton oxford shirt', 'upper_body', 0),
    row('beige', 'Beige chino trousers', 'lower_body', 0),
  ];
  start = performance.now();
  const check = buyCheck({ ...shoe, availableSizes: ['42'] }, wardrobe, {
      ...context,
      fitProfile: profile,
    }),
    checkMs = performance.now() - start;
  const candidates = Array.from({ length: 48 }, (_, i) =>
    row(
      'candidate-' + i,
      i % 2 ? 'Brown leather derby shoes' : 'Brown leather penny loafers',
      'shoes',
      60 + i
    )
  );
  start = performance.now();
  const gain = maximumWardrobeGain(candidates, wardrobe, context),
    gainMs = performance.now() - start;
  expect(gain.evaluated).toBeLessThanOrEqual(16);
  expect(check.wardrobe.usefulOutfitUnlocks.evaluations).toBeLessThanOrEqual(416);
  writeFileSync(
    '/tmp/aiwardrobe-phase2-performance.json',
    JSON.stringify({
      synthetic: true,
      evidenceRows: evidence.length,
      ownedRows: wardrobe.length,
      fitMs,
      checkMs,
      gainMs,
      evaluatedCandidates: gain.evaluated,
      checkEvaluations: check.wardrobe.usefulOutfitUnlocks.evaluations,
    })
  );
});
