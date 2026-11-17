/**
 * closet.test.ts — parsers, fingerprint, duplicate review, scan queue,
 * regions, digital twin.
 */
import { createAsset } from '../../src/services/vision/visualAssets';
import { parseGarment, parseOutfit } from '../../src/services/vision/parsers';
import { MockVisionProvider, DeterministicVisionProvider, setVisionProvider } from '../../src/services/vision/providers';
import { clearInferenceCache } from '../../src/services/vision/inferenceCache';
import { fingerprintVerdict, reviewDuplicate } from '../../src/services/vision/fingerprint';
import { enqueueScan, advanceScan, pendingScans, manualRegion, providerRegion } from '../../src/services/vision/closetScan';
import { aggregateTwinItem, conditionRisks, itemsByVisualRole, bestVisualCompanions, closestOwnedAlternatives } from '../../src/services/vision/twin';
import { createVisualEvidence } from '../../src/services/vision/visualEvidence';

const asset = (kind: 'original_upload' | 'outfit_photo' = 'original_upload') =>
  createAsset({ kind, source: 'user', mimeType: 'image/jpeg' });

beforeEach(() => { clearInferenceCache(); setVisionProvider(new DeterministicVisionProvider()); });

test('garment parser: missing fields stay unknown; exact claims rejected', async () => {
  const p = await parseGarment(asset(), { userId: 'u' });
  expect(p.garmentType).toBe('unknown');
  expect(p.conditionHint).toBe('unknown');
  setVisionProvider(new MockVisionProvider({ materialClass: '100% cashmere size M by Gucci', confidencePerField: { material: 0.9 } }));
  const q = await parseGarment(asset(), { userId: 'u' });
  expect(q.materialClass).toBe('unknown');
  expect(q.confidencePerField.material).toBe(0);
});

test('outfit parser: layered outfit, missing shoes, low-confidence fallback', async () => {
  setVisionProvider(new MockVisionProvider({}, {
    upperBody: 'blazer', lowerBody: 'trousers', outerwear: 'overcoat',
    shoes: null, accessories: ['belt'], dominantPalette: ['navy', 'grey'],
    contrast: 'low', layering: 3, formality: 0.8, confidence: 0.85,
  }));
  const o = await parseOutfit(asset('outfit_photo'), { userId: 'u' });
  expect(o.outerwear).toBe('overcoat');
  expect(o.shoes).toBeNull();
  expect(o.layering).toBe(3);
  setVisionProvider(new DeterministicVisionProvider());
  const fallback = await parseGarment(asset(), { userId: 'u' });
  expect(fallback.provenance).toMatch(/deterministic/);
});

test('fingerprint verdicts across four cases; weak evidence cannot merge', () => {
  const base = { itemId: 'a', category: 'top', subCategory: 'tee', dominantColors: ['black'], silhouette: 'regular', pattern: 'solid', brand: 'uniqlo', modelKey: 'u-tee-bk' };
  const samePhoto = fingerprintVerdict(base, { ...base, itemId: 'b' });
  expect(samePhoto.verdict).toBe('SAME_ITEM_HIGH_CONFIDENCE');
  expect(samePhoto.requiresConfirmation).toBe(true);
  const diffColor = fingerprintVerdict(base, { ...base, itemId: 'c', dominantColors: ['white'], modelKey: undefined, brand: undefined });
  expect(['SAME_FAMILY', 'VISUALLY_SIMILAR']).toContain(diffColor.verdict);
  const similar = fingerprintVerdict(base, { itemId: 'd', category: 'top', subCategory: 'hoodie', dominantColors: ['black'], silhouette: 'boxy', pattern: 'striped' });
  expect(similar.verdict).toBe('VISUALLY_SIMILAR');
  const distinct = fingerprintVerdict(base, { itemId: 'e', category: 'shoes', dominantColors: ['brown'] });
  expect(distinct.verdict).toBe('DISTINCT');
  expect(reviewDuplicate(distinct, 'merge', 'a', 'e')).toBeNull();
  expect(reviewDuplicate(similar, 'merge', 'a', 'd')?.relation).toBeUndefined();
  const ok = reviewDuplicate(samePhoto, 'merge', 'a', 'b');
  expect(ok).toMatchObject({ keepId: 'a', mergedId: 'b', relation: 'merge' });
  expect(reviewDuplicate(samePhoto, 'keep_separate', 'a', 'b')?.relation).toBe('keep_separate');
});

test('scan queue: batch, resume, duplicate prompt, offline fallback', () => {
  const q0 = enqueueScan('u', 'asset1');
  expect(q0.status).toBe('queued');
  const q1 = advanceScan(q0, 'quality_check');
  const q2 = advanceScan(q1, 'awaiting_confirm', { proposedCategory: 'top', proposedColor: 'navy' });
  expect(q2.proposedColor).toBe('navy');
  // illegal jump rejected
  expect(advanceScan(q0, 'saved').status).toBe('queued');
  const dup = advanceScan(q2, 'duplicate_review', { duplicateOfItemId: 'item9' });
  expect(dup.status).toBe('duplicate_review');
  const saved = advanceScan(dup, 'saved');
  expect(saved.status).toBe('saved');
  expect(pendingScans([saved, q0])).toEqual([q0]);
  // offline: queue persists without provider
  const off = enqueueScan('u', 'asset2');
  expect(pendingScans([off])).toHaveLength(1);
});

test('regions: manual fallback is explicit; provider regions carry confidence', () => {
  const parent = asset();
  const m = manualRegion(parent, { x: 0.1, y: 0.1, w: 0.5, h: 0.6 }, 'jacket');
  expect(m.manual).toBe(true);
  expect(m.parentAssetId).toBe(parent.id);
  const p = providerRegion('parent1', { x: 0, y: 0, w: 1, h: 1 }, 0.82, 'trousers');
  expect(p.manual).toBe(false);
  expect(p.confidence).toBe(0.82);
});

test('twin: provenance, conflict resolution, user correction wins', () => {
  const model = createVisualEvidence({ userId: 'u', sourceAssetId: 'a', entityType: 'wardrobe_item', entityId: 'i1', attributeType: 'color', value: 'navy', confidence: 0.7, source: 'remote_model', modelVersion: 'r1' });
  const user = createVisualEvidence({ userId: 'u', sourceAssetId: 'a', entityType: 'wardrobe_item', entityId: 'i1', attributeType: 'color', value: 'black', confidence: 1, source: 'user' });
  const twin = aggregateTwinItem({
    itemId: 'i1', identity: { category: 'top' }, visual: [model, user], conditionEvidence: [],
    fitSummary: 'good', companionIds: ['i2'], substituteIds: ['i3'],
  });
  expect(twin.visualFeatures.color).toEqual(['black']);
  expect(twin.provenance['visual:color']).toMatch(/user/);
  expect(twin.fitSummary).toBe('good');
  const risky = aggregateTwinItem({ itemId: 'i9', identity: {}, visual: [], conditionEvidence: [{ ...model, attributeType: 'condition', value: 'repair' }] });
  expect(conditionRisks([twin, risky]).map((t) => t.itemId)).toEqual(['i9']);
  expect(itemsByVisualRole([twin], 'color', 'black')).toHaveLength(1);
  expect(bestVisualCompanions([twin], 'i1')).toEqual([{ itemId: 'i2', score: 0.5 }]);
  expect(closestOwnedAlternatives([twin], { color: 'black' })).toHaveLength(1);
});
