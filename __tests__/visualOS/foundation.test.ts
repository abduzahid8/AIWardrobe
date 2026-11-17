/**
 * foundation.test.ts — assets/provenance, evidence/correction, quality
 * gate, provider contracts, inference cache.
 */
import { createAsset, deriveAsset, rootAssetId, isProductIdentity, isSupportedMime } from '../../src/services/vision/visualAssets';
import { createVisualEvidence, canonicalVisualValue, applyUserCorrection, unconfirmedVisualEvidence, sourceRank } from '../../src/services/vision/visualEvidence';
import { assessImageQuality } from '../../src/services/vision/qualityGate';
import { DeterministicVisionProvider, MockVisionProvider, setVisionProvider, getVisionProvider, isSafeImageUrl } from '../../src/services/vision/providers';
import { cacheKey, getCached, setCached, invalidateAsset, cacheStats, clearInferenceCache } from '../../src/services/vision/inferenceCache';

const asset = () => createAsset({ kind: 'original_upload', source: 'user', mimeType: 'image/jpeg', localUri: 'file://a.jpg' });

test('transformed assets link to original; thumbnails are not product identity', () => {
  const orig = asset();
  const thumb = deriveAsset(orig, 'thumbnail', { localUri: 'file://t.jpg' });
  expect(rootAssetId(thumb)).toBe(orig.id);
  expect(thumb.transformationChain).toEqual(['original_upload', 'thumbnail']);
  expect(isProductIdentity(orig)).toBe(true);
  expect(isProductIdentity(thumb)).toBe(false);
  expect(isProductIdentity(deriveAsset(orig, 'tryon_result'))).toBe(false);
  expect(isSupportedMime('image/png')).toBe(true);
  expect(isSupportedMime('image/tiff')).toBe(false);
});

test('user correction supersedes inference and wins canonical', () => {
  const a = asset();
  const model = createVisualEvidence({ userId: 'u', sourceAssetId: a.id, entityType: 'wardrobe_item', attributeType: 'color', value: 'navy', confidence: 0.8, source: 'remote_model', modelVersion: 'r1' });
  expect(canonicalVisualValue([model])?.value).toBe('navy');
  const { rows, winner } = applyUserCorrection([model], { userId: 'u', sourceAssetId: a.id, entityType: 'wardrobe_item', attributeType: 'color', value: 'black' });
  expect(winner.source).toBe('user');
  expect(rows.find((r) => r.id === model.id)?.supersededBy).toBe(winner.id);
  expect(canonicalVisualValue(rows)?.value).toBe('black');
  // user outranks merchant outranks model
  const merchant = createVisualEvidence({ userId: 'u', sourceAssetId: a.id, entityType: 'catalog_product', attributeType: 'brand', value: 'uniqlo', confidence: 0.9, source: 'merchant' });
  expect(sourceRank('user')).toBeLessThan(sourceRank('merchant'));
  expect(sourceRank('merchant')).toBeLessThan(sourceRank('remote_model'));
  const weak = createVisualEvidence({ userId: 'u', sourceAssetId: a.id, entityType: 'wardrobe_item', attributeType: 'pattern', value: 'striped', confidence: 0.4, source: 'local_model', modelVersion: 'l1' });
  expect(unconfirmedVisualEvidence([model, merchant, winner, weak])).toHaveLength(1);
});

test('quality gate: dark, blur, occlusion, multi-item, degraded, unusable', () => {
  expect(assessImageQuality({ mimeType: 'image/jpeg', brightness01: 0.05 }).usability).toBe('degraded');
  expect(assessImageQuality({ mimeType: 'image/jpeg', blurScore01: 0.9 }).usability).toBe('unusable');
  const occ = assessImageQuality({ mimeType: 'image/jpeg', occluded: true, mirrorObstruction: true });
  expect(occ.usability).toBe('degraded');
  expect(occ.reasons).toContain('heavy occlusion');
  expect(assessImageQuality({ mimeType: 'image/jpeg', multipleGarments: true }).reasons).toContain('multiple garments');
  expect(assessImageQuality({ mimeType: 'image/tiff' }).usability).toBe('unusable');
  const good = assessImageQuality({ mimeType: 'image/jpeg', brightness01: 0.6, blurScore01: 0.1, garmentFrameShare01: 0.5 });
  expect(good.usability).toBe('usable');
  expect(good.recommendedCorrection).toBeNull();
});

test('deterministic provider never invents claims; mock injectable', async () => {
  setVisionProvider(new DeterministicVisionProvider({ category: 'top', primaryColor: 'navy' }));
  const p = await getVisionProvider().analyzeGarment(asset(), { userId: 'u' });
  expect(p.category).toBe('top');
  expect(p.materialClass).toBe('unknown');
  expect(p.confidencePerField.material).toBe(0);
  setVisionProvider(new MockVisionProvider({ garmentType: 'blazer', confidencePerField: { garment_type: 0.95 } }));
  const m = await getVisionProvider().analyzeGarment(asset(), { userId: 'u' });
  expect(m.garmentType).toBe('blazer');
  setVisionProvider(new DeterministicVisionProvider());
});

test('cache keyed by asset+task+version; invalidation on correction', () => {
  clearInferenceCache();
  const k = cacheKey({ assetId: 'a', task: 'parse-garment', modelVersion: 'v1' });
  expect(getCached(k)).toBeNull();
  setCached(k, 'parse-garment', 'v1', { ok: true });
  expect(getCached(k)).toEqual({ ok: true });
  expect(cacheStats().size).toBe(1);
  expect(invalidateAsset('a')).toBe(1);
  expect(getCached(k)).toBeNull();
});

test('SSRF guards reject unsafe image URLs', () => {
  expect(isSafeImageUrl('https://cdn.example.com/p.jpg')).toBe(true);
  expect(isSafeImageUrl('http://x.com/p.jpg')).toBe(false);
  expect(isSafeImageUrl('https://localhost/p.jpg')).toBe(false);
  expect(isSafeImageUrl('not-a-url')).toBe(false);
});
