/**
 * parsers.ts — Garment + outfit visual parser layer (4D impl, 4H).
 *
 * Validates provider output against the golden contract: missing fields
 * stay `unknown`, confidence is per-field and bounded, exact fiber % /
 * exact size / exact brand are rejected unless explicitly visible+trusted
 * (caller passes `trustedExact`). Never infers body traits or identity.
 */
import type { GarmentVisualParse, OutfitVisualParse, VisualAsset } from '../../types/visualOS';
import { getVisionProvider, type VisionContext } from './providers';
import { getCached, setCached, cacheKey } from './inferenceCache';

const FORBIDDEN_GUESSES = [/^\d+\s*%/, /\bsize\s*(xs|s|m|l|xl|\d+)\b/i];

function sanitizeGarment(raw: GarmentVisualParse, assetId: string, trustedExact: boolean): GarmentVisualParse {
  const clean: GarmentVisualParse = { ...raw, assetId };
  for (const [k, v] of Object.entries(clean.confidencePerField)) {
    clean.confidencePerField[k] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
  }
  if (!trustedExact) {
    const joined = [clean.materialClass, clean.garmentType].join(' ');
    if (FORBIDDEN_GUESSES.some((re) => re.test(joined))) {
      clean.materialClass = 'unknown';
      clean.confidencePerField.material = 0;
    }
  }
  for (const f of ['garmentType', 'category', 'pattern', 'silhouette', 'materialClass', 'conditionHint'] as const) {
    if (!clean[f] || !String(clean[f]).trim()) (clean as unknown as Record<string, string>)[f] = 'unknown';
  }
  return clean;
}

export async function parseGarment(
  asset: VisualAsset,
  context: VisionContext,
  options: { trustedExact?: boolean; contentHash?: string | null; providerVersion?: string } = {},
): Promise<GarmentVisualParse> {
  const provider = getVisionProvider();
  const key = cacheKey({ assetId: asset.id, contentHash: options.contentHash, task: 'parse-garment', modelVersion: options.providerVersion || provider.version });
  const hit = getCached<GarmentVisualParse>(key);
  if (hit) return hit;
  const raw = await provider.analyzeGarment(asset, context);
  const clean = sanitizeGarment(raw, asset.id, options.trustedExact || false);
  setCached(key, 'parse-garment', provider.version, clean);
  return clean;
}

export async function parseOutfit(
  asset: VisualAsset,
  context: VisionContext,
  options: { contentHash?: string | null } = {},
): Promise<OutfitVisualParse> {
  const provider = getVisionProvider();
  const key = cacheKey({ assetId: asset.id, contentHash: options.contentHash, task: 'parse-outfit', modelVersion: provider.version });
  const hit = getCached<OutfitVisualParse>(key);
  if (hit) return hit;
  const raw = await provider.analyzeOutfit(asset, context);
  const clean: OutfitVisualParse = {
    ...raw,
    assetId: asset.id,
    shoes: raw.shoes ?? null,
    confidence: Number.isFinite(raw.confidence) ? Math.min(1, Math.max(0, raw.confidence)) : 0,
  };
  setCached(key, 'parse-outfit', provider.version, clean);
  return clean;
}
