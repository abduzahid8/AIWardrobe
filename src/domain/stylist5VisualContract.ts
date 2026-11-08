import { applyVisualFashion, visualAdmissionReason } from './visualFashion.ts';
import { describeGarment, prepareGarment, type GarmentEvidence } from './garmentDescriptor.ts';

export interface Stylist5HydrationDetail {
  fashion_intelligence?: {
    version?: number;
    descriptor?: Parameters<typeof applyVisualFashion>[0] & { evidence?: unknown };
  };
  visual_fashion?: GarmentEvidence['visualFashion'];
  visual_source_hash?: string;
  visual_embedding?: readonly number[];
  price_peers?: unknown;
}

export function prepareHydratedStylist5Garment(
  item: GarmentEvidence,
  detail: Stylist5HydrationDetail & { price_peers?: unknown }
): GarmentEvidence | null {
  const hydrated: GarmentEvidence = {
    ...item,
    visualFashion: detail.visual_fashion || item.visualFashion,
    visualSourceHash: detail.visual_source_hash || item.visualSourceHash,
    visualEmbedding: detail.visual_embedding || item.visualEmbedding,
    ...(item as GarmentEvidence & { pricePeers?: unknown }).pricePeers === undefined &&
    detail.price_peers !== undefined
      ? { pricePeers: detail.price_peers }
      : {},
  };
  if (visualAdmissionReason(hydrated)) return null;

  const cached = detail.fashion_intelligence;
  const completeDescriptor =
    cached?.version === 1 &&
    cached.descriptor &&
    Array.isArray(cached.descriptor.evidence)
      ? cached.descriptor
      : undefined;
  const base = completeDescriptor || describeGarment(hydrated);
  return prepareGarment(hydrated, applyVisualFashion(base, hydrated));
}
