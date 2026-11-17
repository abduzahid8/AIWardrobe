/**
 * visualAssets.ts — Asset identity & image provenance (Phase 4B).
 *
 * Distinguishes originals from every transformation. A thumbnail URL is
 * NEVER a product identity: `isProductIdentity` always returns false for
 * derived kinds. Content hashes only where caller supplies them.
 */
import type { VisualAsset, VisualAssetKind } from '../../types/visualOS';

let seq = 0;
const nid = (p: string) => `${p}_${Date.now().toString(36)}_${(seq++).toString(36)}`;

export function createAsset(input: {
  kind: VisualAssetKind;
  originalAssetId?: string | null;
  localUri?: string | null;
  remoteUri?: string | null;
  contentHash?: string | null;
  width?: number | null;
  height?: number | null;
  mimeType?: string | null;
  source: VisualAsset['source'];
  retentionClass?: VisualAsset['retentionClass'];
  userOwned?: boolean;
}): VisualAsset {
  const chain = input.originalAssetId ? [input.originalAssetId, input.kind] : [input.kind];
  return {
    id: nid('vas'),
    kind: input.kind,
    originalAssetId: input.originalAssetId ?? null,
    transformationChain: chain,
    localUri: input.localUri ?? null,
    remoteUri: input.remoteUri ?? null,
    contentHash: input.contentHash ?? null,
    width: input.width ?? null,
    height: input.height ?? null,
    mimeType: input.mimeType ?? null,
    createdAt: new Date().toISOString(),
    source: input.source,
    retentionClass: input.retentionClass ?? (input.kind === 'thumbnail' ? 'session' : 'durable'),
    userOwned: input.userOwned ?? true,
  };
}

export function deriveAsset(
  original: VisualAsset,
  kind: Exclude<VisualAssetKind, 'original_upload' | 'inspiration' | 'merchant_image'>,
  patch: Partial<Pick<VisualAsset, 'localUri' | 'remoteUri' | 'width' | 'height'>> = {},
): VisualAsset {
  return {
    ...createAsset({
      kind,
      originalAssetId: original.originalAssetId || original.id,
      source: original.source,
      userOwned: original.userOwned,
    }),
    transformationChain: [...original.transformationChain, kind],
    localUri: patch.localUri ?? null,
    remoteUri: patch.remoteUri ?? null,
    width: patch.width ?? original.width ?? null,
    height: patch.height ?? original.height ?? null,
    mimeType: original.mimeType,
  };
}

/** Root original for any derived asset. */
export function rootAssetId(asset: VisualAsset): string {
  return asset.originalAssetId || asset.id;
}

/**
 * Derived/transformed assets (thumbnails, crops, renders) must never be
 * used as physical product identity.
 */
export function isProductIdentity(asset: VisualAsset): boolean {
  return asset.kind === 'original_upload' || asset.kind === 'merchant_image';
}

const ALLOWED_MIME = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif']);
export function isSupportedMime(mimeType?: string | null): boolean {
  return !!mimeType && ALLOWED_MIME.has(mimeType.toLowerCase());
}
