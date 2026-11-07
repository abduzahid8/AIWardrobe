/**
 * outfitBoard.ts — Digital outfit board + try-on orchestration (4R–4T).
 *
 * Board is UI-independent domain state: add/remove/swap/lock/version/
 * compare/save hooks for diagnostics, BuyCheck, Find Better, VTON.
 * Try-on orchestration marks renders synthetic: they NEVER feed
 * FitEvidence and never claim size correctness. Multi-item plans degrade
 * gracefully per provider capability.
 */
import type { OutfitBoard, TryOnRequest, TryOnResult } from '../../types/visualOS';

let seq = 0;
const nid = (p: string) => `${p}_${Date.now().toString(36)}_${(seq++).toString(36)}`;

export function createBoard(userId: string, itemIds: string[] = []): OutfitBoard {
  const now = new Date().toISOString();
  return {
    id: nid('board'), userId, itemIds: [...itemIds], lockedItemIds: [],
    versions: [{ label: 'v1', itemIds: [...itemIds], createdAt: now }],
    updatedAt: now,
  };
}

export function boardAdd(board: OutfitBoard, itemId: string): OutfitBoard {
  if (board.itemIds.includes(itemId)) return board;
  return touch({ ...board, itemIds: [...board.itemIds, itemId] });
}

export function boardRemove(board: OutfitBoard, itemId: string): OutfitBoard {
  if (board.lockedItemIds.includes(itemId)) return board; // locked pieces stay
  return touch({ ...board, itemIds: board.itemIds.filter((i) => i !== itemId) });
}

export function boardSwap(board: OutfitBoard, fromId: string, toId: string): OutfitBoard {
  if (board.lockedItemIds.includes(fromId) || !board.itemIds.includes(fromId)) return board;
  return touch({ ...board, itemIds: board.itemIds.map((i) => (i === fromId ? toId : i)), alternativesRequestedFor: toId });
}

export function boardLock(board: OutfitBoard, itemId: string): OutfitBoard {
  if (!board.itemIds.includes(itemId)) return board;
  return touch({ ...board, lockedItemIds: [...new Set([...board.lockedItemIds, itemId])] });
}

export function boardSnapshot(board: OutfitBoard, label: string): OutfitBoard {
  return touch({ ...board, versions: [...board.versions, { label, itemIds: [...board.itemIds], createdAt: new Date().toISOString() }].slice(-10) });
}

function touch(b: OutfitBoard): OutfitBoard {
  return { ...b, updatedAt: new Date().toISOString() };
}

// ---------- Try-on orchestration ----------

export const VTON_CAPABILITIES: Record<string, { multiItem: boolean; shoes: boolean; maxItems: number }> = {
  default: { multiItem: false, shoes: false, maxItems: 1 },
};

export function createTryOnRequest(input: {
  personAssetId: string;
  garmentVariantId: string;
  garmentImageAssetId: string;
  category: string;
  preservationConstraints?: string[];
  provider?: string;
  qualityPreset?: TryOnRequest['qualityPreset'];
}): TryOnRequest {
  return {
    id: nid('tryon'),
    personAssetId: input.personAssetId,
    garmentVariantId: input.garmentVariantId,
    garmentImageAssetId: input.garmentImageAssetId,
    category: input.category,
    preservationConstraints: input.preservationConstraints || ['preserve face', 'preserve background'],
    provider: input.provider || 'default',
    qualityPreset: input.qualityPreset || 'standard',
  };
}

export function tryOnUnavailable(request: TryOnRequest, reason: string): TryOnResult {
  return {
    requestId: request.id, outputAssetId: null, provider: request.provider,
    status: 'unsupported', limitations: [reason, 'Showing outfit board without render.'],
    provenance: 'tryon-orchestrator:fallback', synthetic: true,
  };
}

/** Multi-item plan: supported items only, limitation stated (4T). */
export function planMultiItemTryOn(
  requests: readonly TryOnRequest[],
  provider = 'default',
): { planned: TryOnRequest[]; skipped: Array<{ id: string; reason: string }> } {
  const caps = VTON_CAPABILITIES[provider] || VTON_CAPABILITIES.default;
  const planned = requests.slice(0, caps.maxItems);
  const skipped = requests.slice(caps.maxItems).map((r) => ({
    id: r.id,
    reason: `Provider supports max ${caps.maxItems} item(s); "${r.category}" deferred, not faked.`,
  }));
  const shoeSkips = planned.filter((r) => r.category === 'shoes' && !caps.shoes);
  return {
    planned: planned.filter((r) => !(r.category === 'shoes' && !caps.shoes)),
    skipped: [...skipped, ...shoeSkips.map((r) => ({ id: r.id, reason: 'Footwear try-on unsupported by provider.' }))],
  };
}

/** Synthetic renders are visualization only — never fit proof. */
export function syntheticFitDisclaimer(result: TryOnResult): string {
  void result;
  return 'Try-on image is a visualization, not a fit proof. It never feeds FitEvidence and never confirms size correctness.';
}
