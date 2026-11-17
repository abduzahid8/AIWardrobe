/**
 * closetScan.ts — Closet scan queue + garment regions (4F, 4G).
 *
 * Batch-friendly, resumable, offline-safe: scan proceeds on structured
 * metadata when no provider is active. Regions support future
 * segmentation; without a provider, manual-crop fallback (never
 * fabricated masks). User corrections preserved on the queue item.
 */
import type { GarmentRegion, ScanQueueItem, ScanQueueStatus, VisualAsset } from '../../types/visualOS';

let seq = 0;

export function enqueueScan(userId: string, assetId: string): ScanQueueItem {
  return {
    id: `scan_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    userId, assetId, status: 'queued', attempts: 0, updatedAt: new Date().toISOString(),
  };
}

export function advanceScan(
  item: ScanQueueItem,
  to: ScanQueueStatus,
  patch: Partial<Pick<ScanQueueItem, 'proposedCategory' | 'proposedColor' | 'proposedType' | 'duplicateOfItemId'>> = {},
): ScanQueueItem {
  const order: ScanQueueStatus[] = ['queued', 'quality_check', 'awaiting_confirm', 'duplicate_review', 'saved'];
  if (to === 'skipped' || to === 'failed') return { ...item, ...patch, status: to, updatedAt: new Date().toISOString() };
  const fromIdx = order.indexOf(item.status);
  const toIdx = order.indexOf(to);
  if (toIdx < 0 || (item.status !== 'duplicate_review' && toIdx !== fromIdx + 1 && to !== item.status)) {
    return item; // illegal jump rejected; resumable from current state
  }
  return { ...item, ...patch, status: to, attempts: item.attempts + 1, updatedAt: new Date().toISOString() };
}

export function pendingScans(queue: readonly ScanQueueItem[]): ScanQueueItem[] {
  return queue.filter((q) => !['saved', 'skipped', 'failed'].includes(q.status));
}

/** Manual-crop fallback region (4G): caller supplies the box; never invented. */
export function manualRegion(
  parent: VisualAsset,
  bbox: GarmentRegion['bbox'],
  proposedType?: string,
): GarmentRegion {
  return {
    id: `gr_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    parentAssetId: parent.id,
    bbox: { ...bbox },
    confidence: 0.5,
    proposedType: proposedType ?? null,
    cropAssetId: null,
    manual: true,
  };
}

/** Provider-proposed region with confidence; low-confidence rows flagged. */
export function providerRegion(
  parentAssetId: string,
  bbox: GarmentRegion['bbox'],
  confidence: number,
  proposedType?: string,
): GarmentRegion {
  return {
    id: `gr_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    parentAssetId,
    bbox: { ...bbox },
    confidence: Math.min(1, Math.max(0, confidence)),
    proposedType: proposedType ?? null,
    cropAssetId: null,
    manual: false,
  };
}
