import { findScanReviewBySource } from './scanReviews';
import NetInfo from '@react-native-community/netinfo';
import useUploadQueueStore, { flushUploadQueuePersistence, type PendingUpload } from '../store/uploadQueueStore';
import { resolvePendingScanMedia, removePendingScanMedia, preparePendingScanRemoval, cancelPendingScanRemoval, reconcilePendingScanMedia } from './pendingScanMedia';
import { authEvents } from '../../store/authEvents';
import logger from '../utils/logger';

const MAX_RETRIES = 3;

/**
 * UploadQueueProcessor - Background service to re-process failed/pending uploads.
 * Subscribes to the upload queue and attempts to process entries when online.
 */
class UploadQueueProcessor {
    private isProcessing = false;
    private analysisHook: any = null;
    private hookOwner: string | null = null;
    private netUnsub: (() => void) | null = null;
    private intervalId: ReturnType<typeof setInterval> | null = null;
    private initialized = false;
    private hydrationUnsub: (() => void) | null = null;
    private generation = 0;

    initialize() {
        // Guard against double-init — each previous call leaked a
        // NetInfo listener + 5-min setInterval (timer storm on remount).
        if (this.initialized) return;
        this.initialized = true;
        this.hydrationUnsub = useUploadQueueStore.persist.onFinishHydration(() => {
            if (!this.initialized) return;
            this.recoverInterruptedEntries();
            void this.processQueue();
        });
        if (useUploadQueueStore.persist.hasHydrated()) this.recoverInterruptedEntries();
        logger.info('[QueueProcessor] Initializing...');

        // Listen for network changes
        this.netUnsub = NetInfo.addEventListener(state => {
            if (state.isConnected && state.isInternetReachable) {
                logger.info('[QueueProcessor] Online - checking queue...');
                this.processQueue();
            }
        });

        // Periodic check every 5 minutes as a safety net
        this.intervalId = setInterval(() => this.processQueue(), 5 * 60 * 1000);

        // Initial check
        this.processQueue();
    }

    private recoverInterruptedEntries() {
        const owner = authEvents.getUserId();
        const store = useUploadQueueStore.getState();
        for (const entry of store.pendingUploads) {
            if (owner && entry.ownerId === owner && entry.status === 'processing') store.updateStatus(entry.id, 'pending');
        }
    }

    destroy() {
        this.generation++;
        this.hydrationUnsub?.(); this.hydrationUnsub = null;
        this.recoverInterruptedEntries();
        if (this.netUnsub) {
            try { this.netUnsub(); } catch { /* ignore */ }
            this.netUnsub = null;
        }
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.initialized = false;
        this.analysisHook = null; this.hookOwner = null;
    }

    setAnalysisHook(hook: any) {
        this.analysisHook = hook;
        this.hookOwner = authEvents.getUserId();
    }

    /** Serialize user actions with automatic analysis and acknowledge disk writes first. */
    private async changeEntry(id: string, discard: boolean): Promise<boolean> {
        if (this.isProcessing || !useUploadQueueStore.persist.hasHydrated()) return false;
        const owner = authEvents.getUserId(), revision = authEvents.getRevision();
        const current = () => !!owner && owner === authEvents.getUserId() && revision === authEvents.getRevision();
        const entries = useUploadQueueStore.getState().pendingUploads;
        const index = entries.findIndex(item => item.id === id && item.ownerId === owner);
        const entry = entries[index];
        if (!owner || !entry || entry.status === 'processing') return false;
        this.isProcessing = true;
        const restore = async (snapshot: PendingUpload) => {
            if (!current()) return;
            useUploadQueueStore.setState(state => {
                const remaining = state.pendingUploads.filter(item => item.id !== snapshot.id || item.ownerId !== owner);
                remaining.splice(Math.min(index, remaining.length), 0, snapshot);
                return { pendingUploads: remaining };
            });
            await flushUploadQueuePersistence();
        };
        try {
            const store = useUploadQueueStore.getState();
            if (discard) {
                await preparePendingScanRemoval(owner, entry.uri);
                if (!current()) { await cancelPendingScanRemoval(owner, entry.uri); return false; }
                store.removeUpload(id);
            }
            else if (!store.retryUpload(id)) return false;
            await flushUploadQueuePersistence();
            if (!current()) { if (discard) await cancelPendingScanRemoval(owner, entry.uri); return false; }
            if (discard) await removePendingScanMedia(owner, entry.uri, () => useUploadQueueStore.getState().pendingUploads.filter(item => item.ownerId === owner).map(item => item.uri));
            return current();
        } catch (error) {
            // Retain the recoverable record on failed acknowledgement/removal.
            try {
                await restore(entry);
                if (discard) await cancelPendingScanRemoval(owner, entry.uri);
            } catch { /* Keep intent on disk when rollback cannot be acknowledged. */ }
            throw error;
        } finally { this.isProcessing = false; }
    }

    async retryUpload(id: string): Promise<boolean> {
        const accepted = await this.changeEntry(id, false);
        if (accepted) await this.processQueue(id);
        return accepted;
    }

    discardUpload(id: string): Promise<boolean> { return this.changeEntry(id, true); }

    async processQueue(manualRetryId?: string) {
        if (this.isProcessing || !useUploadQueueStore.persist.hasHydrated()) return;
        const owner = authEvents.getUserId();
        const revision = authEvents.getRevision();
        const generation = this.generation;
        if (!owner) return;
        const current = () => generation === this.generation && owner === authEvents.getUserId() && revision === authEvents.getRevision();
        this.isProcessing = true;
        try {
        
        await flushUploadQueuePersistence();
        if (!current()) return;
        await reconcilePendingScanMedia(owner, () => useUploadQueueStore.getState().pendingUploads.filter(entry => entry.ownerId === owner).map(entry => entry.uri));
        if (!current()) return;
        const state = await NetInfo.fetch();
        if (!current()) return;
        const online = state.isConnected && state.isInternetReachable !== false;

        const { pendingUploads, updateStatus, incrementRetry, removeUpload } = useUploadQueueStore.getState();
        // Share the persisted cooldown across this owner's captures. A provider
        // outage must not shift immediately from one failed photo to the next.
        const ownerRetryAt = pendingUploads.reduce((latest, entry) => entry.ownerId === owner &&
            Number.isFinite(entry.nextRetryAt) ? Math.max(latest, entry.nextRetryAt ?? 0) : latest, 0);
        const toProcess = pendingUploads.filter(u => u.ownerId === owner && (u.status === 'pending' || u.status === 'failed') );

        if (toProcess.length === 0) return;

        logger.info(`[QueueProcessor] Processing ${toProcess.length} items...`);

        for (const upload of toProcess) {
            if (!current()) break;
            try {
                const savedReview = await findScanReviewBySource(owner, upload.id);
                if (!current()) break;
                if (!savedReview && (upload.retryCount >= MAX_RETRIES || (upload.id !== manualRetryId && Math.max(ownerRetryAt, upload.nextRetryAt ?? 0) > Date.now()) || !online || this.analysisHook?.canProcess?.() === false)) continue;
                updateStatus(upload.id, 'processing');

                // We need the hook to actually do the analysis.
                // If not set, leave the item as pending and STOP the loop —
                // previously every item was reset to pending in a tight loop,
                // causing a hot spin + repeated store writes on every tick.
                if (!this.analysisHook || this.hookOwner !== owner) {
                    logger.warn('[QueueProcessor] Analysis hook not yet available, skipping...');
                    updateStatus(upload.id, 'pending');
                    break;
                }

                logger.info(`[QueueProcessor] Analyzing ${upload.type}`);
                
                const hook = this.analysisHook;
                const uri = resolvePendingScanMedia(upload.uri, owner);
                let succeeded: unknown;
                if (savedReview) {
                    if (savedReview.detectedItems.length && hook.canProcess?.() !== false) hook.restoreReview?.(savedReview);
                    succeeded = true;
                } else if (upload.type === 'video') {
                    succeeded = await hook.analyzeVideo(uri, { queueOnFailure: false, silent: true, queueId: upload.id });
                } else {
                    succeeded = await hook.analyzeImage(uri, { queueOnFailure: false, silent: true, queueId: upload.id });
                }

                if (!current()) break;
                if (succeeded !== true) throw new Error('analysis_not_confirmed');

                // If analyze succeeds, it usually sets results in the hook state.
                // For the background processor, we mainly want to clear the queue once processed.
                await preparePendingScanRemoval(owner, upload.uri);
                if (!current()) { await cancelPendingScanRemoval(owner, upload.uri); break; }
                removeUpload(upload.id);
                try { await flushUploadQueuePersistence(); }
                catch (error) {
                    if (current()) {
                        useUploadQueueStore.setState(state => ({ pendingUploads: [...state.pendingUploads.filter(entry => entry.id !== upload.id || entry.ownerId !== owner), upload] }));
                        try { await flushUploadQueuePersistence(); await cancelPendingScanRemoval(owner, upload.uri); } catch { /* Reconcile intent on restart. */ }
                    }
                    throw error;
                }
                if (!current()) { await cancelPendingScanRemoval(owner, upload.uri); break; }
                await removePendingScanMedia(owner, upload.uri, () => useUploadQueueStore.getState().pendingUploads.filter(item => item.ownerId === owner).map(item => item.uri));
                logger.info(`[QueueProcessor] Successfully processed ${upload.id}`);
                break; // Preserve this result for review before trying another queued scan.
            } catch (error: any) {
                if (!current()) break;
                logger.error(`[QueueProcessor] Failed to process ${upload.id}:`, error.message);

                // Increment retry count first, then decide the next status.
                // Re-read the latest state after incrementing to get the updated retryCount.
                incrementRetry(upload.id);
                const newRetryCount = (upload.retryCount ?? 0) + 1;

                if (newRetryCount >= MAX_RETRIES) {
                    // Retries exhausted — permanently mark as failed so the user can see it.
                    updateStatus(upload.id, 'failed', error.message);
                    logger.warn(`[QueueProcessor] Item ${upload.id} exhausted ${MAX_RETRIES} retries, marked failed.`);
                } else {
                    // Transient error — reset to pending so the queue can retry on next run.
                    updateStatus(upload.id, 'pending', error.message);
                    logger.info(`[QueueProcessor] Item ${upload.id} reset to pending (attempt ${newRetryCount}/${MAX_RETRIES}).`);
                }
                await flushUploadQueuePersistence();
                break; // One failure per pass: do not cascade through every saved capture during an outage.
            }
        }

        } catch {
            logger.warn('[QueueProcessor] Queue check failed; entries retained');
        } finally { this.isProcessing = false; }
    }
}

export const uploadQueueProcessor = new UploadQueueProcessor();
export default uploadQueueProcessor;
