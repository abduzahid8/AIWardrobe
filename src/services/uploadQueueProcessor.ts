import NetInfo from '@react-native-community/netinfo';
import useUploadQueueStore from '../store/uploadQueueStore';
import { useVideoAnalysis } from '../features/wardrobe/useVideoAnalysis';
import logger from '../utils/logger';

/**
 * UploadQueueProcessor - Background service to re-process failed/pending uploads.
 * Subscribes to the upload queue and attempts to process entries when online.
 */
class UploadQueueProcessor {
    private isProcessing = false;
    private analysisHook: any = null;

    initialize() {
        logger.info('[QueueProcessor] Initializing...');
        
        // Listen for network changes
        NetInfo.addEventListener(state => {
            if (state.isConnected && state.isInternetReachable) {
                logger.info('[QueueProcessor] Online - checking queue...');
                this.processQueue();
            }
        });

        // Periodic check every 5 minutes as a safety net
        setInterval(() => this.processQueue(), 5 * 60 * 1000);
        
        // Initial check
        this.processQueue();
    }

    setAnalysisHook(hook: any) {
        this.analysisHook = hook;
    }

    async processQueue() {
        if (this.isProcessing) return;
        
        const state = await NetInfo.fetch();
        if (!state.isConnected) return;

        const { pendingUploads, updateStatus, removeUpload } = useUploadQueueStore.getState();
        const toProcess = pendingUploads.filter(u => u.status === 'pending' || u.status === 'failed');

        if (toProcess.length === 0) return;

        this.isProcessing = true;
        logger.info(`[QueueProcessor] Processing ${toProcess.length} items...`);

        for (const upload of toProcess) {
            try {
                updateStatus(upload.id, 'processing');
                
                // We need the hook to actually do the analysis
                // If not set, we skip for now (it will be set when the app mounts the hook)
                if (!this.analysisHook) {
                    logger.warn('[QueueProcessor] Analysis hook not yet available, skipping...');
                    updateStatus(upload.id, 'pending');
                    continue;
                }

                logger.info(`[QueueProcessor] Analyzing ${upload.type}: ${upload.uri}`);
                
                if (upload.type === 'video') {
                    await this.analysisHook.analyzeVideo(upload.uri);
                } else {
                    await this.analysisHook.analyzeImage(upload.uri);
                }

                // If analyze succeeds, it usually sets results in the hook state.
                // For the background processor, we mainly want to clear the queue once processed.
                removeUpload(upload.id);
                logger.info(`[QueueProcessor] Successfully processed ${upload.id}`);
            } catch (error: any) {
                logger.error(`[QueueProcessor] Failed to process ${upload.id}:`, error.message);
                updateStatus(upload.id, 'failed', error.message);
            }
        }

        this.isProcessing = false;
    }
}

export const uploadQueueProcessor = new UploadQueueProcessor();
export default uploadQueueProcessor;
