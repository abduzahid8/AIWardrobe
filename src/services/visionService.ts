/**
 * Vision Service — AliceVision AI pipeline client
 *
 * Handles all communication with the Python AliceVision microservice.
 * Responsibilities:
 *   - Video frame analysis
 *   - Clothing segmentation
 *   - Attribute extraction
 *   - Ensemble detection
 *
 * This is the ONLY file that talks to the AliceVision service.
 */

import axios from 'axios';
import Config from '../config/env';

const ALICEVISION_URL = Config.api.alicevisionUrl;
const TIMEOUT_MS = 60000;
const VIDEO_TIMEOUT_MS = 180000;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

// ============================================
// TYPES
// ============================================

export interface DetectedClothingItem {
    category: string;
    specificType: string;
    confidence: number;
    primaryColor: string;
    colorHex: string;
    material?: string;
    pattern?: string;
    cutoutImage?: string;
    attributes?: Record<string, string | number | boolean>;
}

export interface VideoAnalysisResult {
    success: boolean;
    items: DetectedClothingItem[];
    outfits: Array<{
        outfitId: number;
        items: DetectedClothingItem[];
        timestamp?: number;
    }>;
    processingTimeMs: number;
}

export interface SegmentationResult {
    success: boolean;
    items: DetectedClothingItem[];
    framesAnalyzed: number;
    strategy: string;
}

// ============================================
// RETRY UTILITY
// ============================================

async function withRetry<T>(
    fn: () => Promise<T>,
    retries: number = MAX_RETRIES,
    delay: number = RETRY_DELAY_MS
): Promise<T> {
    try {
        return await fn();
    } catch (error) {
        if (retries > 0) {
            console.log(`[VisionService] Retrying... (${MAX_RETRIES - retries + 1}/${MAX_RETRIES})`);
            await new Promise((resolve) => setTimeout(resolve, delay));
            return withRetry(fn, retries - 1, delay * 1.5);
        }
        throw error;
    }
}

// ============================================
// SERVICE
// ============================================

class VisionService {
    /**
     * Analyze video frames with temporal ensemble for consistent detection
     */
    async analyzeVideoFrames(
        frames: string[],
        options?: { detectOutfitChanges?: boolean; minAgreement?: number }
    ): Promise<VideoAnalysisResult> {
        return withRetry(async () => {
            const response = await axios.post(
                `${ALICEVISION_URL}/analyze-video-timeline`,
                {
                    frames,
                    detect_outfit_changes: options?.detectOutfitChanges ?? true,
                    min_agreement: options?.minAgreement ?? 0.5,
                },
                { timeout: VIDEO_TIMEOUT_MS }
            );
            return response.data;
        });
    }

    /**
     * Ensemble detection for maximum accuracy on single image
     */
    async detectClothingEnsemble(imageBase64: string): Promise<{
        success: boolean;
        items: DetectedClothingItem[];
        processingTimeMs: number;
        modelsUsed: string[];
    }> {
        return withRetry(async () => {
            const response = await axios.post(
                `${ALICEVISION_URL}/detect-ensemble`,
                { image: imageBase64 },
                { timeout: 90000 }
            );
            return response.data;
        });
    }

    /**
     * Multi-frame segmentation with voting for video clips
     */
    async segmentMultiFrame(
        frames: string[],
        minAgreement: number = 0.5
    ): Promise<SegmentationResult> {
        return withRetry(async () => {
            const response = await axios.post(
                `${ALICEVISION_URL}/segment-multi-frame`,
                { frames, min_agreement: minAgreement },
                { timeout: 120000 }
            );
            return response.data;
        });
    }

    /**
     * Segment all clothing items in a single image
     */
    async segmentAll(imageBase64: string): Promise<{
        success: boolean;
        totalItems: number;
        items: DetectedClothingItem[];
        processingTimeMs: number;
    }> {
        return withRetry(async () => {
            const response = await axios.post(
                `${ALICEVISION_URL}/segment-all`,
                { image: imageBase64 },
                { timeout: TIMEOUT_MS }
            );
            return response.data;
        });
    }

    /**
     * Health check for AliceVision service
     */
    async checkHealth(): Promise<{
        healthy: boolean;
        features: string[];
        message: string;
    }> {
        try {
            const response = await axios.get(`${ALICEVISION_URL}/health`, { timeout: 5000 });
            return {
                healthy: true,
                features: response.data.features || [],
                message: 'AliceVision AI service is running',
            };
        } catch {
            return {
                healthy: false,
                features: [],
                message: 'AliceVision AI service is unavailable',
            };
        }
    }
}

export const visionService = new VisionService();
export default visionService;
