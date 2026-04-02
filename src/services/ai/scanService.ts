/**
 * Scan Service — Clothing analysis, video scanning, and virtual try-on.
 *
 * Handles:
 * - Single-image clothing analysis with Gemini Vision
 * - Video frame analysis with temporal ensemble
 * - Ensemble detection for max accuracy
 * - Multi-frame segmentation with voting
 * - Virtual try-on via Replicate
 */

import axios from 'axios';
import Config from '../../config/env';
import { aiVisionService } from '../aiVisionService';
import { withRetry, handleAPIError, getAuthHeaders } from './shared';
import type {
    AIAnalysisResult,
    VirtualTryOnResult,
    VideoAnalysisResult,
    DetectedClothingItem,
} from './types';

const API_URL = Config.api.url;
const TIMEOUT_MS = 60000;

// ── Public API ──

export async function analyzeClothing(imageBase64: string): Promise<AIAnalysisResult[]> {
    return withRetry(async () => {
        try {
            const response = await axios.post(
                `${API_URL}/api/analyze-frames`,
                { frames: [imageBase64] },
                {
                    headers: await getAuthHeaders(),
                    timeout: TIMEOUT_MS,
                }
            );

            if (response.data.detectedItems?.length > 0) {
                return response.data.detectedItems;
            }

            return generateLocalAnalysis();
        } catch {
            return generateLocalAnalysis();
        }
    });
}

export function generateLocalAnalysis(): AIAnalysisResult[] {
    return [
        {
            itemType: 'Clothing Item',
            color: 'Unknown',
            style: 'Casual',
            description: 'Detected clothing item',
            confidence: 0.5,
            tags: ['needs-review'],
        },
    ];
}

export async function virtualTryOn(
    humanImageBase64: string,
    garmentImageBase64: string
): Promise<VirtualTryOnResult> {
    const startTime = Date.now();

    return withRetry(async () => {
        try {
            const response = await axios.post(
                `${API_URL}/try-on`,
                {
                    human_image: humanImageBase64,
                    garment_image: garmentImageBase64,
                    description: 'clothing',
                },
                {
                    headers: await getAuthHeaders(),
                    timeout: 120000,
                }
            );

            if (response.data.image) {
                return {
                    success: true,
                    imageUrl: response.data.image,
                    processingTime: Date.now() - startTime,
                };
            }
            throw new Error('No image returned');
        } catch (error) {
            handleAPIError(error, 'Virtual try-on');
        }
    });
}

export async function analyzeVideoFrames(
    frames: string[],
    options?: {
        detectOutfitChanges?: boolean;
        minAgreement?: number;
    }
): Promise<VideoAnalysisResult> {
    return aiVisionService.analyzeVideoFrames(frames, options);
}

export async function detectClothingEnsemble(
    imageBase64: string
): Promise<{
    success: boolean;
    items: DetectedClothingItem[];
    processingTimeMs: number;
    modelsUsed: string[];
}> {
    return aiVisionService.detectClothingEnsemble(imageBase64);
}

export async function segmentMultiFrame(
    frames: string[],
    minAgreement: number = 0.5
): Promise<{
    success: boolean;
    items: DetectedClothingItem[];
    framesAnalyzed: number;
    strategy: string;
}> {
    const result = await aiVisionService.analyzeVideoFrames(frames, { minAgreement });
    return {
        success: result.success,
        items: result.items,
        framesAnalyzed: frames.length,
        strategy: `${aiVisionService.getProvider()}-vision-voting`,
    };
}
