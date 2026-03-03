/**
 * AI Services — Barrel export
 *
 * Re-exports all domain services and types for clean imports.
 * Also provides a backward-compatible AIService facade class
 * that delegates to the individual domain services.
 */

// Re-export domain services
export * from './outfitService';
export * from './chatService';
export * from './scanService';
export * from './healthService';

// Re-export all types
export * from './types';

// Re-export shared utilities
export { withRetry, handleAPIError, getAuthHeaders } from './shared';

// ── Backward-compatible facade ──

import * as outfit from './outfitService';
import * as chat from './chatService';
import * as scan from './scanService';
import * as health from './healthService';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { ChatMessage, ChatResponse, AIOutfitSuggestion, VirtualTryOnResult, AIAnalysisResult, VideoAnalysisResult, DetectedClothingItem, OutfitRecommendation, StylistChatResponse, WardrobeSearchResult } from './types';

/**
 * AIService class — backward-compatible facade.
 *
 * New code should import domain services directly:
 *   import { generateOutfitSuggestions } from '../services/ai/outfitService';
 *
 * This class is kept so existing consumers (aiProviderService.ts, tests)
 * continue to work without changes.
 */
export class AIService {
    private userToken: string | null = null;

    async initialize(): Promise<void> {
        this.userToken = await AsyncStorage.getItem('userToken');
    }

    // Outfit domain
    generateOutfitSuggestions = outfit.generateOutfitSuggestions;
    generateLocalOutfitSuggestions = outfit.generateLocalOutfitSuggestions;
    getWeatherBasedOutfit = outfit.getWeatherBasedOutfit;
    getOutfitRecommendations = outfit.getOutfitRecommendations;

    // Chat domain
    sendChatMessage = chat.sendChatMessage;
    generateLocalChatResponse = chat.generateLocalChatResponse;
    chatWithStylist = chat.chatWithStylist;
    searchWardrobe = chat.searchWardrobe;

    // Scan domain
    analyzeClothing = scan.analyzeClothing;
    generateLocalAnalysis = scan.generateLocalAnalysis;
    virtualTryOn = scan.virtualTryOn;
    analyzeVideoFrames = scan.analyzeVideoFrames;
    detectClothingEnsemble = scan.detectClothingEnsemble;
    segmentMultiFrame = scan.segmentMultiFrame;

    // Health domain
    checkServerHealth = health.checkServerHealth;
    checkAliceVisionHealth = health.checkAliceVisionHealth;
}

// Singleton
export const aiServiceInstance = new AIService();
