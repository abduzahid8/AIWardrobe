/**
 * AI Service — Backward-compatible re-export.
 *
 * The monolithic AIService class has been split into domain services:
 *   - ai/outfitService.ts — Outfit generation + weather recommendations
 *   - ai/chatService.ts   — Chat + semantic search
 *   - ai/scanService.ts   — Clothing analysis + video scanning + VTON
 *   - ai/healthService.ts  — Server + AliceVision health checks
 *
 * New code should import from the domain services directly.
 * This file maintains backward compatibility for existing imports.
 */

export { AIService, aiServiceInstance as aiService } from './ai';
export type {
    AIOutfitSuggestion,
    AIAnalysisResult,
    VirtualTryOnResult,
    ChatMessage,
    ChatResponse,
    VideoAnalysisResult,
    DetectedClothingItem,
    OutfitGroup,
    OutfitRecommendation,
    RecommendedItem,
    StylistChatResponse,
    WardrobeSearchResult,
} from './ai/types';

export default new (require('./ai').AIService)();
