/**
 * AI Service Types — Shared type definitions for all AI domain services.
 *
 * Single source of truth for AI-related interfaces used across
 * outfitService, chatService, scanService, and healthService.
 */

// ── Outfit Types ──

export interface AIOutfitSuggestion {
    id: string;
    description: string;
    occasion: string;
    confidence: number;
    items: {
        type: string;
        color: string;
        recommendation: string;
    }[];
    stylingTips: string[];
}

// ── Analysis Types ──

export interface AIAnalysisResult {
    itemType: string;
    color: string;
    style: string;
    description: string;
    confidence: number;
    material?: string;
    brand?: string;
    season?: string;
    tags?: string[];
}

// ── Try-On Types ──

export interface VirtualTryOnResult {
    success: boolean;
    imageUrl: string;
    processingTime: number;
}

// ── Chat Types ──

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface ChatResponse {
    text: string;
    suggestions?: string[];
}

// ── Video Analysis Types ──

export interface VideoAnalysisResult {
    success: boolean;
    items: DetectedClothingItem[];
    outfits: OutfitGroup[];
    processingTimeMs: number;
}

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

export interface OutfitGroup {
    outfitId: number;
    items: DetectedClothingItem[];
    timestamp?: number;
}

// ── Outfit Recommendation Types ──

export interface OutfitRecommendation {
    items: RecommendedItem[];
    confidence: number;
    reasoning: string;
    occasion: string;
    style: string;
    colorHarmony: string;
}

export interface RecommendedItem {
    id: string;
    category: string;
    specificType: string;
    primaryColor: string;
    colorHex: string;
    styleTags: string[];
}

// ── Stylist Chat Types ──

export interface StylistChatResponse {
    success: boolean;
    response: string;
    suggestedOutfits: OutfitRecommendation[];
    followUpQuestions: string[];
}

// ── Semantic Search Types ──

export interface WardrobeSearchResult {
    success: boolean;
    results: DetectedClothingItem[];
    query: string;
    totalResults: number;
}
