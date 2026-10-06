/**
 * Chat Service — AI chat and semantic search functionality.
 *
 * Handles:
 * - Server-backed AI chat with fallback
 * - Local chat response generation
 * - AliceVision stylist chat
 * - Semantic wardrobe search
 */

import axios from 'axios';
import Config from '../../config/env';
import { withRetry, getAuthHeaders } from './shared';
import type {
    ChatMessage,
    ChatResponse,
    DetectedClothingItem,
    StylistChatResponse,
    WardrobeSearchResult,
} from './types';

const API_URL = Config.api.url;
const ALICEVISION_URL = Config.api.alicevisionUrl;
const TIMEOUT_MS = 60000;

// ── Public API ──

export async function sendChatMessage(
    message: string,
    conversationHistory: ChatMessage[] = [],
    stylePreference?: string
): Promise<ChatResponse> {
    return withRetry(async () => {
        try {
            // Backend mounts the chat router under /api (see backend/api/index.js).
            const response = await axios.post(
                `${API_URL}/api/ai-chat`,
                {
                    query: message,
                    conversationHistory,
                    stylePreference,
                },
                {
                    headers: await getAuthHeaders(),
                    timeout: TIMEOUT_MS,
                }
            );

            if (response.data.text) {
                return {
                    text: response.data.text,
                    suggestions: response.data.suggestions,
                };
            }
            throw new Error('No response from AI');
        } catch {
            return generateLocalChatResponse(message);
        }
    });
}

export function generateLocalChatResponse(message: string, stylePreference?: string): ChatResponse {
    // Single voice: stylistPersona is the source of truth (offline-safe).
    const { personaReply } = require('./stylistPersona') as typeof import('./stylistPersona');
    return personaReply(message, stylePreference);
}

export async function chatWithStylist(
    message: string,
    wardrobeItems?: DetectedClothingItem[],
    conversationHistory?: { role: string; content: string }[]
): Promise<StylistChatResponse> {
    return withRetry(async () => {
        const response = await axios.post(
            `${ALICEVISION_URL}/outfit/chat`,
            {
                message,
                wardrobe_items: wardrobeItems?.map(item => ({
                    id: item.cutoutImage?.slice(0, 20) || Math.random().toString(36),
                    category: item.category,
                    specificType: item.specificType,
                    primaryColor: item.primaryColor,
                    colorHex: item.colorHex,
                })),
                conversation_history: conversationHistory,
            },
            {
                headers: await getAuthHeaders(),
                timeout: 30000,
            }
        );

        return response.data;
    });
}

export async function searchWardrobe(
    query: string,
    wardrobeItems: DetectedClothingItem[],
    topK: number = 5
): Promise<WardrobeSearchResult> {
    return withRetry(async () => {
        const response = await axios.post(
            `${ALICEVISION_URL}/wardrobe/search`,
            {
                query,
                wardrobe_items: wardrobeItems.map(item => ({
                    id: item.cutoutImage?.slice(0, 20) || Math.random().toString(36),
                    category: item.category,
                    specificType: item.specificType,
                    primaryColor: item.primaryColor,
                    colorHex: item.colorHex,
                    material: item.material,
                    pattern: item.pattern,
                })),
                top_k: topK,
            },
            {
                headers: await getAuthHeaders(),
                timeout: 15000,
            }
        );

        return response.data;
    });
}
