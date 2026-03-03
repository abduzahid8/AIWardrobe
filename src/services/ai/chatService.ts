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
            const response = await axios.post(
                `${API_URL}/ai-chat`,
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

export function generateLocalChatResponse(message: string): ChatResponse {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('date')) {
        return {
            text: "For a date night, I'd recommend something that makes you feel confident! A nice blouse with tailored jeans works great. Add a statement accessory and you're set. What's the vibe - casual or fancy?",
            suggestions: ['Casual date outfit', 'Fancy dinner look', 'Coffee date style'],
        };
    }
    if (lowerMessage.includes('work') || lowerMessage.includes('interview')) {
        return {
            text: "For professional settings, stick to classic pieces. A well-fitted blazer, crisp shirt, and tailored pants in neutral colors always work. Would you like specific color recommendations?",
            suggestions: ['Business casual tips', 'Interview outfit help', 'Work wardrobe basics'],
        };
    }
    if (lowerMessage.includes('party') || lowerMessage.includes('event')) {
        return {
            text: "Time to shine! For parties, don't be afraid to go bold with colors or textures. A sequin top, statement earrings, or a fabulous dress can make you stand out. What's the dress code?",
            suggestions: ['Cocktail party look', 'Birthday outfit', 'Club night style'],
        };
    }

    return {
        text: "I'm here to help with your style! Tell me about the occasion, your preferences, or ask for outfit ideas. I can suggest complete looks based on what you're doing!",
        suggestions: ['Date night outfit', 'Work attire help', 'Weekend casual look'],
    };
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
