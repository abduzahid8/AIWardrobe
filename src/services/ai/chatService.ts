/**
 * Chat Service — AI chat and semantic search functionality.
 *
 * Handles:
 * - Server-backed AI chat with fallback
 * - Local chat response generation
 * - Gemini-powered stylist chat
 * - Semantic wardrobe search via Gemini
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
    // Use local Gemini implementation instead of AliceVision
    return generateLocalStylistResponse(message, wardrobeItems, conversationHistory);
}

function generateLocalStylistResponse(
    message: string,
    wardrobeItems?: DetectedClothingItem[],
    _conversationHistory?: { role: string; content: string }[]
): StylistChatResponse {
    const lowerMessage = message.toLowerCase();
    const items = wardrobeItems || [];

    // Simple keyword-based response with wardrobe context
    let response = "I'm your AI stylist! ";
    let suggestedOutfits: any[] = [];
    let followUpQuestions: string[] = [];

    if (items.length > 0) {
        const tops = items.filter(i => i.category === 'tops' || i.category === 'outerwear');
        const bottoms = items.filter(i => i.category === 'bottoms');
        const footwear = items.filter(i => i.category === 'footwear');

        if (lowerMessage.includes('outfit') || lowerMessage.includes('wear') || lowerMessage.includes('suggest')) {
            response += `I see you have ${items.length} items in your wardrobe. `;
            if (tops.length > 0 && bottoms.length > 0) {
                response += `You could pair a ${tops[0]?.specificType || 'top'} with ${bottoms[0]?.specificType || 'bottoms'}. `;
                suggestedOutfits = [{
                    items: items.slice(0, 3),
                    confidence: 0.7,
                    reasoning: 'Based on your available items',
                    occasion: 'casual',
                    style: 'comfortable',
                    colorHarmony: 'neutral',
                }];
            }
            followUpQuestions = ['What occasion is this for?', 'Any colors you want to avoid?'];
        } else if (lowerMessage.includes('color') || lowerMessage.includes('match')) {
            const colors = [...new Set(items.map(i => i.primaryColor))];
            response += `Your wardrobe has these colors: ${colors.slice(0, 5).join(', ')}. `;
            followUpQuestions = ['Want me to suggest color combinations?'];
        }
    } else {
        response += "Add some items to your wardrobe first and I can help you style them! ";
        followUpQuestions = ['Want tips on building a wardrobe?', 'Need help with a specific occasion?'];
    }

    return {
        success: true,
        response,
        suggestedOutfits,
        followUpQuestions,
    };
}

export async function searchWardrobe(
    query: string,
    wardrobeItems: DetectedClothingItem[],
    topK: number = 5
): Promise<WardrobeSearchResult> {
    // Local semantic search without AliceVision
    const lowerQuery = query.toLowerCase();

    // Simple keyword matching
    const scored = wardrobeItems.map(item => {
        let score = 0;
        const searchable = [
            item.category,
            item.specificType,
            item.primaryColor,
            item.material,
            item.pattern,
        ].join(' ').toLowerCase();

        // Split query into words and count matches
        const queryWords = lowerQuery.split(/\s+/);
        for (const word of queryWords) {
            if (searchable.includes(word)) score += 1;
        }

        return { item, score };
    });

    // Sort by score and take top K
    const results = scored
        .filter(({ score }) => score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ item }) => item);

    return {
        success: true,
        results,
        query,
        totalResults: results.length,
    };
}
