/**
 * LLM Service — Text generation via GPT/Gemini
 *
 * Handles all LLM API calls. Used for:
 *   - AI chat stylist (conversational)
 *   - Outfit reasoning explanations (cosmetic text)
 *   - Style tips generation
 *
 * This is COSMETIC AI — it generates nice text, not core logic.
 * Core outfit scoring lives in suggestionEngine.ts (rule-based).
 */

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from '../config/env';

const API_URL = Config.api.url;
const TIMEOUT_MS = 30000;
const MAX_RETRIES = 2;

// ============================================
// TYPES
// ============================================

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface ChatResponse {
    text: string;
    suggestions?: string[];
}

export interface StylistChatResponse {
    success: boolean;
    response: string;
    followUpQuestions: string[];
}

// ============================================
// RETRY
// ============================================

async function withRetry<T>(
    fn: () => Promise<T>,
    retries: number = MAX_RETRIES
): Promise<T> {
    try {
        return await fn();
    } catch (error) {
        if (retries > 0) {
            await new Promise((r) => setTimeout(r, 1500));
            return withRetry(fn, retries - 1);
        }
        throw error;
    }
}

// ============================================
// LOCAL FALLBACKS
// ============================================

function generateLocalChatResponse(message: string, style?: string): ChatResponse {
    // Single voice: stylistPersona is the source of truth (offline-safe).
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { personaReply } = require('./ai/stylistPersona') as typeof import('./ai/stylistPersona');
    return personaReply(message, style);
}

// ============================================
// SERVICE
// ============================================

class LLMService {
    private userToken: string | null = null;

    async initialize(): Promise<void> {
        this.userToken = await AsyncStorage.getItem('userToken');
    }

    private getHeaders(): Record<string, string> {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (this.userToken) {
            headers['Authorization'] = `Bearer ${this.userToken}`;
        }
        return headers;
    }

    /**
     * Send a chat message to the AI stylist
     */
    async chat(
        message: string,
        conversationHistory: ChatMessage[] = [],
        stylePreference?: string
    ): Promise<ChatResponse> {
        return withRetry(async () => {
            try {
                const response = await axios.post(
                    `${API_URL}/ai-chat`,
                    { query: message, conversationHistory, stylePreference },
                    { headers: this.getHeaders(), timeout: TIMEOUT_MS }
                );

                if (response.data.text) {
                    return {
                        text: response.data.text,
                        suggestions: response.data.suggestions,
                    };
                }
                throw new Error('No response from AI');
            } catch {
                return generateLocalChatResponse(message, stylePreference);
            }
        });
    }

    /**
     * Generate a natural-language explanation for an outfit suggestion.
     * Used to enrich the reasoning from suggestionEngine.
     */
    async explainOutfit(
        outfitDescription: string,
        weather?: { temp: number; condition: string },
        occasion?: string
    ): Promise<string> {
        try {
            const response = await axios.post(
                `${API_URL}/ai-chat`,
                {
                    query: `Explain in one sentence why this outfit works: ${outfitDescription}. Weather: ${weather?.temp || 'unknown'}°C ${weather?.condition || ''}. Occasion: ${occasion || 'casual'}.`,
                    conversationHistory: [],
                },
                { headers: this.getHeaders(), timeout: 15000 }
            );
            return response.data.text || outfitDescription;
        } catch {
            return outfitDescription;
        }
    }

    /**
     * Check API server health
     */
    async checkHealth(): Promise<{ healthy: boolean; message: string }> {
        try {
            await axios.get(`${API_URL}/health`, { timeout: 5000 });
            return { healthy: true, message: 'Server is running' };
        } catch {
            return { healthy: false, message: 'Server is currently unavailable' };
        }
    }
}

export const llmService = new LLMService();
export default llmService;
