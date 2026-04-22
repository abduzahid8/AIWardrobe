/**
 * AI Services — Barrel export
 *
 * Re-exports all domain services and types for clean imports.
 *
 * Usage:
 *   import { generateOutfitSuggestions } from '../services/ai';
 *   import { sendChatMessage } from '../services/ai';
 *   import type { AIOutfitSuggestion } from '../services/ai';
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
