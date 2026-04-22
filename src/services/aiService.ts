/**
 * AI Service — Re-exports from domain services.
 *
 * Domain services:
 *   - ai/outfitService.ts — Outfit generation + weather recommendations
 *   - ai/chatService.ts   — Chat + semantic search
 *   - ai/scanService.ts   — Clothing analysis + video scanning + VTON
 *   - ai/healthService.ts  — Server + AliceVision health checks
 *
 * Prefer importing from the domain services directly:
 *   import { generateOutfitSuggestions } from './ai/outfitService';
 */

// Re-export all domain functions, types, and shared utilities
export * from './ai';
